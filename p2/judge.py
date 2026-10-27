"""WP5: single-judge runner for Phase 0 (Claude Opus 5.5, D21): blinding, randomization, V1 audit sample,
seeded hallucinations (~10% of answers, D10), citation verifier, final label assembly."""
import hashlib
import os
import random
import re
import unicodedata

from .common import RUNS, STATE, ENGLISH, NT_IDS, norm_greek, parse_ref, jload, jdump
from .prompts import judge_packet, lxx_ref
from .morph import decode_greek, GK_TENSE, HE_STEM
from . import witnesses as W

JUDGE_MODEL = "claude-opus-5-5"
RUBRIC_VERSION = "r1"
MAX_TOKENS = 6000
AUDIT_FRACTION = 0.15
SEED_RATE = 0.10
K_ALLOWED = {"attribution", "lexical", "grammar", "historical", "lxx", "witness", "other", "crossref", "variant", "occurrence"}

RUBRIC = """You are Judge A in a research study that measures hallucination in original-language Bible exegesis written by AI models. You receive a SOURCE PACKET (machine-readable source data, authoritative) and a list of CLAIMS extracted from one anonymous answer. Label every claim independently. You do not know which model wrote the answer.

Labels:
S = Supported: the packet confirms it, or (only for claim types the packet cannot cover) it is well-established scholarship you are confident of.
C = Contradicted: the packet, or well-established scholarship, says otherwise (wrong text, wrong parsing, wrong direction of a variant, a witness that does not contain the passage, a wrong count, a misattributed view, a wrong verse reference).
U = Unsupported-fabricated: asserts a specific checkable fact (a reading, witness, count, quotation, page, attribution, cross-reference) that you cannot confirm and believe is invented or not established.
O = Opinion: an interpretive or theological judgment, not checkable.
N = Not determinable: checkable in principle, but neither the packet nor confident knowledge decides it.

Citation rule (verified by a script): every S, C or U label must cite evidence.
- Cite packet rows by their tag exactly as printed, e.g. [V:Mark 9:23], [A:Mark 9:24#2], [M:Mark 9:24], [B:Mark 9:24], [L:Genesis 1:1], [W:P75], [O:πιστεύω].
- Quotation and morphology claims MUST cite packet tags; text and parsing in the packet are authoritative (the SBLGNT/NA text stands for the critical text the prompt calls NA28).
- Only for attribution, lexical, grammar, historical, Septuagint, witness-attestation, cross-reference, variant and occurrence claims that the packet does not cover may you cite your own knowledge, written as "K: <basis in a few words>". If you are not confident, use N or U, not S.
Be strict and literal. A hedge ("probably", "I am not certain") does not make a false statement true; label the content. Orthographic slips (accents, movable nu) in an otherwise correct quotation are S. A correct statement attached to the wrong verse is C. Keep "why" under 20 words. Use the record_labels tool and label every claim id exactly once."""

# Output encoding (format only; labeling criteria are the RUBRIC): compact arrays to cut output tokens (J22).
TOOL = {"name": "record_labels",
        "description": ("Record one label per claim as a compact array [id, label, cite, why]. label is S, C, U, O or N. "
                        "cite as required by the citation rule (empty string for O and N). why: at most 10 words, "
                        "only for C and U; empty string otherwise."),
        "input_schema": {"type": "object", "properties": {"labels": {"type": "array", "items": {
            "type": "array", "description": "[id, label, cite, why]"}}}, "required": ["labels"]}}
FORMAT_NOTE = ("\n\nOutput: call record_labels once with labels = [[id, label, cite, why], ...], one entry per claim id; "
               "why (<= 10 words) only for C and U, otherwise an empty string.")


def _norm_label(x):
    if isinstance(x, dict):
        return x
    x = list(x) + [""] * (4 - len(x))
    return {"id": str(x[0]), "label": str(x[1]).strip().upper()[:1], "cite": str(x[2] or ""), "why": str(x[3] or "")}

MAJOR_UNCIALS = ["01", "02", "03", "04", "05", "06", "032", "038", "019", "044", "037"]


def _h(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def witness_rows(p):
    if p["lang"] != "greek":
        return "Manuscript-contents table: not available for the Hebrew Bible in this packet."
    t = W.load()
    lines = ["Manuscript contents (Gregory-Aland; chapters; lacunae not listed). Witnesses not listed are not in this table:"]
    for ga, e in sorted(t.items(), key=lambda x: x[0]):
        books = e.get("books") or {}
        if ga.startswith("P") and p["book"] in books or ga in MAJOR_UNCIALS:
            sig = (" " + e.get("siglum", "")) if e.get("siglum") else ""
            lines.append("[W:%s]%s %s: %s" % (ga, sig, e.get("name", ""), e.get("contents_raw", "")[:160]))
    return "\n".join(lines)


def occurrence_rows(p, corpus):
    lines = ["Occurrence index (lemma tokens; SBLGNT/MorphGNT or WLC/OSHB):"]
    for lem in p["lexemes"]:
        if p["lang"] == "greek":
            occ = corpus.lemma_index["greek"].get(norm_greek(lem), [])
            inbook = [k for k in occ if k[0] == p["book"]]
            lines.append("[O:%s] %s %d in %s (%d verses); %d in the NT" % (lem, lem, len(inbook), ENGLISH[p["book"]],
                                                                      len(set(inbook)), len(occ)))
        else:
            ids = corpus.hebrew_lemma_ids(lem)
            for i in ids[:2]:
                occ = corpus.lemma_index["hebrew"].get(i, [])
                inbook = [k for k in occ if k[0] == p["book"]]
                lines.append("[O:%s] %s (%s) %d in %s; %d in the Hebrew Bible" % (lem, lem, i, len(inbook), ENGLISH[p["book"]], len(occ)))
    return "\n".join(lines)


def full_packet(p, corpus):
    return "\n\n".join([judge_packet(p, corpus), witness_rows(p), occurrence_rows(p, corpus)])


def referenced_verses(claims, p, corpus, limit=25):
    """Text of verses the claims cite outside the packet (dynamic part of the judge prompt)."""
    have = {(p["book"], c, v) for c, v in p["verses"]}
    out, seen = [], set()
    for cl in claims:
        r = parse_ref(cl.get("ref") or "")
        if not r:
            continue
        for k in corpus.verses_in(r)[:4]:
            if k in have or k in seen:
                continue
            seen.add(k)
            if k[0] in NT_IDS:
                txt = corpus.gk_text.get(k, "")
                out.append("[V:%s %d:%d] %s" % (ENGLISH[k[0]], k[1], k[2], txt))
            else:
                txt = corpus.he_text.get(k, "")
                lx = corpus.lxx.get(lxx_ref(*k) or ("", 0, 0))
                out.append("[V:%s %d:%d] %s" % (ENGLISH[k[0]], k[1], k[2], txt))
                if lx:
                    out.append("[L:%s %d:%d] %s" % (ENGLISH[k[0]], k[1], k[2], lx))
            if len(seen) >= limit:
                break
        if len(seen) >= limit:
            break
    return "\n".join(out)


# ------------------------------------------------------------------ seeded hallucinations
_DISTRACT_GK = ["σκότος", "θάλασσα", "ἄρτος", "πλοῖον", "ἀγαπάω", "δένδρον", "χρυσός", "ὄρος"]


def _seed_norm(s):
    s = re.sub(r"[\u2e00-\u2e7f\u0591-\u05af\u05bd\u05c0\u05c3\u05c6\u0387.,;·:!?()\[\]⸀-⸅]", "", s or "")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.lower().split())


def seeded_claims(run_id, p, corpus):
    if _h("seed?", run_id) % 1000 >= SEED_RATE * 1000:
        return []
    n = 1 + _h("seedn", run_id) % 3
    rng = random.Random(_h("seedrng", run_id))
    out = []
    focus = [(p["book"], c, v) for c, v in p["focus"]]
    gens = ["morph", "swapref", "witness" if p["lang"] == "greek" else "lxx"]
    rng.shuffle(gens)
    for g in gens[:n]:
        cl = None
        if g == "morph":
            if p["lang"] == "greek":
                verbs = [(k, w) for k in focus for w in corpus.morph.get(k, []) if w["pos"] == "V-" and w["parse"][1] in "PA"]
                if verbs:
                    k, w = rng.choice(verbs)
                    wrong = w["parse"][0] + ("A" if w["parse"][1] == "P" else "P") + w["parse"][2:]
                    cl = {"type": "morphology", "ref": "%s %d:%d" % (ENGLISH[k[0]], k[1], k[2]), "orig": w["word"],
                          "parse": decode_greek(w["pos"], wrong).replace("verb: ", ""), "lemma": w["lemma"]}
                    cl["claim"] = "In %s, %s is %s." % (cl["ref"], w["word"], cl["parse"])
            else:
                verbs = [(k, w) for k in focus for w in corpus.he_words.get(k, []) if "V" in w["morph"]]
                verbs = [(k, w, s) for k, w in verbs for s in w["morph"][1:].split("/") if s.startswith("V") and len(s) > 2]
                if verbs:
                    k, w, s = rng.choice(verbs)
                    stem = HE_STEM.get(s[1], "qal")
                    wrong = "piel" if stem != "piel" else "hiphil"
                    cl = {"type": "morphology", "ref": "%s %d:%d" % (ENGLISH[k[0]], k[1], k[2]), "orig": w["text"],
                          "parse": "%s %s" % (wrong.capitalize(), "verb"), "lemma": None}
                    cl["claim"] = "In %s, %s is a %s form." % (cl["ref"], w["text"], wrong.capitalize())
        elif g == "swapref":
            k = rng.choice(focus)
            words = (corpus.gk_text.get(k) if p["lang"] == "greek" else corpus.he_text.get(k, "")).split()
            if len(words) >= 4:
                i = rng.randrange(0, len(words) - 3)
                q = " ".join(words[i:i + 3])
                ch = k[1] + 1 if corpus.chapter_exists(k[0], k[1] + 1) else max(1, k[1] - 1)
                vv = 1 + rng.randrange(5)
                ref = "%s %d:%d" % (ENGLISH[k[0]], ch, vv)
                tgt = corpus.gk_text.get((k[0], ch, vv)) if p["lang"] == "greek" else corpus.he_text.get((k[0], ch, vv))
                if tgt is not None and _seed_norm(q) in _seed_norm(tgt):
                    continue  # Stage C fix: the words also occur in the claimed verse, so the seed would be true
                cl = {"type": "quote", "ref": ref, "orig": q, "edition": "critical" if p["lang"] == "greek" else "mt",
                      "claim": "%s reads %s." % (ref, q)}
        elif g == "witness":
            wit = "𝔓46" if p["book"] in ("Matt", "Mark", "Luke", "John", "Acts", "Rev") else "𝔓75"
            k = focus[0]
            cl = {"type": "witness", "ref": "%s %d:%d" % (ENGLISH[k[0]], k[1], k[2]), "witnesses": [wit],
                  "claim": "%s attests the critical-text reading of %s %d:%d." % (wit, ENGLISH[k[0]], k[1], k[2])}
        elif g == "lxx":
            k = rng.choice(focus)
            lx = corpus.lxx.get(lxx_ref(*k) or ("", 0, 0))
            if lx:
                present = set(norm_greek(lx).split())
                cand = [d for d in _DISTRACT_GK if norm_greek(d) not in present]
                ws = [w for w in corpus.he_words.get(k, []) if w["morph"][1:2] in ("N", "V")]
                if cand and ws:
                    d = rng.choice(cand)
                    hw = rng.choice(ws)["text"]
                    cl = {"type": "lxx", "ref": "%s %d:%d" % (ENGLISH[k[0]], k[1], k[2]), "orig": d,
                          "claim": "In %s %d:%d the Septuagint renders %s with %s." % (ENGLISH[k[0]], k[1], k[2], hw, d)}
        if cl:
            cl.update({"id": "seed%d" % (len(out) + 1), "section": 0, "span": "", "seeded": True, "truth": "false",
                       "seed_kind": g})
            for f in ("edition", "parse", "lemma", "count", "count_scope", "variant_op", "variant_tradition"):
                cl.setdefault(f, None)
            cl.setdefault("witnesses", [])
            out.append(cl)
    return out


# ------------------------------------------------------------------ request
def build_request(run_id, p, corpus, claims, v1res, supplement=False):
    """claims: extracted claims; v1res: {id: v1 dict}. Returns (params, key_map, seeds, packet_text, meta).
    supplement=True: judge only the given claims (no audit sample, no seeding) - used after a V1 version change."""
    rng = random.Random(_h("judge" + ("-supp" if supplement else ""), run_id))
    undecided = [c for c in claims if (v1res.get(c["id"]) or {}).get("v1") is None]
    decided = [c for c in claims if (v1res.get(c["id"]) or {}).get("v1") is not None]
    audit = [] if supplement else [c for c in decided if rng.random() < AUDIT_FRACTION]
    seeds = [] if supplement else seeded_claims(run_id, p, corpus)
    items = undecided + audit + seeds
    rng.shuffle(items)
    key_map = {}
    lines = []
    for c in items:  # tolerate malformed extractor rows (e.g. numbers where strings belong)
        for f in ("type", "ref", "orig", "parse", "lemma", "edition", "claim", "span", "count_scope"):
            if c.get(f) is not None and not isinstance(c.get(f), str):
                c[f] = str(c[f])
        if not isinstance(c.get("witnesses"), list):
            c["witnesses"] = [str(c["witnesses"])] if c.get("witnesses") else []
    for i, c in enumerate(items):
        kid = "k%02d" % (i + 1)
        key_map[kid] = c["id"]
        parts = ["%s | type: %s" % (kid, c.get("type"))]
        for f in ("ref", "orig", "parse", "lemma", "edition"):
            if c.get(f):
                parts.append("%s: %s" % (f, c[f]))
        if c.get("witnesses"):
            parts.append("witnesses: %s" % ", ".join(c["witnesses"]))
        if c.get("count") is not None:
            parts.append("count: %s (%s)" % (c["count"], c.get("count_scope")))
        parts.append("claim: %s" % c.get("claim"))
        if c.get("span"):
            parts.append('context: "%s"' % c["span"][:160])
        lines.append(" | ".join(parts))
    packet = full_packet(p, corpus)
    from .web import judge_block
    wb = judge_block(run_id)
    if wb:
        packet = packet + "\n\n" + wb
    extra = referenced_verses(items, p, corpus)
    user = ("Passage under study: %s (focus %s).\n\n%sCLAIMS (%d):\n%s" % (
        p["pericope_ref"], p["focus_ref"],
        ("ADDITIONAL VERSES REFERENCED BY THE CLAIMS:\n%s\n\n" % extra) if extra else "", len(items), "\n".join(lines))
        + FORMAT_NOTE)
    params = {
        "max_tokens": MAX_TOKENS,
        # no cache_control: same-passage judge requests rarely fall inside the 5-minute cache window, so cache writes
        # (1.25x input) cost more than they save (J23)
        "system": [{"type": "text", "text": RUBRIC + "\n\nSOURCE PACKET\n\n" + packet}],
        "messages": [{"role": "user", "content": user}],
        "tools": [TOOL],
        # Opus 5.5 rejects forced tool_choice; the rubric instructs the tool call and api.tool_input falls back to JSON text
        "tool_choice": {"type": "auto"},
    }
    meta = {"n_undecided": len(undecided), "n_audit": len(audit), "n_seeded": len(seeds), "rubric": RUBRIC_VERSION,
            "packet_text_for_verify": None}
    return params, key_map, seeds, packet + "\n" + extra, meta


_TAG = re.compile(r"\[(V|A|M|B|L|W|O|X):[^\]]+\]")


def verify_citation(label, cite, ctype, packet_text):
    if label not in ("S", "C", "U"):
        return True, ""
    cite = cite or ""
    tags = _TAG.findall(cite)
    full = re.findall(r"\[(?:V|A|M|B|L|W|O|X):[^\]]+\]", cite)
    if full:
        missing = [t for t in full if t not in packet_text]
        if missing:
            return False, "tag not in packet: %s" % missing[0]
        return True, ""
    if re.match(r"^\s*K\s*:", cite):
        if ctype in K_ALLOWED:
            return True, ""
        return False, "knowledge citation not allowed for %s" % ctype
    return False, "no verifiable citation"


def assemble(run_id, claims, v1res, judged, key_map, seeds, packet_text):
    """Combine V1 and judge labels into final labels (labels.json)."""
    by_id = {c["id"]: c for c in claims + seeds}
    jl = {}
    for x in judged or []:
        try:
            x = _norm_label(x)
        except Exception:
            continue
        cid = key_map.get(x.get("id"))
        if cid:
            jl[cid] = x
    out = []
    n_cite_fail = n_cite = 0
    for cid, c in by_id.items():
        v = (v1res.get(cid) or {})
        j = jl.get(cid)
        rec = {"id": cid, "type": c.get("type"), "section": c.get("section"), "hedged": c.get("hedged", False),
               "seeded": bool(c.get("seeded")), "v1": v.get("v1"), "v1_rule": v.get("rule"), "v1_why": v.get("why"),
               "judge": j.get("label") if j else None, "judge_cite": j.get("cite") if j else None,
               "judge_why": j.get("why") if j else None}
        if j:
            ok, why = verify_citation(j.get("label"), j.get("cite"), c.get("type"), packet_text)
            if j.get("label") in ("S", "C", "U"):
                n_cite += 1
                n_cite_fail += 0 if ok else 1
            rec["cite_ok"] = ok
            rec["cite_fail"] = why
        if rec["seeded"]:
            rec["final"] = None
            rec["source"] = "seeded"
        elif v.get("v1") in ("S", "C"):
            rec["final"] = v["v1"]
            rec["source"] = "v1"
        elif j:
            rec["final"] = j["label"] if rec.get("cite_ok", True) else "N"
            rec["source"] = "judge" if rec.get("cite_ok", True) else "judge-cite-fail"
        else:
            rec["final"] = "N"
            rec["source"] = "missing"
        rec["hallucinated"] = rec["final"] in ("C", "U")
        rec["checkable"] = rec["final"] in ("S", "C", "U")
        out.append(rec)
    res = {"run_id": run_id, "rubric": RUBRIC_VERSION, "judge_model": JUDGE_MODEL, "labels": out,
           "citation_checked": n_cite, "citation_failed": n_cite_fail}
    jdump(res, os.path.join(RUNS, run_id, "labels.json"))
    return res


def _finalize(rec):
    if rec.get("seeded"):
        rec["final"], rec["source"] = None, "seeded"
    elif rec.get("v1") in ("S", "C"):
        rec["final"], rec["source"] = rec["v1"], "v1"
    elif rec.get("judge"):
        ok = rec.get("cite_ok", True)
        rec["final"] = rec["judge"] if ok else "N"
        rec["source"] = "judge" if ok else "judge-cite-fail"
    else:
        rec["final"], rec["source"] = "N", "missing"
    rec["hallucinated"] = rec["final"] in ("C", "U")
    rec["checkable"] = rec["final"] in ("S", "C", "U")


def refinalize(run_id, v1new):
    """Apply a new V1 result to an already-judged run. Returns ids that now need a judge label."""
    path = os.path.join(RUNS, run_id, "labels.json")
    lab = jload(path)
    need = []
    for rec in lab["labels"]:
        if rec.get("seeded"):
            continue
        v = v1new.get(rec["id"]) or {}
        rec["v1"], rec["v1_rule"], rec["v1_why"] = v.get("v1"), v.get("rule"), v.get("why")
        _finalize(rec)
        if rec["source"] == "missing":
            need.append(rec["id"])
    lab["v1_version_applied"] = True
    jdump(lab, path)
    return need


def merge_supplement(run_id, judged, key_map, packet_text, claims_by_id):
    path = os.path.join(RUNS, run_id, "labels.json")
    lab = jload(path)
    jl = {}
    for x in judged or []:
        try:
            x = _norm_label(x)
        except Exception:
            continue
        cid = key_map.get(x.get("id"))
        if cid:
            jl[cid] = x
    for rec in lab["labels"]:
        j = jl.get(rec["id"])
        if not j:
            continue
        rec["judge"], rec["judge_cite"], rec["judge_why"] = j.get("label"), j.get("cite"), j.get("why")
        ok, why = verify_citation(j.get("label"), j.get("cite"), (claims_by_id.get(rec["id"]) or {}).get("type"), packet_text)
        rec["cite_ok"], rec["cite_fail"], rec["supplement"] = ok, why, True
        if j.get("label") in ("S", "C", "U"):
            lab["citation_checked"] = lab.get("citation_checked", 0) + 1
            lab["citation_failed"] = lab.get("citation_failed", 0) + (0 if ok else 1)
        _finalize(rec)
    jdump(lab, path)
