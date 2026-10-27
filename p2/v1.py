"""WP4: V1 automatic source / morphology checker.

Labels a claim only when the open source packet decides it: 'S' (supported) or 'C' (contradicted).
Otherwise returns None and the claim goes to the judge. V1 labels are ground truth (plan 6.2 step 2),
so every rule is conservative: ambiguity -> None.
"""
import re
from difflib import SequenceMatcher

from .common import NT_IDS, OT_IDS, ENGLISH, norm_greek, norm_hebrew, parse_ref, book_id
from .morph import parse_greek_prose, compare_greek, parse_hebrew_prose, compare_hebrew_verb
from . import witnesses as W
from .prompts import lxx_ref

V1_VERSION = "v1.1"  # 13 Oct: tightened after the judge-vs-V1 audit (DECISIONS J24)

_NEG = re.compile(r"\b(not|never|no|absent|lacks?|without|nowhere)\b", re.I)
_REFS = re.compile(r"\d+:\d+")


def _toks(s, lang):
    return (norm_greek(s) if lang == "greek" else norm_hebrew(s)).split()


def _contains(hay, needle):
    n = len(needle)
    if n == 0:
        return False
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return True
    return False


def _best_window(hay, needle):
    """Best token-level similarity of needle to any window of hay (same length +-1)."""
    best = 0.0
    n = len(needle)
    if not hay or not n:
        return 0.0
    for L in (n - 1, n, n + 1):
        if L <= 0:
            continue
        for i in range(0, max(1, len(hay) - L + 1)):
            r = SequenceMatcher(None, hay[i:i + L], needle).ratio()
            if r > best:
                best = r
    return best


def _orthographic_only(hay, needle):
    """True if needle matches some window token-for-token with every differing token char-similar >= 0.8
    (movable nu, elision, itacism-level spelling), i.e. the same words."""
    n = len(needle)
    for i in range(len(hay) - n + 1):
        w = hay[i:i + n]
        ok = True
        for a, b in zip(w, needle):
            if a != b and SequenceMatcher(None, a, b).ratio() < 0.8:
                ok = False
                break
        if ok:
            return True
    return False


def _segments(orig):
    """Split a quotation on ellipses / brackets into contiguous segments."""
    parts = re.split(r"\.\.\.|…|\[\s*\.\.\.\s*\]", orig or "")
    return [p for p in parts if p.strip()]


class Checker:
    def __init__(self, corpus, passage):
        self.c = corpus
        self.p = passage
        self.lang = passage["lang"]
        self.wit = W.load()

    # --------------------------------------------------------------- helpers
    def _keys(self, ref_s, default="window"):
        r = parse_ref(ref_s) if ref_s else None
        if r is None:
            return None, [(self.p["book"], c, v) for c, v in self.p[default]]
        return r, self.c.verses_in(r)

    def _lang_of_keys(self, keys):
        if not keys:
            return None
        return "greek" if keys[0][0] in NT_IDS else "hebrew"

    def _greek_eds(self, keys):
        crit, na, byz = [], [], []
        for k in keys:
            crit += _toks(self.c.gk_text.get(k, ""), "greek")
            na += _toks(self.c.na_text.get(k, ""), "greek")
            byz += _toks(self.c.rp_text.get(k, ""), "greek")
        return crit, na, byz

    def _hebrew_text(self, keys):
        out = []
        for k in keys:
            out += _toks(self.c.he_text.get(k, ""), "hebrew")
            for kt, q in self.c.he_kq.get(k, []):
                out += _toks(q, "hebrew")
        return out

    # ----------------------------------------------------------------- rules
    def check(self, cl):
        cl = dict(cl)
        for f in ("type", "ref", "orig", "parse", "lemma", "edition", "claim", "count_scope", "variant_op", "variant_tradition"):
            if cl.get(f) is not None and not isinstance(cl[f], str):
                cl[f] = str(cl[f])
        t = (cl.get("type") or "").lower()
        try:
            if t == "witness" or (cl.get("witnesses") and t in ("variant", "witness")):
                r = self.witness(cl)
                if r[0]:
                    return r
            if t == "morphology":
                return self.morphology(cl)
            if t == "variant" and cl.get("variant_op") in ("add", "omit"):
                r = self.variant_addomit(cl)
                if r[0] or r[1].startswith("decided"):
                    return r
            if t in ("quote", "variant", "lxx") and cl.get("orig"):
                return self.quote(cl)
            if t == "occurrence":
                return self.occurrence(cl)
            if t == "crossref":
                return self.crossref(cl)
        except Exception as e:  # never let a checker bug label a claim
            return None, "v1 error: %r" % e, "error"
        return None, "no applicable rule", "none"

    def quote(self, cl):
        orig = cl.get("orig") or ""
        ed = (cl.get("edition") or "unspecified").lower()
        r, keys = self._keys(cl.get("ref"))
        if r is not None and not keys:
            if r["book"] in NT_IDS + OT_IDS and r["v1"] is not None:
                return "C", "reference %s does not exist" % cl.get("ref"), "quote-ref"
            return None, "unparsed ref", "quote"
        lang = self._lang_of_keys(keys)
        if ed == "lxx" or (lang == "hebrew" and orig and re.search(r"[Ͱ-Ͽἀ-῿]", orig)):
            return self.lxx_quote(cl, keys)
        segs = [_toks(s, lang) for s in _segments(orig)]
        segs = [s for s in segs if s]
        if not segs or sum(len(s) for s in segs) < 2:
            return None, "quotation too short", "quote"
        if lang == "greek":
            crit, na, byz = self._greek_eds(keys)
            in_crit = all(_contains(crit, s) or _contains(na, s) for s in segs)
            in_byz = all(_contains(byz, s) for s in segs)
            if ed in ("byzantine", "tr", "majority"):
                if in_byz:
                    return "S", "matches Robinson-Pierpont", "quote"
                if ed == "tr":
                    return None, "TR not in packet", "quote"
                if in_crit:
                    return "C", "critical-text reading presented as Byzantine", "quote-edition"
            else:
                if in_crit:
                    return "S", "matches SBLGNT/NA27", "quote"
                if in_byz:
                    if ed in ("critical", "na28", "na", "ubs"):
                        return "C", "Byzantine (RP) reading presented as the critical text", "quote-edition"
                    # edition unstated and the text matches only RP: what the claim asserts about editions is
                    # unclear -> judge (v1.1)
                    return None, "matches RP only; edition unstated", "quote"
            # not found verbatim: orthographic-only differences are not hallucinations
            if all(_orthographic_only(crit, s) or _orthographic_only(na, s) or _orthographic_only(byz, s) for s in segs):
                return "S", "matches up to orthography", "quote-ortho"
            # quoted correctly but at another verse of the passage?
            allk = [(self.p["book"], c, v) for c, v in self.p["verses"]]
            pc, pn, pb = self._greek_eds(allk)
            if r is not None and all(_contains(pc, s) or _contains(pb, s) for s in segs):
                return "C", "text exists in the pericope but not at %s" % cl.get("ref"), "quote-misref"
            sim = max(_best_window(crit + byz, s) for s in segs)
            return "C", "not found in SBLGNT/NA27/RP at %s (best similarity %.2f)" % (cl.get("ref") or "window", sim), "quote-text"
        else:
            mt = self._hebrew_text(keys)
            if all(_contains(mt, s) for s in segs):
                return "S", "matches WLC (consonantal)", "quote"
            if all(_orthographic_only(mt, s) for s in segs):
                # consonantal comparison already ignores vowels; a letter-level difference is not orthography -> judge
                return None, "near-match to WLC (letter-level difference)", "quote-ortho"
            allk = [(self.p["book"], c, v) for c, v in self.p["verses"]]
            if r is not None and all(_contains(self._hebrew_text(allk), s) for s in segs):
                return "C", "text exists in the unit but not at %s" % cl.get("ref"), "quote-misref"
            sim = max(_best_window(mt, s) for s in segs)
            return "C", "not found in WLC at %s (best similarity %.2f)" % (cl.get("ref") or "window", sim), "quote-text"

    def lxx_quote(self, cl, keys):
        segs = [_toks(s, "greek") for s in _segments(cl.get("orig"))]
        segs = [s for s in segs if s]
        if not segs or sum(len(s) for s in segs) < 3:
            return None, "LXX quotation too short", "lxx"
        lxx = []
        for k in keys:
            lk = lxx_ref(*k)
            if lk is None or lk not in self.c.lxx:
                return None, "LXX alignment unavailable", "lxx"
            lxx += _toks(self.c.lxx[lk], "greek")
        if not lxx:
            return None, "no LXX text", "lxx"
        sims = [_best_window(lxx, s) for s in segs]
        # Swete vs Rahlfs and orthography differ; only gross mismatches contradict
        if min(sims) >= 0.8:
            return "S", "matches Swete LXX (similarity %.2f)" % min(sims), "lxx"
        if max(sims) < 0.4:
            return "C", "does not match the Swete LXX of the verse (similarity %.2f)" % max(sims), "lxx"
        return None, "partial LXX match %.2f" % min(sims), "lxx"

    def variant_addomit(self, cl):
        """'Byzantine adds X' / 'critical text omits X' style claims (Greek only)."""
        if self.lang != "greek":
            return None, "hebrew variant: judge", "variant"
        r, keys = self._keys(cl.get("ref"), default="focus")
        x = _toks(cl.get("orig") or "", "greek")
        if not x or not keys or self._lang_of_keys(keys) != "greek":
            return None, "no reading text", "variant"
        crit, na, byz = self._greek_eds(keys)
        in_c = _contains(crit, x) or _contains(na, x)
        in_b = _contains(byz, x)
        trad = (cl.get("variant_tradition") or "").lower()
        op = cl.get("variant_op")
        # normalise to: 'the longer text (containing X) is tradition T'
        longer = None
        if op == "add":
            longer = trad
        elif op == "omit":
            longer = "critical" if trad in ("byzantine", "majority", "tr") else ("byzantine" if trad in ("critical", "alexandrian", "na28") else None)
        if longer is None:
            return None, "tradition unclear", "variant"
        if longer in ("byzantine", "majority", "tr"):
            if in_b and not in_c:
                return "S", "decided: X in RP, absent from SBLGNT/NA27", "variant-direction"
            if in_c and not in_b:
                return "C", "decided: direction reversed (X is in the critical text, not RP)", "variant-direction"
        if longer in ("critical", "alexandrian", "na28", "na"):
            if in_c and not in_b:
                return "S", "decided: X in SBLGNT/NA27, absent from RP", "variant-direction"
            if in_b and not in_c:
                return "C", "decided: direction reversed (X is in RP, not the critical text)", "variant-direction"
        return None, "undecided (X in both or neither)", "variant"

    def witness(self, cl):
        if self.lang != "greek":
            return None, "hebrew witness: judge", "witness"
        r, keys = self._keys(cl.get("ref"), default="focus")
        book = keys[0][0] if keys else self.p["book"]
        ch = keys[0][1] if keys else None
        if book not in NT_IDS:
            return None, "not NT", "witness"
        bad = []
        for s in cl.get("witnesses") or []:
            for ga in W.resolve(s, book):
                ok = W.contains(self.wit, ga, book, ch)
                if ok is False:
                    e = self.wit.get(ga, {})
                    bad.append("%s (%s) does not contain %s %s; contents: %s" % (s, ga, ENGLISH[book], ch,
                                                                                   e.get("contents_raw", "")[:80]))
        if bad:
            return "C", "; ".join(bad), "witness-contents"
        return None, "witness contents compatible or unknown", "witness"

    def _find_tokens(self, form, keys):
        """Greek: MorphGNT tokens whose normalized form (or lemma) equals form."""
        nf = norm_greek(form).replace(" ", "")
        hits = [w for k in keys for w in self.c.morph.get(k, []) if norm_greek(w["word"]) == nf]
        if hits:
            return hits, "form"
        hits = [w for k in keys for w in self.c.morph.get(k, []) if norm_greek(w["lemma"]) == nf]
        return hits, "lemma"

    def morphology(self, cl):
        form = (cl.get("orig") or "").strip()
        parse = cl.get("parse") or cl.get("claim") or ""
        if not form:
            return None, "no form", "morph"
        r, keys = self._keys(cl.get("ref"), default="window")
        if not keys:
            return None, "ref not found", "morph"
        lang = self._lang_of_keys(keys)
        if len(form.split()) > 2:
            return None, "multi-word form", "morph"
        if lang == "greek":
            claimed = parse_greek_prose(parse)
            hits, how = self._find_tokens(form.split()[-1] if len(form.split()) == 2 else form, keys)
            if not hits:
                return None, "form not found at ref", "morph"
            if how == "lemma" and len({w["parse"] for w in hits}) > 1:
                return None, "lexical form with several inflected occurrences", "morph"
            res = [compare_greek(claimed, w["pos"], w["parse"]) for w in hits]
            if any(x[0] == "S" for x in res):
                return "S", [x for x in res if x[0] == "S"][0][1], "morph"
            if res and all(x[0] == "C" for x in res):
                return "C", res[0][1], "morph"
            return None, "undecided", "morph"
        else:
            claimed = parse_hebrew_prose(parse)
            nf = norm_hebrew(form).replace(" ", "")
            hits = []
            for k in keys:
                for w in self.c.he_words.get(k, []):
                    if norm_hebrew(w["text"]).replace(" ", "") == nf:
                        hits.append(w)
            if not hits:
                return None, "form not found at ref", "morph"
            res = []
            for w in hits:
                segs = w["morph"][1:].split("/")
                vsegs = [s for s in segs if s.startswith("V")]
                if not vsegs:
                    if "conj" in claimed or "stem" in claimed:
                        res.append(("C", "claimed verbal parsing; OSHB: not a verb (%s)" % w["morph"]))
                    else:
                        res.append((None, "non-verb"))
                    continue
                res.append(compare_hebrew_verb(claimed, vsegs[0]))
            if any(x[0] == "S" for x in res):
                return "S", [x for x in res if x[0] == "S"][0][1], "morph"
            if res and all(x[0] == "C" for x in res):
                return "C", res[0][1], "morph"
            return None, "undecided", "morph"

    def _scope_keys(self, scope, lang):
        s = (scope or "").strip().lower()
        if not s:
            return None
        if lang == "greek":
            if s in ("nt", "new testament", "the new testament", "gnt", "greek new testament"):
                return lambda k: True
        else:
            if s in ("hebrew bible", "ot", "old testament", "mt", "tanakh", "the hebrew bible", "the old testament",
                     "hb", "masoretic text"):
                return lambda k: True
        b = book_id(s.replace("the gospel of", "").replace("gospel of", "").strip())
        if b:
            return lambda k, b=b: k[0] == b
        return None

    def occurrence(self, cl):
        lemma = cl.get("lemma") or cl.get("orig") or ""
        if not lemma.strip():
            return None, "no lemma", "occurrence"
        lang = "greek" if re.search(r"[Ͱ-Ͽἀ-῿]", lemma) else (
            "hebrew" if re.search(r"[א-ת]", lemma) else None)
        if lang is None:
            return None, "lemma not in original script", "occurrence"
        if lang == "greek":
            occ = self.c.lemma_index["greek"].get(norm_greek(lemma.split()[0]), [])
            if not occ:
                return None, "lemma not in MorphGNT index", "occurrence"
            occ_sets = [occ]
        else:
            ids = self.c.hebrew_lemma_ids(lemma.split()[0])
            if not ids:
                return None, "lemma not resolved to Strong's", "occurrence"
            occ_sets = [self.c.lemma_index["hebrew"].get(i, []) for i in ids]
            if len(ids) > 1:
                occ_sets.append([k for i in ids for k in self.c.lemma_index["hebrew"].get(i, [])])
        if cl.get("count") is not None:
            try:
                claimed = int(cl["count"])
            except (TypeError, ValueError):
                return None, "count not an integer", "occurrence"
            f = self._scope_keys(cl.get("count_scope"), lang)
            if f is None:
                return None, "scope not recognised: %s" % cl.get("count_scope"), "occurrence"
            actuals = []
            for occ in occ_sets:
                tok = [k for k in occ if f(k)]
                actuals.append((len(tok), len(set(tok))))
            for n_tok, n_vs in actuals:
                if claimed in (n_tok, n_vs):
                    return "S", "count %d matches index %d tokens / %d verses" % (claimed, n_tok, n_vs), "occurrence-count"
            for n_tok, n_vs in actuals:
                tol = max(1, round(0.1 * n_tok))
                if abs(claimed - n_tok) <= tol or abs(claimed - n_vs) <= max(1, round(0.1 * n_vs)):
                    # close but not exact: could be an edition/homograph difference -> judge (v1.1)
                    return None, "count %d near index %d tokens / %d verses" % (claimed, n_tok, n_vs), "occurrence-count"
            if len(occ_sets) == 1:
                n_tok, n_vs = actuals[0]
                return "C", "count %d vs index %d tokens / %d verses" % (claimed, n_tok, n_vs), "occurrence-count"
            return None, "homograph lemmas; counts %s" % actuals, "occurrence"
        if cl.get("ref"):
            text = (cl.get("claim") or "")
            if _NEG.search(text) or len(_REFS.findall(text)) > 1:
                return None, "negated or multi-reference claim", "occurrence"
            r, keys = self._keys(cl.get("ref"))
            if r is None or not keys:
                if r is not None and r["v1"] is not None and r["book"] in NT_IDS + OT_IDS:
                    return "C", "reference does not exist", "occurrence-ref"
                return None, "ref not usable", "occurrence"
            if self._lang_of_keys(keys) != lang:
                return None, "cross-language reference", "occurrence"
            hit = any(k in set(keys) for occ in occ_sets for k in occ)
            if hit:
                return "S", "lemma occurs in %s" % cl["ref"], "occurrence-ref"
            return "C", "lemma does not occur in %s" % cl["ref"], "occurrence-ref"
        return None, "nothing to check", "occurrence"

    def crossref(self, cl):
        r = parse_ref(cl.get("ref") or "")
        if r is None:
            return None, "ref not parsed", "crossref"
        if r["book"] not in NT_IDS + OT_IDS:
            return None, "unknown book", "crossref"
        keys = self.c.verses_in(r)
        if not self.c.chapter_exists(r["book"], r["c1"]):
            return "C", "chapter %s %d does not exist" % (ENGLISH[r["book"]], r["c1"]), "crossref-exists"
        if r["v1"] is not None and not keys:
            return "C", "verse %s does not exist" % cl.get("ref"), "crossref-exists"
        if r["v1"] is not None and not self.c.verse_exists(r["book"], r["c1"], r["v1"]):
            return "C", "verse %s does not exist" % cl.get("ref"), "crossref-exists"
        orig = cl.get("orig") or cl.get("lemma")
        if orig and keys:
            lang = self._lang_of_keys(keys)
            script_ok = (lang == "greek" and re.search(r"[Ͱ-Ͽἀ-῿]", orig)) or \
                        (lang == "hebrew" and re.search(r"[א-ת]", orig))
            if script_ok and len(orig.split()) == 1:
                if lang == "greek":
                    nf = norm_greek(orig)
                    words = {norm_greek(w["word"]) for k in keys for w in self.c.morph.get(k, [])}
                    lems = {norm_greek(w["lemma"]) for k in keys for w in self.c.morph.get(k, [])}
                    rp = set(_toks(" ".join(self.c.rp_text.get(k, "") for k in keys), "greek"))
                    if nf in words or nf in lems or nf in rp:
                        # presence does not verify what the claim says about the verse -> judge (v1.1)
                        return None, "word/lemma present in %s" % cl["ref"], "crossref-content"
                    # cognate or different form: allow stem overlap of 4+ letters
                    stem = nf[:4]
                    if any(w.startswith(stem) for w in words | lems | rp):
                        return None, "related form present", "crossref"
                    return "C", "%s not found in %s" % (orig, cl["ref"]), "crossref-content"
                else:
                    ids = self.c.hebrew_lemma_ids(orig)
                    nf = norm_hebrew(orig).replace(" ", "")
                    text = self._hebrew_text(keys)
                    if any(k in set(keys) for i in ids for k in self.c.lemma_index["hebrew"].get(i, [])) or nf in text:
                        return None, "word/lemma present in %s" % cl["ref"], "crossref-content"
                    if not ids:
                        return None, "lemma unresolved", "crossref"
                    return "C", "%s not found in %s" % (orig, cl["ref"]), "crossref-content"
        # existence alone does not verify what the claim says about the verse -> judge
        return None, "reference exists; content left to judge", "crossref"


def run_v1(corpus, passage, claims):
    ck = Checker(corpus, passage)
    out = []
    for cl in claims:
        lab, why, rule = ck.check(cl)
        out.append({"id": cl["id"], "v1": lab, "why": why, "rule": rule})
    return out
