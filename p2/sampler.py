"""WP1: pericope sampling frame, strata, seeded stratified draw, split assignment, focus verses.

Pre-registered (D15): seed and this code are committed before any model sees a passage.
"""
import hashlib
import random
from collections import Counter, defaultdict

from .common import NT_GENRE, OT_GENRE, ENGLISH, norm_greek

SEED = 20261012          # committed seed for the Program 2 passage draw
SPLITS = [("design", 8), ("probe", 8), ("calibration", 4), ("evaluation", 12)]
PER_LANG = sum(n for _, n in SPLITS)
FIXED = {"greek": ("Mark", 9, 14, 29, (9, 23), (9, 24))}  # Mark 9:14-29, focus 9:23-24
FIXED_LEXEMES = ["πιστεύω", "δύναμαι", "βοηθέω"]  # as in the 11 Oct pilot prompt
MIN_V, MAX_V = 8, 20


def _rng(*parts):
    h = hashlib.sha256(("%d|" % SEED + "|".join(map(str, parts))).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def frame(corpus):
    """All 8-20 verse paragraph units, Aramaic excluded. Returns list of dicts."""
    units = []
    for lang, paras, genre in (("greek", corpus.gk_paras, NT_GENRE), ("hebrew", corpus.he_paras, OT_GENRE)):
        seen = set()
        for book, vs in paras:
            vs = sorted(set(vs))
            if not (MIN_V <= len(vs) <= MAX_V):
                continue
            keys = [(book, c, v) for c, v in vs]
            if lang == "hebrew" and any(k in corpus.he_aramaic for k in keys):
                continue
            uid = "%s.%d.%d-%d.%d" % (book, vs[0][0], vs[0][1], vs[-1][0], vs[-1][1])
            if uid in seen:
                continue
            seen.add(uid)
            if lang == "greek":
                dens = sum(len(corpus.app.get(k, [])) for k in keys) / len(keys)
            else:
                dens = sum(len(corpus.he_kq.get(k, [])) + corpus.he_notes.get(k, 0) for k in keys) / len(keys)
            fam = sum(corpus.xref_votes.get(k, 0) for k in keys) / len(keys)
            units.append({"id": uid, "lang": lang, "book": book, "verses": vs, "n_verses": len(vs),
                          "genre": genre[book], "density": dens, "familiarity": fam})
    b, c, v1, v2, _, _ = FIXED["greek"]
    fid = "%s.%d.%d-%d.%d" % (b, c, v1, c, v2)
    if not any(u["id"] == fid for u in units):
        vs = [(c, v) for v in range(v1, v2 + 1)]
        keys = [(b, c, v) for c, v in vs]
        units.append({"id": fid, "lang": "greek", "book": b, "verses": vs, "n_verses": len(vs), "genre": "narrative",
                      "density": sum(len(corpus.app.get(k, [])) for k in keys) / len(keys),
                      "familiarity": sum(corpus.xref_votes.get(k, 0) for k in keys) / len(keys)})
    for lang in ("greek", "hebrew"):
        us = [u for u in units if u["lang"] == lang]
        ds = sorted(u["density"] for u in us)
        med = ds[len(ds) // 2]
        fs = sorted(u["familiarity"] for u in us)
        t1, t2 = fs[len(fs) // 3], fs[2 * len(fs) // 3]
        for u in us:
            u["dens_stratum"] = "high" if u["density"] > med else "low"
            u["fam_tercile"] = "low" if u["familiarity"] <= t1 else ("mid" if u["familiarity"] <= t2 else "high")
            u["stratum"] = "%s|%s|%s|%s" % (lang, u["genre"], u["dens_stratum"], u["fam_tercile"])
    return units


def draw(units):
    """Seeded stratified draw: per language, genre quotas as equal as possible; within a genre the
    six density x familiarity cells are visited in seeded random order, one passage per visit;
    unfilled quota (few eligible units) is redistributed to the least-represented strata."""
    chosen = []
    for lang in ("greek", "hebrew"):
        us = [u for u in units if u["lang"] == lang]
        genres = sorted({u["genre"] for u in us})
        quota = {g: PER_LANG // len(genres) for g in genres}
        for g in genres[: PER_LANG - sum(quota.values())]:
            quota[g] += 1
        picked = []
        fixed = FIXED.get(lang)
        if fixed:
            b, c, v1, v2, _, _ = fixed
            fid = "%s.%d.%d-%d.%d" % (b, c, v1, c, v2)
            fu = next(u for u in us if u["id"] == fid)
            fu["fixed"] = True
            picked.append(fu)
            quota[fu["genre"]] -= 1
        for g in genres:
            cells = defaultdict(list)
            for u in us:
                if u["genre"] == g and u not in picked:
                    cells[(u["dens_stratum"], u["fam_tercile"])].append(u)
            for k in cells:
                cells[k].sort(key=lambda u: u["id"])
            rng = _rng("draw", lang, g)
            order = sorted(cells)
            rng.shuffle(order)
            n = 0
            while n < quota[g] and any(cells.values()):
                for k in order:
                    if n >= quota[g]:
                        break
                    if cells[k]:
                        u = cells[k].pop(rng.randrange(len(cells[k])))
                        if any(p["book"] == u["book"] and set(p["verses"]) & set(u["verses"]) for p in picked):
                            continue
                        picked.append(u)
                        n += 1
        rng = _rng("fill", lang)
        left = sorted([u for u in us if u not in picked], key=lambda u: u["id"])
        rng.shuffle(left)
        left.sort(key=lambda u: Counter(p["stratum"] for p in picked)[u["stratum"]])
        for u in left:
            if len(picked) >= PER_LANG:
                break
            if any(p["book"] == u["book"] and set(p["verses"]) & set(u["verses"]) for p in picked):
                continue
            picked.append(u)
        chosen.extend(picked)
    return chosen


def assign_splits(chosen):
    """Stratum-balanced, seeded assignment to design/probe/calibration/evaluation."""
    for lang in ("greek", "hebrew"):
        us = [u for u in chosen if u["lang"] == lang]
        rng = _rng("split", lang)
        rng.shuffle(us)
        us.sort(key=lambda u: (u["genre"], u["dens_stratum"], u["fam_tercile"]))
        need = dict(SPLITS)
        got = Counter()
        fixed = [u for u in us if u.get("fixed")]
        for u in fixed:
            u["split"] = "design"
            got["design"] += 1
        total = len(us)
        for i, u in enumerate([u for u in us if not u.get("fixed")]):
            def deficit(s):
                return need[s] * (i + len(fixed) + 1) / total - got[s]
            cands = [s for s, _ in SPLITS if got[s] < need[s]]
            best = max(deficit(s) for s in cands)
            top = [s for s in cands if abs(deficit(s) - best) < 1e-9]
            s = top[rng.randrange(len(top))]
            u["split"] = s
            got[s] += 1
    return chosen


def half_design(chosen):
    """Seeded 4+4 subset of the design split for the 122B and repair cells (fixed passage included)."""
    out = []
    for lang in ("greek", "hebrew"):
        d = sorted([u for u in chosen if u["lang"] == lang and u["split"] == "design"], key=lambda u: u["id"])
        rng = _rng("half", lang)
        fixed = [u for u in d if u.get("fixed")]
        rest = [u for u in d if not u.get("fixed")]
        rng.shuffle(rest)
        bygen = defaultdict(list)
        for u in rest:
            bygen[u["genre"]].append(u)
        sel = list(fixed)
        gs = sorted(bygen)
        rng.shuffle(gs)
        while len(sel) < 4 and any(bygen.values()):
            for g in gs:
                if len(sel) < 4 and bygen[g]:
                    sel.append(bygen[g].pop(0))
        for u in sel:
            u["half"] = True
        out.extend(sel)
    return out


def choose_focus(u, corpus):
    lang = u["lang"]
    fixed = FIXED.get(lang)
    if fixed and u.get("fixed"):
        return [fixed[4], fixed[5]]
    vs = u["verses"]
    inner = vs[1:-1] if len(vs) > 4 else vs
    best, best_score = None, -1
    rng = _rng("focus", u["id"])
    tiebreak = {v: rng.random() for v in inner}
    for i in range(len(inner) - 1):
        a, b = inner[i], inner[i + 1]
        if a[0] != b[0]:
            continue
        score = 0
        for c, v in (a, b):
            k = (u["book"], c, v)
            if lang == "greek":
                score += 2 * len(corpus.app.get(k, []))
                score += sum(1 for w in corpus.morph.get(k, []) if w["pos"] == "V-")
            else:
                score += 2 * (len(corpus.he_kq.get(k, [])) + corpus.he_notes.get(k, 0))
                score += sum(1 for w in corpus.he_words.get(k, []) if "/V" in "/" + w["morph"][1:])
        score += tiebreak[a] * 0.5
        if score > best_score:
            best, best_score = [a, b], score
    return best


_GK_STOP = None
_HE_STOP = None


def _stoplists(corpus):
    global _GK_STOP, _HE_STOP
    if _GK_STOP is None:
        cg = Counter()
        for ws in corpus.morph.values():
            for w in ws:
                cg[w["lemma"]] += 1
        _GK_STOP = {l for l, _ in cg.most_common(40)}
        ch = Counter()
        for ws in corpus.he_words.values():
            for w in ws:
                ch[w["lemma"].split("/")[-1]] += 1
        _HE_STOP = {l for l, _ in ch.most_common(40)}
    return _GK_STOP, _HE_STOP


def choose_lexemes(u, focus, corpus, k=3):
    gk_stop, he_stop = _stoplists(corpus)
    cands = []
    seen = set()
    for c, v in focus:
        key = (u["book"], c, v)
        if u["lang"] == "greek":
            for w in corpus.morph.get(key, []):
                if w["pos"] in ("N-", "V-", "A-") and w["lemma"] not in gk_stop and w["lemma"] not in seen:
                    seen.add(w["lemma"])
                    n = len(corpus.lemma_index["greek"].get(norm_greek(w["lemma"]), []))
                    cands.append((n, w["lemma"]))
        else:
            for w in corpus.he_words.get(key, []):
                segs = w["morph"][1:].split("/")
                lem = w["lemma"].split("/")[-1].strip()
                nonsuf = [s for s in segs if s and not s.startswith("S")]
                core = nonsuf[-1] if nonsuf else ""
                if core[:1] in ("N", "V", "A") and not core.startswith("Np") and "־" not in corpus.strong.get("H" + lem.split()[0], "") \
                        and lem not in he_stop and lem not in seen:
                    seen.add(lem)
                    sid = "H" + lem.split()[0]
                    n = len(corpus.lemma_index["hebrew"].get(sid, []))
                    cands.append((n, corpus.strong.get(sid, lem)))
    # distinctive but not rare: corpus frequency >= 5, lowest frequencies first; no proper names
    cands = [c for c in cands if c[0] >= 5 and not c[1][:1].isupper()]
    cands.sort(key=lambda x: (x[0], x[1]))
    return [l for _, l in cands[:k]]


def build_passages(corpus):
    units = frame(corpus)
    chosen = draw(units)
    assign_splits(chosen)
    half_design(chosen)
    out = []
    for u in sorted(chosen, key=lambda u: (u["lang"], u["split"], u["id"])):
        focus = choose_focus(u, corpus)
        vs = u["verses"]
        i0 = vs.index(focus[0])
        i1 = vs.index(focus[-1])
        window = vs[max(0, i0 - 2): i1 + 3]
        u2 = dict(u)
        u2["focus"] = focus
        u2["window"] = window
        u2["lexemes"] = FIXED_LEXEMES if u.get("fixed") else choose_lexemes(u, focus, corpus)
        u2["pericope_ref"] = _ref(u["book"], vs)
        u2["focus_ref"] = _ref(u["book"], focus)
        u2["window_ref"] = _ref(u["book"], window)
        u2.setdefault("half", False)
        u2.setdefault("fixed", False)
        out.append(u2)
    return out, units


def _ref(book, vs):
    a, b = vs[0], vs[-1]
    name = ENGLISH[book]
    if a == b:
        return "%s %d:%d" % (name, a[0], a[1])
    if a[0] == b[0]:
        return "%s %d:%d-%d" % (name, a[0], a[1], b[1])
    return "%s %d:%d-%d:%d" % (name, a[0], a[1], b[0], b[1])
