"""Morphology codes <-> prose (MorphGNT and OSHB) and a conservative comparison for V1."""
import re

# ---------------------------------------------------------------- Greek (MorphGNT)
GK_POS = {"A-": "adjective", "C-": "conjunction", "D-": "adverb", "I-": "interjection", "N-": "noun",
          "P-": "preposition", "RA": "article", "RD": "demonstrative pronoun", "RI": "interrogative/indefinite pronoun",
          "RP": "personal pronoun", "RR": "relative pronoun", "V-": "verb", "X-": "particle"}
GK_PERSON = {"1": "1st", "2": "2nd", "3": "3rd"}
GK_TENSE = {"P": "present", "I": "imperfect", "F": "future", "A": "aorist", "X": "perfect", "Y": "pluperfect"}
GK_VOICE = {"A": "active", "M": "middle", "P": "passive"}
GK_MOOD = {"I": "indicative", "D": "imperative", "S": "subjunctive", "O": "optative", "N": "infinitive", "P": "participle"}
GK_CASE = {"N": "nominative", "G": "genitive", "D": "dative", "A": "accusative", "V": "vocative"}
GK_NUM = {"S": "singular", "P": "plural"}
GK_GEN = {"M": "masculine", "F": "feminine", "N": "neuter"}
GK_DEG = {"C": "comparative", "S": "superlative"}


def decode_greek(pos, parse):
    p = parse
    bits = []
    if p[1] != "-":
        bits.append(GK_TENSE.get(p[1], p[1]))
    if p[2] != "-":
        bits.append(GK_VOICE.get(p[2], p[2]))
    if p[3] != "-":
        bits.append(GK_MOOD.get(p[3], p[3]))
    if p[0] != "-":
        bits.append(GK_PERSON[p[0]] + " person")
    if p[4] != "-":
        bits.append(GK_CASE[p[4]])
    if p[5] != "-":
        bits.append(GK_NUM[p[5]])
    if p[6] != "-":
        bits.append(GK_GEN[p[6]])
    if len(p) > 7 and p[7] != "-":
        bits.append(GK_DEG.get(p[7], ""))
    return GK_POS.get(pos, pos) + (": " + " ".join(bits) if bits else "")


def greek_features(pos, parse):
    p = parse
    f = {"pos": pos}
    for i, (k, table) in enumerate([("person", GK_PERSON), ("tense", GK_TENSE), ("voice", GK_VOICE),
                                    ("mood", GK_MOOD), ("case", GK_CASE), ("number", GK_NUM), ("gender", GK_GEN)]):
        if p[i] != "-":
            f[k] = table[p[i]]
    return f


_GK_PROSE = [
    ("tense", r"\bpluperfect\b", "pluperfect"), ("tense", r"(?<!plu)\bperfect\b", "perfect"),
    ("tense", r"\bimperfect\b", "imperfect"), ("tense", r"\baorist\b", "aorist"),
    ("tense", r"\bfuture\b", "future"), ("tense", r"\bpresent\b", "present"),
    ("mood", r"\bindicative\b", "indicative"), ("mood", r"\bimperative\b", "imperative"),
    ("mood", r"\bsubjunctive\b", "subjunctive"), ("mood", r"\boptative\b", "optative"),
    ("mood", r"\binfinitive\b", "infinitive"), ("mood", r"\bparticiple\b|\bparticipial\b", "participle"),
    ("person", r"\b(1st|first)\b(?!\s*aorist)", "1st"), ("person", r"\b(2nd|second)\b(?!\s*aorist)", "2nd"),
    ("person", r"\b(3rd|third)\b", "3rd"),
    ("case", r"\bnominative\b", "nominative"), ("case", r"\bgenitive\b", "genitive"),
    ("case", r"\bdative\b", "dative"), ("case", r"\baccusative\b", "accusative"), ("case", r"\bvocative\b", "vocative"),
    ("number", r"\bsingular\b|\bsg\b", "singular"), ("number", r"\bplural\b|\bpl\b", "plural"),
    ("gender", r"\bmasculine\b|\bmasc\b", "masculine"), ("gender", r"\bfeminine\b|\bfem\b", "feminine"),
    ("gender", r"\bneuter\b", "neuter"),
]
_GK_ABBR = {"pres": "present", "impf": "imperfect", "fut": "future", "aor": "aorist", "pf": "perfect", "perf": "perfect",
            "plpf": "pluperfect", "act": "active", "mid": "middle", "pass": "passive", "ind": "indicative",
            "impv": "imperative", "imper": "imperative", "subj": "subjunctive", "opt": "optative", "inf": "infinitive",
            "ptc": "participle", "part": "participle", "ptcp": "participle", "nom": "nominative", "gen": "genitive",
            "dat": "dative", "acc": "accusative", "voc": "vocative", "sing": "singular", "plur": "plural",
            "neut": "neuter"}


def parse_greek_prose(s):
    """Extract claimed features from prose like 'aorist active participle, nom. masc. sg.'.

    Returns dict feature -> set of values.
    """
    t = (s or "").lower()
    t = re.sub(r"[\.,;/()\-]", " ", t)
    t = " ".join(_GK_ABBR.get(w, w) for w in t.split())
    f = {}
    for k, rx, val in _GK_PROSE:
        if re.search(rx, t):
            f.setdefault(k, set()).add(val)
    voices = set()
    if re.search(r"\bactive\b", t):
        voices.add("active")
    if re.search(r"\bmiddle\b", t):
        voices.add("middle")
    if re.search(r"\bpassive\b", t):
        voices.add("passive")
    if voices:
        f["voice"] = voices
    if "deponent" in t:
        f["deponent"] = {"yes"}
    if re.search(r"\b(second|2nd|first|1st)\s+aorist\b", t) and not re.search(r"\b(person|singular|plural|sg|pl)\b", t):
        f.pop("person", None)
    return f


def compare_greek(claimed, pos, parse):
    """Return ('S'|'C'|None, detail). Conservative: only clear mismatches contradict."""
    actual = greek_features(pos, parse)
    if not claimed:
        return None, "no features"
    mism, checked = [], []
    for k in ("tense", "mood", "person", "case", "number", "gender"):
        if k in claimed and k in actual:
            checked.append(k)
            if actual[k] not in claimed[k] and len(claimed[k]) == 1:
                mism.append("%s: claimed %s, MorphGNT %s" % (k, "/".join(sorted(claimed[k])), actual[k]))
        elif k in claimed and k not in actual and k in ("tense", "mood", "person"):
            return None, "feature %s not applicable" % k
    if "voice" in claimed and "voice" in actual:
        checked.append("voice")
        cv, av = claimed["voice"], actual["voice"]
        if av not in cv:
            ambiguous_mp = actual.get("tense") in ("present", "imperfect", "perfect", "pluperfect")
            if "deponent" in claimed:
                pass
            elif ambiguous_mp and av in ("middle", "passive") and cv & {"middle", "passive"}:
                pass
            else:
                mism.append("voice: claimed %s, MorphGNT %s" % ("/".join(sorted(cv)), av))
    if mism:
        return "C", "; ".join(mism)
    if checked:
        return "S", "matches " + ",".join(checked)
    return None, "nothing comparable"


# ---------------------------------------------------------------- Hebrew (OSHB)
HE_STEM = {"q": "qal", "N": "niphal", "p": "piel", "P": "pual", "h": "hiphil", "H": "hophal", "t": "hithpael",
           "o": "polel", "O": "polal", "r": "hithpolel", "m": "poel", "M": "poal", "k": "palel", "K": "pulal",
           "Q": "qal passive", "l": "pilpel", "L": "polpal", "f": "hithpalpel", "D": "nithpael", "j": "pealal",
           "i": "pilel", "u": "hothpaal", "c": "tiphil", "v": "hishtaphel", "w": "nithpalel", "y": "nithpoel",
           "z": "hithpoel"}
HE_CONJ = {"p": "perfect", "q": "sequential perfect", "i": "imperfect", "w": "sequential imperfect",
           "h": "cohortative", "j": "jussive", "v": "imperative", "r": "participle active",
           "s": "participle passive", "a": "infinitive absolute", "c": "infinitive construct"}
HE_PERSON = {"1": "1st", "2": "2nd", "3": "3rd"}
HE_GEN = {"m": "masculine", "f": "feminine", "c": "common", "b": "both"}
HE_NUM = {"s": "singular", "p": "plural", "d": "dual"}
HE_STATE = {"a": "absolute", "c": "construct", "d": "determined"}
HE_POS = {"A": "adjective", "C": "conjunction", "D": "adverb", "N": "noun", "P": "pronoun", "R": "preposition",
          "S": "suffix", "T": "particle", "V": "verb"}


def split_he_morph(m):
    if not m:
        return "", []
    return m[0], m[1:].split("/")


def he_verb_features(seg):
    """'Vqw3ms' -> dict. Participles carry gender/number/state instead of person."""
    if not seg.startswith("V") or len(seg) < 3:
        return None
    f = {"stem": HE_STEM.get(seg[1], seg[1]), "conj": HE_CONJ.get(seg[2], seg[2])}
    rest = seg[3:]
    if seg[2] in "rs":
        if len(rest) >= 1:
            f["gender"] = HE_GEN.get(rest[0])
        if len(rest) >= 2:
            f["number"] = HE_NUM.get(rest[1])
        if len(rest) >= 3:
            f["state"] = HE_STATE.get(rest[2])
    elif seg[2] in "ac":
        pass
    else:
        if len(rest) >= 1:
            f["person"] = HE_PERSON.get(rest[0])
        if len(rest) >= 2:
            f["gender"] = HE_GEN.get(rest[1])
        if len(rest) >= 3:
            f["number"] = HE_NUM.get(rest[2])
    return {k: v for k, v in f.items() if v}


def decode_hebrew(m):
    lang, segs = split_he_morph(m)
    out = []
    for s in segs:
        if not s:
            continue
        if s[0] == "V":
            f = he_verb_features(s) or {}
            out.append("verb " + " ".join(f[k] for k in ("stem", "conj", "person", "gender", "number", "state") if f.get(k)))
        elif s[0] == "N":
            kind = {"c": "common", "p": "proper", "g": "gentilic", "x": ""}.get(s[1:2], "")
            g = HE_GEN.get(s[2:3], "")
            n = HE_NUM.get(s[3:4], "")
            st = HE_STATE.get(s[4:5], "")
            out.append(" ".join(x for x in ("noun", kind, g, n, st) if x))
        elif s[0] == "S":
            out.append("suffix " + s[1:])
        else:
            out.append(HE_POS.get(s[0], s[0]) + ((" " + s[1:]) if len(s) > 1 else ""))
    return ("Aramaic: " if lang == "A" else "") + " + ".join(out)


_HE_STEM_RX = [("qal passive", r"\bqal passive\b"), ("qal", r"\bqal\b|\bg stem\b"), ("niphal", r"\bni?ph'?al\b|\bnifal\b"),
               ("piel", r"\bpi'?el\b|\bd stem\b"), ("pual", r"\bpu'?al\b"), ("hiphil", r"\bhiph'?il\b|\bhifil\b"),
               ("hophal", r"\bhoph'?al\b|\bhofal\b"), ("hithpael", r"\bhithpa'?el\b|\bhitpael\b"),
               ("polel", r"\bpolel\b"), ("hithpolel", r"\bhithpolel\b"), ("poel", r"\bpoel\b"), ("pilpel", r"\bpilpel\b")]


def parse_hebrew_prose(s):
    t = (s or "").lower().replace("-", " ").replace("’", "'")
    t = re.sub(r"[\.,;/()]", " ", t)
    f = {}
    for val, rx in _HE_STEM_RX:
        if re.search(rx, t):
            f.setdefault("stem", set()).add(val)
            if val == "qal passive":
                break
    conj = set()
    if re.search(r"wayyiqtol|waw consecutive imperfect|waw conversive imperfect|consecutive imperfect|preterite|narrative tense|sequential imperfect|vav consecutive imperfect", t):
        conj.add("sequential imperfect")
    elif re.search(r"weqatal|waw consecutive perfect|consecutive perfect|sequential perfect|waw conversive perfect", t):
        conj.add("sequential perfect")
    elif re.search(r"\bimperfect\b|\byiqtol\b|\bprefix conjugation\b", t):
        conj.add("imperfect")
    elif re.search(r"\bperfect\b|\bqatal\b|\bsuffix conjugation\b", t):
        conj.add("perfect")
    if re.search(r"\bimperative\b", t):
        conj = {"imperative"}
    if re.search(r"\bjussive\b", t):
        conj = {"jussive"}
    if re.search(r"\bcohortative\b", t):
        conj = {"cohortative"}
    if re.search(r"infinitive absolute|inf abs", t):
        conj = {"infinitive absolute"}
    elif re.search(r"infinitive construct|inf cstr|inf const", t):
        conj = {"infinitive construct"}
    elif re.search(r"\binfinitive\b", t):
        conj = {"infinitive absolute", "infinitive construct"}
    if re.search(r"\bparticiple\b|\bptc\b", t):
        conj = {"participle active", "participle passive"}
    if conj:
        f["conj"] = conj
    for val, rx in [("1st", r"\b(1st|first)( person)?\b|\b1[mfc][sp]\b|\b1c\b"),
                    ("2nd", r"\b(2nd|second)( person)?\b|\b2[mfc][sp]\b"),
                    ("3rd", r"\b(3rd|third)( person)?\b|\b3[mfc][sp]\b")]:
        if re.search(rx, t):
            f.setdefault("person", set()).add(val)
    for val, rx in [("masculine", r"\bmasculine\b|\bmasc\b|\b[123]m[sp]\b|\bm[sp]\b"),
                    ("feminine", r"\bfeminine\b|\bfem\b|\b[123]f[sp]\b|\bf[sp]\b"),
                    ("common", r"\bcommon\b|\b[123]c[sp]\b")]:
        if re.search(rx, t):
            f.setdefault("gender", set()).add(val)
    for val, rx in [("singular", r"\bsingular\b|\bsg\b|\b[123]?[mfc]s\b"),
                    ("plural", r"\bplural\b|\bpl\b|\b[123]?[mfc]p\b"), ("dual", r"\bdual\b")]:
        if re.search(rx, t):
            f.setdefault("number", set()).add(val)
    return f


_CONJ_FAMILY = {"perfect": "perf", "sequential perfect": "perf", "imperfect": "impf", "sequential imperfect": "impf",
                "jussive": "impf", "cohortative": "impf", "imperative": "impv", "participle active": "ptc",
                "participle passive": "ptc", "infinitive absolute": "inf", "infinitive construct": "inf"}


def compare_hebrew_verb(claimed, seg):
    actual = he_verb_features(seg)
    if not actual or not claimed:
        return None, "not a verb or nothing claimed"
    if "person" in claimed and "person" not in actual:
        # e.g. '2fs participle' for an archaic-ending participle: contested analysis -> judge (v1.1)
        return None, "person claimed for a non-finite form"
    mism, checked = [], []
    if "stem" in claimed:
        checked.append("stem")
        if actual["stem"] not in claimed["stem"] and len(claimed["stem"]) == 1:
            mism.append("stem: claimed %s, OSHB %s" % (list(claimed["stem"])[0], actual["stem"]))
    if "conj" in claimed:
        checked.append("conj")
        ac = actual["conj"]
        cc = claimed["conj"]
        fam_a = _CONJ_FAMILY.get(ac)
        fam_c = {_CONJ_FAMILY.get(x) for x in cc}
        if fam_a not in fam_c:
            mism.append("conjugation: claimed %s, OSHB %s" % ("/".join(sorted(cc)), ac))
        elif ac == "imperfect" and cc == {"sequential imperfect"}:
            mism.append("claimed wayyiqtol, OSHB plain imperfect")
        elif ac == "sequential imperfect" and cc == {"jussive"}:
            mism.append("claimed jussive, OSHB wayyiqtol")
    for k in ("person", "gender", "number"):
        if k in claimed and actual.get(k):
            checked.append(k)
            av = actual[k]
            if av in ("common", "both"):
                continue
            if av not in claimed[k] and len(claimed[k]) == 1 and "common" not in claimed[k]:
                mism.append("%s: claimed %s, OSHB %s" % (k, list(claimed[k])[0], av))
    if mism:
        return "C", "; ".join(mism)
    if checked:
        return "S", "matches " + ",".join(checked)
    return None, "nothing comparable"
