"""Shared paths, book tables, reference parsing and Greek/Hebrew normalization.

Program 2 (Open-weight LLM Reduced Hallucination Test Plan, v0.4).
"""
import json
import os
import re
import unicodedata

HOME = os.path.expanduser("~")
P2 = os.environ.get("P2_ROOT", os.path.join(HOME, "llm_tests", "program2"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENDATA = os.environ.get("P2_OPENDATA", os.path.join(P2, "opendata"))
RAWDATA = os.path.join(P2, "rawdata")
RUNS = os.path.join(P2, "runs")
STATE = os.path.join(P2, "state")
LOGS = os.path.join(P2, "logs")
PRIVATE = os.path.join(P2, "private")  # licensed / key material, never committed

for _d in (RUNS, STATE, LOGS):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def jload(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Books
# --------------------------------------------------------------------------
# (osis, morphgnt file stem, sblapp stem, rp csv stem, english name, aliases)
NT_BOOKS = [
    ("Matt", "61-Mt", "61-Mt", "MAT", "Matthew", ["mt", "matt", "mat", "matthew"]),
    ("Mark", "62-Mk", "62-Mk", "MAR", "Mark", ["mk", "mar", "mark", "mrk"]),
    ("Luke", "63-Lk", "63-Lk", "LUK", "Luke", ["lk", "luk", "luke"]),
    ("John", "64-Jn", "64-Jn", "JOH", "John", ["jn", "joh", "john", "jhn"]),
    ("Acts", "65-Ac", "65-Ac", "ACT", "Acts", ["ac", "act", "acts"]),
    ("Rom", "66-Ro", "66-Ro", "ROM", "Romans", ["ro", "rom", "romans"]),
    ("1Cor", "67-1Co", "67-1Co", "1CO", "1 Corinthians", ["1co", "1cor", "1corinthians"]),
    ("2Cor", "68-2Co", "68-2Co", "2CO", "2 Corinthians", ["2co", "2cor", "2corinthians"]),
    ("Gal", "69-Ga", "69-Ga", "GAL", "Galatians", ["ga", "gal", "galatians"]),
    ("Eph", "70-Eph", "70-Eph", "EPH", "Ephesians", ["eph", "ephesians"]),
    ("Phil", "71-Php", "71-Php", "PHP", "Philippians", ["php", "phil", "philippians"]),
    ("Col", "72-Col", "72-Col", "COL", "Colossians", ["col", "colossians"]),
    ("1Thess", "73-1Th", "73-1Th", "1TH", "1 Thessalonians", ["1th", "1thess", "1thessalonians"]),
    ("2Thess", "74-2Th", "74-2Th", "2TH", "2 Thessalonians", ["2th", "2thess", "2thessalonians"]),
    ("1Tim", "75-1Ti", "75-1Ti", "1TI", "1 Timothy", ["1ti", "1tim", "1timothy"]),
    ("2Tim", "76-2Ti", "76-2Ti", "2TI", "2 Timothy", ["2ti", "2tim", "2timothy"]),
    ("Titus", "77-Tit", "77-Tit", "TIT", "Titus", ["tit", "titus"]),
    ("Phlm", "78-Phm", "78-Phm", "PHM", "Philemon", ["phm", "phlm", "philemon"]),
    ("Heb", "79-Heb", "79-Heb", "HEB", "Hebrews", ["heb", "hebrews"]),
    ("Jas", "80-Jas", "80-Jas", "JAM", "James", ["jas", "jam", "james"]),
    ("1Pet", "81-1Pe", "81-1Pe", "1PE", "1 Peter", ["1pe", "1pet", "1peter"]),
    ("2Pet", "82-2Pe", "82-2Pe", "2PE", "2 Peter", ["2pe", "2pet", "2peter"]),
    ("1John", "83-1Jn", "83-1Jn", "1JO", "1 John", ["1jn", "1jo", "1john"]),
    ("2John", "84-2Jn", "84-2Jn", "2JO", "2 John", ["2jn", "2jo", "2john"]),
    ("3John", "85-3Jn", "85-3Jn", "3JO", "3 John", ["3jn", "3jo", "3john"]),
    ("Jude", "86-Jud", "86-Jud", "JUD", "Jude", ["jud", "jude"]),
    ("Rev", "87-Re", "87-Re", "REV", "Revelation", ["re", "rev", "revelation", "apocalypse", "apoc"]),
]

# (osis/OSHB name, english, aliases, swete file or None, genre)
OT_BOOKS = [
    ("Gen", "Genesis", ["gen", "ge", "gn", "genesis"], "01.Genesis.txt", "narrative"),
    ("Exod", "Exodus", ["ex", "exod", "exo", "exodus"], "02.Exodus.txt", "narrative"),
    ("Lev", "Leviticus", ["lev", "lv", "leviticus"], "03.Leviticus.txt", "discourse"),
    ("Num", "Numbers", ["num", "nu", "nm", "numbers"], "04.Numeri.txt", "narrative"),
    ("Deut", "Deuteronomy", ["deut", "dt", "deu", "deuteronomy"], "05.Deuteronomium.txt", "discourse"),
    ("Josh", "Joshua", ["josh", "jos", "joshua"], "06.Josue.txt", "narrative"),
    ("Judg", "Judges", ["judg", "jdg", "judges"], "08.Judices.txt", "narrative"),
    ("Ruth", "Ruth", ["ruth", "ru", "rut"], "10.Ruth.txt", "narrative"),
    ("1Sam", "1 Samuel", ["1sam", "1sa", "1samuel", "1kingdoms"], "11.Regnorum_I.txt", "narrative"),
    ("2Sam", "2 Samuel", ["2sam", "2sa", "2samuel", "2kingdoms"], "12.Regnorum_II.txt", "narrative"),
    ("1Kgs", "1 Kings", ["1kgs", "1ki", "1kings", "1kg", "3kingdoms"], "13.Regnorum_III.txt", "narrative"),
    ("2Kgs", "2 Kings", ["2kgs", "2ki", "2kings", "2kg", "4kingdoms"], "14.Regnorum_IV.txt", "narrative"),
    ("1Chr", "1 Chronicles", ["1chr", "1ch", "1chronicles"], "15.Paralipomenon_I.txt", "narrative"),
    ("2Chr", "2 Chronicles", ["2chr", "2ch", "2chronicles"], "16.Paralipomenon_II.txt", "narrative"),
    ("Ezra", "Ezra", ["ezra", "ezr"], "18.Esdras_B.txt", "narrative"),
    ("Neh", "Nehemiah", ["neh", "ne", "nehemiah"], "18.Esdras_B.txt", "narrative"),
    ("Esth", "Esther", ["esth", "est", "esther"], "19.Esther.txt", "narrative"),
    ("Job", "Job", ["job", "jb"], "32.Job.txt", "poetry"),
    ("Ps", "Psalms", ["ps", "psa", "psalm", "psalms", "pss"], "27.Psalmi.txt", "poetry"),
    ("Prov", "Proverbs", ["prov", "pr", "pro", "proverbs"], "29.Proverbia.txt", "poetry"),
    ("Eccl", "Ecclesiastes", ["eccl", "ecc", "qoh", "ecclesiastes", "qoheleth"], None, "poetry"),
    ("Song", "Song of Songs", ["song", "sng", "songofsongs", "songofsolomon", "cant", "canticles"], "31.Canticum.txt", "poetry"),
    ("Isa", "Isaiah", ["isa", "is", "isaiah"], "48.Isaias.txt", "poetry"),
    ("Jer", "Jeremiah", ["jer", "je", "jeremiah"], "49.Jeremias.txt", "poetry"),
    ("Lam", "Lamentations", ["lam", "la", "lamentations"], "51.Threni_seu_Lamentationes.txt", "poetry"),
    ("Ezek", "Ezekiel", ["ezek", "eze", "ezk", "ezekiel"], "53.Ezechiel.txt", "poetry"),
    ("Dan", "Daniel", ["dan", "da", "dn", "daniel"], "57.Daniel_Theodotionis_versio.txt", "narrative"),
    ("Hos", "Hosea", ["hos", "ho", "hosea"], "36.Osee.txt", "poetry"),
    ("Joel", "Joel", ["joel", "jl"], "39.Joel.txt", "poetry"),
    ("Amos", "Amos", ["amos", "am"], "37.Amos.txt", "poetry"),
    ("Obad", "Obadiah", ["obad", "ob", "oba", "obadiah"], "40.Abdias.txt", "poetry"),
    ("Jonah", "Jonah", ["jonah", "jon", "jnh"], "41.Jonas.txt", "narrative"),
    ("Mic", "Micah", ["mic", "mi", "micah"], "38.Michaeas.txt", "poetry"),
    ("Nah", "Nahum", ["nah", "na", "nahum"], "42.Nahum.txt", "poetry"),
    ("Hab", "Habakkuk", ["hab", "hb", "habakkuk"], "43.Habacuc.txt", "poetry"),
    ("Zeph", "Zephaniah", ["zeph", "zep", "zephaniah"], "44.Sophonias.txt", "poetry"),
    ("Hag", "Haggai", ["hag", "hg", "haggai"], "45.Aggaeus.txt", "poetry"),
    ("Zech", "Zechariah", ["zech", "zec", "zechariah"], "46.Zacharias.txt", "poetry"),
    ("Mal", "Malachi", ["mal", "ml", "malachi"], "47.Malachias.txt", "poetry"),
]

NT_GENRE = {b[0]: ("narrative" if b[0] in ("Matt", "Mark", "Luke", "John", "Acts") else
                   "poetry" if b[0] == "Rev" else "discourse") for b in NT_BOOKS}
OT_GENRE = {b[0]: b[4] for b in OT_BOOKS}
NT_IDS = [b[0] for b in NT_BOOKS]
OT_IDS = [b[0] for b in OT_BOOKS]
ENGLISH = {b[0]: b[4] for b in NT_BOOKS}
ENGLISH.update({b[0]: b[1] for b in OT_BOOKS})

_ALIAS = {}
for b in NT_BOOKS:
    for a in b[5] + [b[0].lower(), b[4].lower().replace(" ", "")]:
        _ALIAS[a] = b[0]
for b in OT_BOOKS:
    for a in b[2] + [b[0].lower(), b[1].lower().replace(" ", "")]:
        _ALIAS[a] = b[0]
# roman numeral / word prefixes
for k, v in list(_ALIAS.items()):
    if k[0] in "123":
        n = k[0]
        rest = k[1:]
        for p in {"1": ["i", "first"], "2": ["ii", "second"], "3": ["iii", "third"]}[n]:
            _ALIAS.setdefault(p + rest, v)


def book_id(name):
    """Map a free-form book name ('Mk', 'Matthew', '1 Cor', 'Psalm') to an OSIS-like id."""
    if not name:
        return None
    k = re.sub(r"[\s\.]", "", name.lower())
    return _ALIAS.get(k)


def lang_of(book):
    return "greek" if book in NT_IDS else "hebrew" if book in OT_IDS else None


_REF_RE = re.compile(
    r"(?P<book>(?:[123]|i{1,3})?\s?[A-Za-z][A-Za-z\.]*(?:\s(?:of\s)?[A-Z][a-z]+)?)\s*"
    r"(?P<c1>\d+)(?:[:\.,](?P<v1>\d+)[a-z]?)?"
    r"(?:\s*[-–—]\s*(?:(?P<c2>\d+)[:\.])?(?P<v2>\d+)[a-z]?)?")


def parse_ref(s):
    """Parse 'Mark 9:23', 'Mk 9.23-24', 'Ps 22:1', 'Luke 9:37-10:2'.

    Returns dict(book, c1, v1, c2, v2) or None. v1 None means whole chapter.
    """
    if not s:
        return None
    s = s.strip()
    m = _REF_RE.search(s)
    if not m:
        return None
    b = book_id(m.group("book"))
    if not b:
        parts = m.group("book").split()
        b = book_id(parts[-1]) if parts else None
    if not b:
        return None
    c1 = int(m.group("c1"))
    v1 = int(m.group("v1")) if m.group("v1") else None
    if m.group("v2"):
        c2 = int(m.group("c2")) if m.group("c2") else c1
        v2 = int(m.group("v2"))
        if v1 is None:  # 'Mark 9-10' chapter range
            c2, v2, v1 = v2, None, None
    else:
        c2, v2 = c1, v1
    return {"book": b, "c1": c1, "v1": v1, "c2": c2, "v2": v2}


def fmt_ref(book, c, v, c2=None, v2=None):
    s = "%s %d:%d" % (book, c, v)
    if v2 is not None and (c2, v2) != (c, v):
        s += "-%d" % v2 if c2 in (None, c) else "-%d:%d" % (c2, v2)
    return s


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
_GREEK_PUNCT = "·.,;:·‧⸀⸁⸂⸃⸄⸅⸆⸇⸈⸉⸊⸋⸌⸍[]()⟦⟧{}'\"“”‘’—–-!?*†‡¶§«»<>|/\\"
_GREEK_TRANS = {ord(c): " " for c in _GREEK_PUNCT}
_GREEK_TRANS[ord("ʼ")] = ""
_GREEK_TRANS[ord("᾽")] = ""
_GREEK_TRANS[ord("ʹ")] = ""
_GREEK_TRANS[ord("ς")] = "σ"
_GREEK_TRANS[ord("ϲ")] = "σ"


def norm_greek(s, keep_accents=False):
    """Lower-case, strip punctuation and critical signs; by default strip all diacritics."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    if not keep_accents:
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().translate(_GREEK_TRANS)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^Ͱ-Ͽἀ-῿\s]", " ", s)
    return " ".join(s.split())


_HEB_FINAL = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


def norm_hebrew(s):
    """Consonantal skeleton: drop cantillation, points, maqaf->space, final forms -> medial."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    out = []
    for ch in s:
        o = ord(ch)
        if 0x0591 <= o <= 0x05C7 and ch not in "־׀׃׆":
            continue
        if ch in "־׀׃׆/":
            out.append(" ")
            continue
        out.append(ch)
    s = "".join(out).translate(_HEB_FINAL)
    s = re.sub(r"[^א-ת\s]", " ", s)
    return " ".join(s.split())


def norm(s, lang):
    return norm_greek(s) if lang == "greek" else norm_hebrew(s)


def is_greek_text(s):
    return bool(re.search(r"[Ͱ-Ͽἀ-῿]", s or ""))


def is_hebrew_text(s):
    return bool(re.search(r"[א-ת]", s or ""))
