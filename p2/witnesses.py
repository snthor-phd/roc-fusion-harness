"""Manuscript-contents table (open data: Wikipedia lists of NT papyri and uncials, CC BY-SA).

Used by V1 only to decide witness claims that are impossible on content grounds (e.g. Papyrus 75 for Mark).
Lacunae are ignored, so 'contains' is generous and contradictions are conservative.
"""
import os
import re

from .common import RAWDATA, STATE, NT_IDS, book_id, jdump, jload

GOSPELS = ["Matt", "Mark", "Luke", "John"]
PAUL = ["Rom", "1Cor", "2Cor", "Gal", "Eph", "Phil", "Col", "1Thess", "2Thess", "1Tim", "2Tim", "Titus", "Phlm", "Heb"]
CATHOLIC = ["Jas", "1Pet", "2Pet", "1John", "2John", "3John", "Jude"]

_CATS = [(r"new testament|\bnt\b", NT_IDS), (r"gospel|\bgosp\b|\bevangel", GOSPELS), (r"\bacts\b", ["Acts"]),
         (r"pauline|\bpaul\b", PAUL), (r"catholic|general epistles|\bce\b", CATHOLIC),
         (r"revelation|apocalypse|\brev\b", ["Rev"])]


def _clean(cell):
    c = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", cell, flags=re.S)
    c = re.sub(r"\{\{[^}]*\}\}", lambda m: m.group(0), c)
    c = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"<[^>]+>", " ", c)
    c = c.replace("&nbsp;", " ").replace("'''", "").replace("''", "")
    return " ".join(c.split())


def _rows(wikitext):
    """Split wiki tables into rows of cells (rowspan prefixes removed)."""
    rows = []
    for block in re.split(r"\n\|-[^\n]*\n", wikitext):
        cells = []
        for line in block.split("\n"):
            if line.startswith("|") and not line.startswith("|}") and not line.startswith("|+"):
                parts = line[1:].split("||")
                for p in parts:
                    p = re.sub(r"^\s*(rowspan|colspan|bgcolor|style)\s*=\s*\"?[^|\"]*\"?\s*\|", "", p)
                    cells.append(p.strip())
        if cells:
            rows.append(cells)
    return rows


def _parse_contents(text):
    """Return {book: [(c1, c2)] or None(all)} from a contents cell."""
    t = _clean(text)
    out = {}
    low = t.lower()
    for rx, books in _CATS:
        if re.search(rx, low):
            for b in books:
                out[b] = None
    # explicit book + chapter ranges: 'Luke 3:18-4:2, 4:34-5:10 John 1:1-5:3'
    cur = None
    for tok in re.finditer(r"((?:[123]\s?)?[A-Z][a-z]+)\s*(\d+)?|(\d+)(?::\d+[a-z]?)?\s*[-–]\s*(\d+)(?=:)|(\d+):\d+", t):
        name, ch = tok.group(1), tok.group(2)
        if name:
            b = book_id(name)
            if b in NT_IDS:
                cur = b
                if out.get(cur, 0) is None:
                    continue
                out.setdefault(cur, [])
                if ch:
                    out[cur].append((int(ch), int(ch)))
            continue
        if cur is None or out.get(cur) is None:
            continue
        if tok.group(3) and tok.group(4):
            out[cur].append((int(tok.group(3)), int(tok.group(4))))
        elif tok.group(5):
            out[cur].append((int(tok.group(5)), int(tok.group(5))))
    # chapter ranges like '3:18-4:2' : make each listed range cover start..end chapter
    for b, rng in list(out.items()):
        if rng == []:
            out[b] = None  # book named but no chapters parsed: treat as whole book (generous)
    return out


def build():
    table = {}
    pap = open(os.path.join(RAWDATA, "List_of_New_Testament_papyri.wiki"), encoding="utf-8").read()
    for cells in _rows(pap):
        m = re.search(r"\{\{Papyrus link\|(\d+)\}\}", cells[0]) if cells else None
        if not m or len(cells) < 3:
            continue
        n = m.group(1)
        contents = cells[2]
        # chapter ranges in papyri: expand 'Luke 3:18-4:2' properly
        books = _parse_ranges(contents)
        table["P" + n] = {"name": "Papyrus %s" % n, "contents_raw": _clean(contents)[:300], "books": books}
    unc = open(os.path.join(RAWDATA, "List_of_New_Testament_uncials.wiki"), encoding="utf-8").read()
    for cells in _rows(unc):
        if len(cells) < 5:
            continue
        num = _clean(cells[0])
        if not re.match(r"^0\d{1,3}[a-z]?$", num):
            continue
        sig = _clean(cells[1])
        name = _clean(cells[2])
        contents = cells[4]
        books = _parse_contents(contents)
        table[num] = {"name": name, "siglum": sig, "contents_raw": _clean(contents)[:300], "books": books}
    jdump(table, os.path.join(STATE, "witnesses.json"))
    return table


def _parse_ranges(cell):
    """Papyri contents: 'Luke 3:18-4:2, 4:34-5:10, 5:37-17:35<br>John 1:1-5:3a' -> {book: [(c1,c2)...]}"""
    t = _clean(cell)
    out = {}
    cur = None
    pos = 0
    for m in re.finditer(r"((?:[123]\s?)?[A-Z][a-z]+)|(\d+)(?::(\d+)[a-z]?)?(?:\s*[-–]\s*(\d+)(?::(\d+))?)?", t):
        if m.group(1):
            b = book_id(m.group(1))
            if b in NT_IDS:
                cur = b
                out.setdefault(cur, [])
            elif m.group(1).lower() in ("frg", "fragment", "frgs"):
                pass
            continue
        if cur is None or not m.group(2):
            continue
        c1 = int(m.group(2))
        if m.group(3) is None and m.group(4) is None:
            out[cur].append((c1, c1))
        elif m.group(4) and m.group(5):
            out[cur].append((c1, int(m.group(4))))  # 3:18-4:2 -> chapters 3..4
        else:
            out[cur].append((c1, c1))  # 3:18-25 within chapter 3
    for b in list(out):
        if not out[b]:
            out[b] = None
    return out


_GREEK_SIGLA = {"א": "01", "ℵ": "01", "aleph": "01", "sinaiticus": "01", "a": "02", "alexandrinus": "02", "b": "03",
                "vaticanus": "03", "c": "04", "ephraemi": "04", "w": "032", "washingtonianus": "032", "freer": "032",
                "θ": "038", "koridethi": "038", "theta": "038", "l": "019", "regius": "019", "δ": "037",
                "ψ": "044", "psi": "044", "ξ": "040", "zacynthius": "040", "z": "035", "n": "022", "σ": "042",
                "φ": "043", "e": "07", "f": "09", "g": "011", "h": "013", "k": "017", "m": "021", "s": "028",
                "u": "030", "v": "031", "γ": "036", "λ": "039", "π": "041", "ω": "045", "t": "029", "x": "033",
                "y": "034", "p": "025", "q": "026", "r": "027", "bezae": "05", "claromontanus": "06"}
# letters whose meaning depends on the corpus (Gospels vs Paul vs Acts)
_PAUL_ALT = {"d": "06", "f": "010", "g": "012", "e": "08", "h": "015", "k": "018", "l": "020", "p": "025"}


def resolve(siglum, book):
    """Map a siglum as written by a model to Gregory-Aland ids (list; empty if unknown)."""
    s = (siglum or "").strip()
    s = s.replace("𝔓", "P").replace("ᴘ", "P")
    s = s.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    s = re.sub(r"[\s\.\^]", "", s)
    m = re.match(r"^(?:P|p|Papyrus|papyrus)(\d{1,3})$", s)
    if m:
        return ["P" + str(int(m.group(1)))]
    m = re.match(r"^0(\d{1,3})$", s)
    if m:
        return ["0" + m.group(1)]
    low = re.sub(r"\(.*?\)|\*|[0-9]+$|codex", "", s.lower())
    low = low.strip()
    if book in PAUL and low in _PAUL_ALT:
        return [_PAUL_ALT[low]]
    if low == "d" and book in GOSPELS + ["Acts"]:
        return ["05"]
    if low in _GREEK_SIGLA:
        return [_GREEK_SIGLA[low]]
    return []


def contains(table, ga, book, chapter):
    """True/False if decidable; None if unknown witness or no contents info."""
    e = table.get(ga)
    if not e:
        return None
    books = e.get("books") or {}
    if not books:
        return None
    if book not in books:
        return False
    rngs = books[book]
    if rngs is None or chapter is None:
        return True
    return any(a <= chapter <= b for a, b in rngs)


def load():
    t = jload(os.path.join(STATE, "witnesses.json"))
    if t is None:
        t = build()
    for k, v in t.items():
        if v.get("books"):
            v["books"] = {b: (None if r is None else [tuple(x) for x in r]) for b, r in v["books"].items()}
    return t
