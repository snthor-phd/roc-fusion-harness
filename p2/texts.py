"""Loaders for the open source-packet data (WP1).

SBLGNT text + paragraphs + apparatus, MorphGNT, Robinson-Pierpont (with NA27 variants),
OSHB/WLC (text, morphology, ketiv/qere, petuhah/setumah), Strong's Hebrew lemmas,
Swete LXX, OpenBible cross-reference votes.
"""
import csv
import io
import os
import pickle
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

from .common import (NT_BOOKS, OT_BOOKS, NT_IDS, OT_IDS, OPENDATA, RAWDATA, STATE,
                     norm_greek, norm_hebrew)

CACHE = os.path.join(STATE, "corpus_cache.pkl")
CACHE_VERSION = 4


class Corpus:
    """All open text data, keyed by (book, chapter, verse)."""

    def __init__(self):
        self.gk_text = {}        # SBLGNT verse text (with critical signs)
        self.gk_paras = []       # list of (book, [ (c,v), ... ]) SBLGNT paragraphs
        self.app = {}            # (b,c,v) -> list of apparatus unit strings
        self.morph = defaultdict(list)  # (b,c,v) -> [dict(text, word, norm, lemma, pos, parse)]
        self.rp_text = {}        # Robinson-Pierpont 2005 Byzantine text
        self.na_text = {}        # NA27 text reconstructed from RP variant notes
        self.he_words = defaultdict(list)  # (b,c,v) -> [dict(text, lemma, morph, ketiv, qere)]
        self.he_text = {}        # WLC verse text (pointed, with maqaf)
        self.he_paras = []       # (book, [(c,v)...]) units between petuhah/setumah
        self.he_kq = defaultdict(list)  # (b,c,v) -> [(ketiv, qere)]
        self.he_notes = defaultdict(int)  # (b,c,v) -> count of OSHB variant notes
        self.he_aramaic = set()  # (b,c,v) verses containing Aramaic words
        self.strong = {}         # 'H1254' -> pointed lemma
        self.lxx = {}            # (book(MT id), c, v) -> Swete text, LXX versification
        self.xref_votes = defaultdict(int)  # (b,c,v) -> summed OpenBible cross-ref votes
        self.verse_order = {}    # book -> [(c,v)...] in order
        self.lemma_index = defaultdict(lambda: defaultdict(list))  # lang -> normlemma|Hnnnn -> [(b,c,v)]

    # ------------------------------------------------------------------ Greek
    def load_sblgnt(self):
        base = os.path.join(OPENDATA, "LogosBible_SBLGNT", "data", "sblgnt")
        for b in NT_BOOKS:
            osis = b[0]
            path = os.path.join(base, "text", osis + ".txt")
            order = []
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if "\t" not in line:
                        continue
                    ref, txt = line.rstrip("\n").split("\t", 1)
                    m = re.match(r".*?(\d+):(\d+)$", ref.strip())
                    if not m:
                        continue
                    key = (osis, int(m.group(1)), int(m.group(2)))
                    self.gk_text[key] = txt.strip()
                    order.append(key[1:])
            self.verse_order[osis] = order
            root = ET.parse(os.path.join(base, "xml", osis + ".xml")).getroot()
            for p in root.iter("p"):
                vs = []
                for vn in p.iter("verse-number"):
                    m = re.match(r".*?(\d+):(\d+)$", vn.get("id", ""))
                    if m:
                        vs.append((int(m.group(1)), int(m.group(2))))
                if vs:
                    self.gk_paras.append((osis, vs))

    def load_apparatus(self):
        base = os.path.join(OPENDATA, "sblapp")
        for b in NT_BOOKS:
            path = os.path.join(base, b[2] + "-APP.txt")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    if "\t" not in line:
                        continue
                    ref, body = line.rstrip("\n").split("\t", 1)
                    m = re.match(r".*?(\d+):(\d+)$", ref.strip())
                    if not m:
                        continue
                    units = [u.strip().replace("\xa0", " ") for u in body.split("•") if u.strip()]
                    self.app[(b[0], int(m.group(1)), int(m.group(2)))] = units

    def load_morphgnt(self):
        base = os.path.join(OPENDATA, "morphgnt_sblgnt")
        for b in NT_BOOKS:
            path = os.path.join(base, b[1] + "-morphgnt.txt")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    ref, pos, parse, text, word, nrm, lemma = parts[:7]
                    key = (b[0], int(ref[2:4]), int(ref[4:6]))
                    self.morph[key].append({"text": text, "word": word, "norm": nrm, "lemma": lemma,
                                            "pos": pos, "parse": parse})
                    self.lemma_index["greek"][norm_greek(lemma)].append(key)

    def load_rp(self):
        base = os.path.join(OPENDATA, "byztxt_byzantine-majority-text", "csv-unicode", "ccat", "with-variants")
        for b in NT_BOOKS:
            path = os.path.join(base, b[3] + ".csv")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    key = (b[0], int(row["chapter"]), int(row["verse"]))
                    raw = row["text"]
                    self.rp_text[key] = " ".join(re.sub(r"\{[^}]*\}", " ", raw).replace("¶", "").split())
                    self.na_text[key] = apply_na_variants(raw)

    # ----------------------------------------------------------------- Hebrew
    def load_oshb(self):
        base = os.path.join(OPENDATA, "openscriptures_morphhb", "wlc")
        ns = "http://www.bibletechnologies.net/2003/OSIS/namespace"
        for b in OT_BOOKS:
            osis = b[0]
            root = ET.parse(os.path.join(base, osis + ".xml")).getroot()
            order = []
            cur_para = []
            for verse in root.iter("{%s}verse" % ns):
                oid = verse.get("osisID")
                _, c, v = oid.split(".")
                key = (osis, int(c), int(v))
                order.append(key[1:])
                words = []
                texts = []
                para_end = False
                for el in verse:
                    tag = el.tag.split("}")[1]
                    if tag == "w":
                        w = _oshb_word(el)
                        words.append(w)
                        texts.append(w["text"])
                        if w["morph"].startswith("A"):
                            self.he_aramaic.add(key)
                    elif tag == "seg":
                        t = el.get("type")
                        if t in ("x-pe", "x-samekh"):
                            para_end = True
                        elif t == "x-maqqef" and texts:
                            texts[-1] += "־"
                    elif tag == "note":
                        if el.get("type") == "variant":
                            self.he_notes[key] += 1
                            q = [_oshb_word(x) for x in el.iter("{%s}w" % ns)]
                            k = [w for w in words if w.get("ketiv")]
                            if q:
                                self.he_kq[key].append((k[-1]["text"] if k else "", " ".join(x["text"] for x in q)))
                                for x in q:
                                    x["qere"] = True
                                    words.append(x)
                self.he_words[key] = words
                self.he_text[key] = " ".join(texts).replace("־ ", "־")
                for w in words:
                    lem = w["lemma"].split("/")[-1].strip()
                    if lem:
                        self.lemma_index["hebrew"]["H" + lem.split()[0]].append(key)
                cur_para.append(key[1:])
                if para_end:
                    self.he_paras.append((osis, cur_para))
                    cur_para = []
            if cur_para:
                self.he_paras.append((osis, cur_para))
            self.verse_order[osis] = order

    def load_strong(self):
        path = os.path.join(OPENDATA, "openscriptures_HebrewLexicon", "HebrewStrong.xml")
        ns = "{http://openscriptures.github.com/morphhb/namespace}"
        root = ET.parse(path).getroot()
        for e in root.iter(ns + "entry"):
            w = e.find(ns + "w")
            if w is not None and w.text:
                self.strong[e.get("id")] = w.text

    def load_lxx(self):
        base = os.path.join(OPENDATA, "nathans_lxx-swete", "data")
        files = {}
        for b in OT_BOOKS:
            if b[3] and b[0] != "Neh":
                files.setdefault(b[3], []).append(b[0])
        for fname, mt_books in files.items():
            path = os.path.join(base, fname)
            if not os.path.exists(path):
                continue
            verses = defaultdict(list)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue
                    nums = parts[0].split(".")
                    if len(nums) < 3:
                        continue
                    try:
                        c, v = int(nums[-2]), int(nums[-1])
                    except ValueError:
                        continue
                    verses[(c, v)].append(parts[1].strip())
            mt = mt_books[0]
            for (c, v), ws in verses.items():
                self.lxx[(mt, c, v)] = " ".join(ws)

    # ------------------------------------------------------------ familiarity
    def load_xref_votes(self):
        path = os.path.join(RAWDATA, "openbible_xrefs.zip")
        if not os.path.exists(path):
            return
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".txt")][0]
            data = io.TextIOWrapper(z.open(name), encoding="utf-8")
            next(data)
            for line in data:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    votes = int(parts[2])
                except ValueError:
                    continue
                if votes <= 0:
                    continue
                for ref in (parts[0], parts[1].split("-")[0]):
                    k = _openbible_key(ref)
                    if k:
                        self.xref_votes[k] += votes

    # --------------------------------------------------------------- helpers
    def lang(self, book):
        return "greek" if book in NT_IDS else "hebrew"

    def verse_exists(self, book, c, v):
        if book in NT_IDS:
            return (book, c, v) in self.gk_text or (book, c, v) in self.rp_text
        return (book, c, v) in self.he_text

    def chapter_exists(self, book, c):
        return any(cc == c for cc, _ in self.verse_order.get(book, []))

    def verses_in(self, ref):
        """Expand a parsed ref dict into a list of (b,c,v) keys that exist."""
        if not ref:
            return []
        b = ref["book"]
        order = self.verse_order.get(b, [])
        if ref["v1"] is None:
            return [(b, c, v) for c, v in order if ref["c1"] <= c <= (ref["c2"] or ref["c1"])]
        lo, hi = (ref["c1"], ref["v1"]), (ref["c2"], ref["v2"])
        return [(b, c, v) for c, v in order if lo <= (c, v) <= hi]

    def edition_texts(self, key):
        """Greek editions available for a verse: name -> text."""
        out = {}
        if key in self.gk_text:
            out["SBLGNT"] = self.gk_text[key]
        if key in self.na_text:
            out["NA27"] = self.na_text[key]
        if key in self.rp_text:
            out["RP"] = self.rp_text[key]
        return out

    def hebrew_lemma_ids(self, lemma_text):
        """Strong's ids whose lemma has the same consonants as lemma_text."""
        n = norm_hebrew(lemma_text).replace(" ", "")
        if not hasattr(self, "_strong_by_cons"):
            d = defaultdict(list)
            for k, w in self.strong.items():
                d[norm_hebrew(w).replace(" ", "")].append(k)
            self._strong_by_cons = dict(d)
        return self._strong_by_cons.get(n, [])


def _oshb_word(el):
    t = (el.text or "").replace("/", "")
    return {"text": t, "lemma": el.get("lemma", ""), "morph": el.get("morph", ""),
            "ketiv": el.get("type") == "x-ketiv", "qere": False}


_OB = {"Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth", "1Sam", "2Sam", "1Kgs", "2Kgs", "1Chr", "2Chr",
       "Ezra", "Neh", "Esth", "Job", "Ps", "Prov", "Eccl", "Song", "Isa", "Jer", "Lam", "Ezek", "Dan", "Hos", "Joel",
       "Amos", "Obad", "Jonah", "Mic", "Nah", "Hab", "Zeph", "Hag", "Zech", "Mal", "Matt", "Mark", "Luke", "John",
       "Acts", "Rom", "1Cor", "2Cor", "Gal", "Eph", "Phil", "Col", "1Thess", "2Thess", "1Tim", "2Tim", "Titus",
       "Phlm", "Heb", "Jas", "1Pet", "2Pet", "1John", "2John", "3John", "Jude", "Rev"}


def _openbible_key(ref):
    p = ref.split(".")
    if len(p) != 3 or p[0] not in _OB:
        return None
    try:
        return (p[0], int(p[1]), int(p[2]))
    except ValueError:
        return None


def apply_na_variants(raw):
    """Reconstruct the NA27 text from an RP 'with-variants' line.

    Format: '... δύνασαι {NA δύνασαι πιστεῦσαι > δύνῃ } πιστεῦσαι ...' : the brace follows the
    first word of the RP span; the span replaces len(span) words starting at that word.
    """
    raw = raw.replace("¶", " ")
    tokens = re.findall(r"\{[^}]*\}|[^\s{}]+", raw)
    out = []
    skip = 0
    for t in tokens:
        if t.startswith("{"):
            m = re.match(r"\{\s*NA\s+(.*?)\s*>\s*(.*?)\s*\}", t)
            if not m:
                continue
            n = len(m.group(1).split())
            na = m.group(2).strip()
            if out:
                out.pop()
            if na not in ("-", "–"):
                out.extend(na.replace("[", "").replace("]", "").split())
            skip = max(0, n - 1)
            continue
        if skip:
            skip -= 1
            continue
        out.append(t)
    return " ".join(out)


def apparatus_readings(unit):
    """Parse one SBLGNT apparatus unit 'δύνῃ WH Treg NIV ] δύνασαι πιστεῦσαι RP'.

    Returns [(reading, [editions])...]; '+ x' means addition, '–' omission (kept literally).
    """
    out = []
    if "]" not in unit:
        return out
    lhs, rhs = unit.split("]", 1)
    for part in [lhs] + rhs.split(";"):
        toks = part.strip().split()
        eds = []
        while toks and re.match(r"^(WH|Treg|NIV|RP|NA|Holmes|WHmarg|Tregmarg|NIVmarg|RPmarg|ECM|UBS)$", toks[-1]):
            eds.insert(0, toks.pop())
        if eds:
            out.append((" ".join(toks), eds))
    return out


def load_corpus(force=False):
    if not force and os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as f:
                obj = pickle.load(f)
            if getattr(obj, "_v", 0) == CACHE_VERSION:
                return obj
        except Exception:
            pass
    c = Corpus()
    c.load_sblgnt()
    c.load_apparatus()
    c.load_morphgnt()
    c.load_rp()
    c.load_oshb()
    c.load_strong()
    c.load_lxx()
    c.load_xref_votes()
    c._v = CACHE_VERSION
    c.lemma_index = {k: dict(v) for k, v in c.lemma_index.items()}
    c.morph = dict(c.morph)
    c.he_words = dict(c.he_words)
    c.he_kq = dict(c.he_kq)
    c.he_notes = dict(c.he_notes)
    c.xref_votes = dict(c.xref_votes)
    try:
        with open(CACHE, "wb") as f:
            pickle.dump(c, f)
    except OSError:
        pass
    return c
