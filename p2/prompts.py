"""Fixed eight-section exegesis templates (one per language) and source packets (G0 / G2 / judge)."""
from .common import ENGLISH
from .morph import decode_greek, decode_hebrew

RULES_GK = ("Rules: Cite Greek accurately with accents. Distinguish clearly between what the text says, what "
            "scholars commonly argue, and your own inference. Do not invent quotations, page numbers, or manuscript "
            "sigla you are not confident of; flag uncertainty explicitly. Be thorough but avoid padding.")
RULES_HE = ("Rules: Cite Hebrew accurately with vowel points. Distinguish clearly between what the text says, what "
            "scholars commonly argue, and your own inference. Do not invent quotations, page numbers, or manuscript "
            "sigla you are not confident of; flag uncertainty explicitly. Be thorough but avoid padding.")

TEMPLATE_GK = """You are a careful New Testament exegete working from the original languages. Produce a rigorous exegesis of {focus}, read within its pericope ({pericope}).

Work from the Greek text (NA28/UBS5 base text) and proceed in this order:

1. TEXT. Give the Greek of {window} with your own literal English translation, line by line.

2. TEXTUAL CRITICISM. Identify the significant textual variants in {focus} (compare the Alexandrian witnesses with the Byzantine / Textus Receptus tradition). For each variant: state the readings, the main witnesses as best you know them, which reading is preferred and why (internal and external evidence), and how the variant changes the sense. If you are unsure of a manuscript attestation, say so rather than guess.

3. GRAMMAR AND SYNTAX. Analyze the main clauses and constructions of {focus}: the function of key articles, conjunctions and cases, and any conditional, purpose, result, or participial constructions.

4. VERB MORPHOLOGY. Parse every finite verb, participle, and infinitive in {focus} (tense, voice, mood, person and number; for participles also case, number and gender), give its lexical form, and explain the significance of tense/aspect and mood where it matters.

5. KEY LEXEMES. Brief word studies for {lexemes}: how often each occurs in {book} and in the New Testament, at least two other passages where it occurs, and any relevant Septuagint background.

6. SEMITIC AND OLD TESTAMENT BACKGROUND. Comment on any plausible Aramaic or Hebrew substratum and any Old Testament quotations or echoes, clearly marking what is established versus speculative.

7. LITERARY CONTEXT. How does {focus} function within {pericope} and within {book}? Identify parallel passages (including any Synoptic parallels) and say how they differ.

8. THEOLOGICAL SYNTHESIS. What does the passage teach? Engage at least two contrasting interpretive positions found in the scholarly literature.

{rules}"""

TEMPLATE_HE = """You are a careful Old Testament exegete working from the original languages. Produce a rigorous exegesis of {focus}, read within its literary unit ({pericope}).

Work from the Hebrew Masoretic Text (Leningrad Codex / BHS) and proceed in this order:

1. TEXT. Give the pointed Hebrew of {window} with your own literal English translation, line by line.

2. TEXTUAL CRITICISM. Identify any ketiv/qere and the significant textual variants in {focus} (Masoretic Text, Dead Sea Scrolls, Samaritan Pentateuch where relevant, and the Septuagint). For each: state the readings, the main witnesses as best you know them, which reading is preferred and why, and how it changes the sense. If you are unsure of an attestation, say so rather than guess.

3. GRAMMAR AND SYNTAX. Analyze the clauses of {focus}: word order, waw sequences, construct chains, prepositions and particles, and any conditional, purpose, or result constructions.

4. VERB MORPHOLOGY. Parse every verb in {focus} (root, stem, conjugation, person, gender and number; for participles and infinitives the form and state), and explain the significance of the verbal forms where it matters.

5. KEY LEXEMES. Brief word studies for {lexemes}: how often each occurs in {book} and in the Hebrew Bible, at least two other passages where it occurs, and the main Septuagint equivalents.

6. SEPTUAGINT. Give the Septuagint Greek of {focus} as best you know it, say where the Septuagint differs from the Masoretic Text, and how the Septuagint renders the key terms.

7. LITERARY CONTEXT. How does {focus} function within {pericope} and within {book}? Identify parallel or related passages elsewhere in the Hebrew Bible and in the New Testament.

8. THEOLOGICAL SYNTHESIS. What does the passage teach? Engage at least two contrasting interpretive positions found in the scholarly literature.

{rules}"""

APP_KEY = ("Apparatus key: the reading before ']' is the SBLGNT text; editions: WH = Westcott-Hort 1881, Treg = "
           "Tregelles, NIV = Greek text behind the NIV (Holmes), RP = Robinson-Pierpont 2005 Byzantine text; '+' = adds, "
           "'–' = omits.")


def base_prompt(p):
    if p["lang"] == "greek":
        tpl, rules = TEMPLATE_GK, RULES_GK
    else:
        tpl, rules = TEMPLATE_HE, RULES_HE
    lex = p["lexemes"]
    if len(lex) >= 3:
        lexs = ", ".join(lex[:-1]) + ", and " + lex[-1]
    elif len(lex) == 2:
        lexs = lex[0] + " and " + lex[1]
    elif lex:
        lexs = lex[0]
    else:
        lexs = "the key terms"
    return tpl.format(focus=p["focus_ref"], pericope=p["pericope_ref"], window=p["window_ref"],
                      lexemes=lexs, book=ENGLISH[p["book"]], rules=rules)


def _refs(p, which):
    return [(p["book"], c, v) for c, v in p[which]]


def _label(b, c, v):
    return "%s %d:%d" % (ENGLISH[b], c, v)


def greek_text_block(p, corpus):
    lines = ["SBLGNT text of %s:" % p["pericope_ref"]]
    for k in _refs(p, "verses"):
        lines.append("[V:%s] %s" % (_label(*k), corpus.gk_text.get(k, "")))
    return "\n".join(lines)


def greek_app_block(p, corpus):
    lines = ["SBLGNT apparatus for %s. %s" % (p["pericope_ref"], APP_KEY)]
    for k in _refs(p, "verses"):
        for i, u in enumerate(corpus.app.get(k, [])):
            lines.append("[A:%s#%d] %s" % (_label(*k), i + 1, u))
    if len(lines) == 1:
        lines.append("(no apparatus entries)")
    return "\n".join(lines)


def greek_byz_block(p, corpus):
    lines = ["Robinson-Pierpont 2005 Byzantine text of %s:" % p["window_ref"]]
    for k in _refs(p, "window"):
        lines.append("[B:%s] %s" % (_label(*k), corpus.rp_text.get(k, "")))
    return "\n".join(lines)


def greek_morph_block(p, corpus, which="window"):
    lines = ["MorphGNT morphology for %s (form | lexical form | parsing):" % p[which + "_ref"]]
    for k in _refs(p, which):
        for w in corpus.morph.get(k, []):
            lines.append("[M:%s] %s | %s | %s" % (_label(*k), w["word"], w["lemma"], decode_greek(w["pos"], w["parse"])))
    return "\n".join(lines)


def hebrew_text_block(p, corpus):
    lines = ["Westminster Leningrad Codex text of %s:" % p["pericope_ref"]]
    for k in _refs(p, "verses"):
        lines.append("[V:%s] %s" % (_label(*k), corpus.he_text.get(k, "")))
    return "\n".join(lines)


def hebrew_kq_block(p, corpus):
    lines = ["Ketiv/qere in %s:" % p["pericope_ref"]]
    for k in _refs(p, "verses"):
        for i, (kt, q) in enumerate(corpus.he_kq.get(k, [])):
            lines.append("[A:%s#%d] ketiv %s ; qere %s" % (_label(*k), i + 1, kt, q))
    if len(lines) == 1:
        lines.append("(no ketiv/qere in this unit)")
    lines.append("(No critical apparatus of the Hebrew Bible is available in this source packet.)")
    return "\n".join(lines)


def hebrew_morph_block(p, corpus, which="window"):
    lines = ["OSHB morphology for %s (form | Strong's number and lemma | parsing):" % p[which + "_ref"]]
    for k in _refs(p, which):
        for w in corpus.he_words.get(k, []):
            lem = w["lemma"].split("/")[-1].strip()
            sid = "H" + lem.split()[0] if lem and lem.split()[0].isdigit() else ""
            lemma_txt = corpus.strong.get(sid, "")
            tag = " (qere)" if w.get("qere") else (" (ketiv)" if w.get("ketiv") else "")
            lines.append("[M:%s] %s%s | %s %s | %s" % (_label(*k), w["text"], tag, sid, lemma_txt, decode_hebrew(w["morph"])))
    return "\n".join(lines)


def lxx_ref(book, c, v):
    """MT -> Swete LXX reference (Psalms renumbering; Jeremiah 25-52, Nehemiah, Ecclesiastes unavailable)."""
    if book == "Ps":
        if c <= 8 or c >= 148:
            return (book, c, v)
        if c == 9:
            return (book, 9, v)
        if c == 10:
            return None
        if 11 <= c <= 113:
            return (book, c - 1, v)
        if c in (114, 115):
            return None
        if c == 116:
            return (book, 114, v) if v <= 9 else (book, 115, v - 9)
        if 117 <= c <= 146:
            return (book, c - 1, v)
        if c == 147:
            return (book, 146, v) if v <= 11 else (book, 147, v - 11)
    if book == "Jer" and c >= 25:
        return None
    if book in ("Neh", "Eccl"):
        return None
    return (book, c, v)


def lxx_block(p, corpus):
    lines = ["Septuagint (Swete edition; aligned to MT verses where the alignment is reliable) for %s:" % p["window_ref"]]
    for k in _refs(p, "window"):
        lk = lxx_ref(*k)
        txt = corpus.lxx.get(lk) if lk else None
        lines.append("[L:%s] %s" % (_label(*k), txt if txt else "(alignment not available)"))
    return "\n".join(lines)


def g3_extra(p, corpus):
    """G3 additions (Stage B): manuscript-contents table (Greek), lemma occurrence index, Septuagint of the window
    (Hebrew). The Swete LXX is not lemmatized, so no LXX occurrence counts are given for Greek lexemes."""
    from .judge import witness_rows, occurrence_rows
    parts = [witness_rows(p), occurrence_rows(p, corpus)]
    if p["lang"] == "greek":
        parts.append("Septuagint occurrence counts: not available in this packet (the Swete text is not lemmatized).")
    else:
        parts.append(lxx_block(p, corpus))
    return parts


def grounding_packet(p, corpus, level):
    """G0: none. G2: text + apparatus + morphology (Greek adds the Byzantine text of the window).
    G3: G2 + witness, occurrence and LXX tables."""
    if level == "G0":
        return ""
    if p["lang"] == "greek":
        parts = [greek_text_block(p, corpus), greek_app_block(p, corpus), greek_byz_block(p, corpus),
                 greek_morph_block(p, corpus, "window")]
    else:
        parts = [hebrew_text_block(p, corpus), hebrew_kq_block(p, corpus), hebrew_morph_block(p, corpus, "window")]
    if level == "G3":
        parts += g3_extra(p, corpus)
    return "\n\n".join(parts)


def generation_prompt(p, corpus, grounding):
    base = base_prompt(p)
    if grounding == "G0":
        return base
    packet = grounding_packet(p, corpus, grounding)
    if grounding == "G3":
        return (base + "\n\nSOURCE MATERIALS. The following machine-readable source data are authoritative for the "
                "text, the apparatus, the morphology, manuscript contents, word occurrence counts and the Septuagint. "
                "Base your quotations, parsing, counts and witnesses on them; do not contradict them.\n\n" + packet)
    return (base + "\n\nSOURCE MATERIALS. The following machine-readable source data are authoritative for the text, "
            "the apparatus, and the morphology. Base your quotations and parsing on them; do not contradict them.\n\n"
            + packet)


def judge_packet(p, corpus):
    """Ground-truth packet for the judge (cached per passage)."""
    if p["lang"] == "greek":
        parts = [greek_text_block(p, corpus), greek_app_block(p, corpus), greek_byz_block(p, corpus),
                 greek_morph_block(p, corpus, "window")]
    else:
        parts = [hebrew_text_block(p, corpus), hebrew_kq_block(p, corpus), hebrew_morph_block(p, corpus, "window"),
                 lxx_block(p, corpus)]
    return "\n\n".join(parts)
