"""WP3: claim extractor (Claude Opus 4.8 per plan Section 5; frozen claim lists; no labeling).

x2.0 (12 Oct 11:05, DECISIONS J22): compact positional output and a 60-claim cap. The x1.0 object-per-claim schema
ran past 12,000 output tokens on 9-12k-token answers (truncated tool call, ~$0.19 each, unusable).
"""
import os
import re

from .common import RUNS, jload, jdump

EXTRACT_MODEL = "claude-opus-5-5"  # Opus 4.8 in Phase 0; Opus 5.5 via the Claude CLI from Stage B
EXTRACT_VERSION = "x2.1-cli"
MAX_TOKENS = 12000  # 9000 truncated one Hebrew G0 answer twice (long pointed-Hebrew quotes), 13 Oct
MAX_CLAIMS = 60

CLAIM_TYPES = ["quote", "variant", "witness", "morphology", "occurrence", "crossref", "lxx", "attribution",
               "lexical", "grammar", "historical", "other"]
TYPE_PRIORITY = {t: i for i, t in enumerate(CLAIM_TYPES)}

SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {"type": "array", "items": {"type": "integer"},
                     "description": "Numbers 1-8 of the requested sections that received substantive content."},
        "claims": {"type": "array", "maxItems": MAX_CLAIMS, "items": {"type": "array", "description":
                   "[section, type, ref, orig, claim, extra, span, hedged]"}},
    },
    "required": ["sections", "claims"],
}

SYSTEM = """You split one exegesis answer into atomic, checkable claims for a hallucination study. You do NOT judge whether claims are true and never correct them: record exactly what the answer asserts.

OUTPUT (record_claims tool). "sections": the numbered sections 1-8 that received substantive content. "claims": at most 60 claims, each a compact array
  [section, type, ref, orig, claim, extra, span, hedged]
- section: integer 1-8 (0 if outside the numbered sections)
- type: quote | variant | witness | morphology | occurrence | crossref | lxx | attribution | lexical | grammar | historical | other
- ref: verse the claim is about, e.g. "Mark 9:23" or "Genesis 1:1-2"; "" if none
- orig: Greek/Hebrew exactly as written in the answer (quotation, form, reading); "" if none
- claim: self-contained restatement, at most 15 words (for section-1 quote lines just "text of <ref>")
- extra: semicolon list of key=value, only keys that apply: ed=critical|byzantine|tr|mt|ketiv|qere|lxx ; parse=<parsing as stated> ; lemma=<lexical form> ; count=<integer> ; scope=<Mark|New Testament|Hebrew Bible|...> ; wit=<sigla comma-separated as written> ; op=add|omit|substitute|transpose ; trad=<tradition that adds/omits/reads: byzantine|critical|western|...>
- span: verbatim excerpt of the answer where the claim is made, at most 40 characters, copied exactly
- hedged: 1 if the answer flags the point as uncertain/from memory/speculative, else 0

WHAT TO EXTRACT, in this priority order if there are more than 60:
1 quote: in section 1 one claim per verse line of original-language text (ref = that verse, orig = the Greek/Hebrew as written, ed=critical for Greek or ed=mt for Hebrew unless the answer names another text). Skip the English translations.
2 variant: readings attributed to a tradition. "Byzantine adds X" -> orig X, op=add, trad=byzantine. "NA28 omits X" -> op=omit, trad=critical. "The Byzantine text reads Y" -> type quote, ed=byzantine. The preferred reading -> type variant with ed of the preferred text.
3 witness: manuscripts/versions/fathers cited for a reading (wit=..., ref, orig = the reading).
4 morphology: one claim per parsed form (orig = form, parse=..., lemma=..., ref).
5 occurrence: frequencies ("X occurs N times in Mark" -> lemma, count, scope) and "X also occurs in Mark 6:5" (lemma, ref).
6 crossref: what another verse says or contains; parallels, quotations, echoes.
7 lxx: Septuagint text or renderings (ed=lxx; ref = the Hebrew verse).
8 attribution: views or statements attributed to named scholars, commentaries, editions, ancient sources.
9 lexical, grammar, historical: specific checkable statements only.
Skip theological or interpretive judgments, value statements, summaries and restatements of the prompt. One assertion per claim; a list of witnesses for one reading is one claim."""


def build_request(run, passage, answer):
    lang = "Greek (New Testament)" if passage["lang"] == "greek" else "Hebrew (Old Testament)"
    user = ("Passage: %s (focus verses %s). Language: %s.\n\nAnswer to split into claims:\n<answer>\n%s\n</answer>"
            % (passage["pericope_ref"], passage["focus_ref"], lang, answer))
    return {
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
        "tools": [{"name": "record_claims", "description": "Record the extracted claims.", "input_schema": SCHEMA}],
        "tool_choice": {"type": "tool", "name": "record_claims"},
    }


_KEYMAP = {"ed": "edition", "parse": "parse", "lemma": "lemma", "count": "count", "scope": "count_scope",
           "wit": "witnesses", "op": "variant_op", "trad": "variant_tradition"}


def _expand(row):
    """Positional array -> claim dict (tolerates short rows and object rows)."""
    if isinstance(row, dict):
        return row
    row = list(row) + [None] * (8 - len(row))
    sec, typ, ref, orig, claim, extra, span, hedged = row[:8]
    ref, orig, claim, extra, span, typ = [None if x is None else str(x) for x in (ref, orig, claim, extra, span, typ)]
    c = {"section": int(sec) if str(sec).isdigit() else 0, "type": (typ or "other").strip().lower(),
         "ref": ref or None, "orig": orig or None, "claim": claim or "", "span": span or "",
         "hedged": bool(int(hedged)) if str(hedged).strip() in ("0", "1") else bool(hedged)}
    for part in re.split(r";\s*", extra or ""):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k not in _KEYMAP or not v:
            continue
        key = _KEYMAP[k]
        if key == "count":
            m = re.search(r"\d+", v)
            c[key] = int(m.group(0)) if m else None
        elif key == "witnesses":
            c[key] = [w.strip() for w in v.split(",") if w.strip()]
        else:
            c[key] = v.lower() if key in ("edition", "variant_op", "variant_tradition") else v
    if c["type"] not in TYPE_PRIORITY:
        c["type"] = "other"
    return c


def save(run_id, data, meta):
    rows = data.get("claims") or []
    claims = []
    for row in rows:
        try:
            claims.append(_expand(row))
        except Exception:
            continue
    for i, c in enumerate(claims):
        c["id"] = "c%03d" % (i + 1)
        for k in ("ref", "orig", "edition", "parse", "lemma", "count", "count_scope", "variant_op", "variant_tradition"):
            c.setdefault(k, None)
        c.setdefault("witnesses", [])
        c.setdefault("hedged", False)
    out = {"run_id": run_id, "extractor": EXTRACT_MODEL, "extract_version": EXTRACT_VERSION,
           "sections_answered": data.get("sections") or data.get("sections_answered") or [], "claims": claims,
           "capped": len(rows) >= MAX_CLAIMS, "meta": meta}
    jdump(out, os.path.join(RUNS, run_id, "claims.json"))
    return out
