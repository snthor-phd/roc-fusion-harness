"""Known-answer tests for the V1 checker (plan 12.1: must pass before any labeling).

Truths come from the source data themselves (SBLGNT, RP, MorphGNT, OSHB) and from Appendix A of the plan.
"""
import pytest

from p2.texts import load_corpus
from p2.v1 import Checker

C = load_corpus()

MARK9 = {"id": "Mark.9.14-9.29", "lang": "greek", "book": "Mark", "verses": [(9, v) for v in range(14, 30)],
         "focus": [(9, 23), (9, 24)], "window": [(9, v) for v in range(21, 27)]}
GEN1 = {"id": "Gen.1.1-1.5", "lang": "hebrew", "book": "Gen", "verses": [(1, v) for v in range(1, 6)],
        "focus": [(1, 1), (1, 2)], "window": [(1, v) for v in range(1, 6)]}


def lab(p, **cl):
    cl.setdefault("id", "t")
    return Checker(C, p).check(cl)[0]


# ---------------------------------------------------------------- quotes / editions (Greek)
def test_quote_critical_ok():
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="Τὸ εἰ δύνῃ, πάντα δυνατὰ τῷ πιστεύοντι", edition="critical") == "S"


def test_quote_fabricated_pilot_text():
    # Appendix A: Qwen3.6 gave ἐὰν δύνασαι πιστεῦσαι as the critical text
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="ἐὰν δύνασαι πιστεῦσαι", edition="critical") == "C"


def test_quote_byzantine_as_critical():
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="τὸ εἰ δύνασαι πιστεῦσαι", edition="critical") == "C"


def test_quote_byzantine_ok():
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="τὸ εἰ δύνασαι πιστεῦσαι", edition="byzantine") == "S"


def test_quote_unaccented_still_ok():
    assert lab(MARK9, type="quote", ref="Mark 9:24", orig="βοηθει μου τη απιστια", edition="critical") == "S"


def test_quote_invented_address():
    # Appendix A: invented vocative for the spirit in v.25
    assert lab(MARK9, type="quote", ref="Mark 9:25", orig="ὦ ταχὺ πνεῦμα", edition="critical") == "C"


def test_quote_true_address():
    assert lab(MARK9, type="quote", ref="Mark 9:25", orig="Τὸ ἄλαλον καὶ κωφὸν πνεῦμα", edition="critical") == "S"


def test_quote_misnumbered_verse():
    assert lab(MARK9, type="quote", ref="Mark 9:22", orig="βοήθει μου τῇ ἀπιστίᾳ", edition="critical") == "C"


def test_quote_too_short_is_undecided():
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="δύνῃ", edition="critical") is None


def test_quote_nonexistent_verse():
    assert lab(MARK9, type="quote", ref="Mark 9:51", orig="πάντα δυνατὰ", edition="critical") == "C"


# ---------------------------------------------------------------- variant direction
def test_byz_adds_meta_dakruon():
    assert lab(MARK9, type="variant", ref="Mark 9:24", orig="μετὰ δακρύων", variant_op="add",
               variant_tradition="byzantine") == "S"


def test_reversed_direction_meta_dakruon():
    # Appendix A: put μετὰ δακρύων in the critical text
    assert lab(MARK9, type="variant", ref="Mark 9:24", orig="μετὰ δακρύων", variant_op="add",
               variant_tradition="critical") == "C"


def test_critical_omits_pisteusai():
    assert lab(MARK9, type="variant", ref="Mark 9:23", orig="πιστεῦσαι", variant_op="omit",
               variant_tradition="critical") == "S"


def test_byz_adds_kurie():
    assert lab(MARK9, type="variant", ref="Mark 9:24", orig="κύριε", variant_op="add", variant_tradition="byzantine") == "S"


# ---------------------------------------------------------------- witnesses
def test_p75_for_mark_is_contradicted():
    assert lab(MARK9, type="witness", ref="Mark 9:23", witnesses=["𝔓75", "B"]) == "C"


def test_vaticanus_for_mark_undecided():
    assert lab(MARK9, type="witness", ref="Mark 9:23", witnesses=["B", "א"]) is None


def test_p46_for_mark_contradicted():
    assert lab(MARK9, type="witness", ref="Mark 9:24", witnesses=["P46"]) == "C"


def test_p45_for_mark9_compatible():
    assert lab(MARK9, type="witness", ref="Mark 9:24", witnesses=["P45"]) is None


# ---------------------------------------------------------------- morphology (Greek)
@pytest.mark.parametrize("form,parse,exp", [
    ("δύνῃ", "present middle indicative, 2nd person singular", "S"),
    ("δύνῃ", "present middle/passive (deponent) indicative 2nd singular", "S"),
    ("δύνῃ", "present passive indicative 2sg", "S"),
    ("δύνῃ", "present subjunctive 2nd singular", "C"),
    ("κράξας", "aorist active participle, nominative masculine singular", "S"),
    ("κράξας", "present active participle", "C"),
    ("βοήθει", "present active imperative, second person singular", "S"),
    ("βοήθει", "aorist active imperative 2sg", "C"),
    ("βοήθει", "present active indicative 3rd singular", "C"),
    ("πιστεύοντι", "present active participle dative masculine singular", "S"),
    ("πιστεύοντι", "aorist participle", "C"),
])
def test_greek_morph(form, parse, exp):
    ref = "Mark 9:23" if form in ("δύνῃ", "πιστεύοντι") else "Mark 9:24"
    assert lab(MARK9, type="morphology", ref=ref, orig=form, parse=parse) == exp


def test_morph_second_aorist_not_person():
    # '2nd aorist' must not be read as a person claim
    assert lab(MARK9, type="morphology", ref="Mark 9:24", orig="κράξας", parse="first aorist active participle") == "S"


# ---------------------------------------------------------------- occurrence and cross references
def _count(lemma, book=None):
    occ = C.lemma_index["greek"][__import__("p2.common", fromlist=["x"]).norm_greek(lemma)]
    return len([k for k in occ if book is None or k[0] == book])


def test_occurrence_count_true():
    n = _count("βοηθέω")
    assert lab(MARK9, type="occurrence", lemma="βοηθέω", count=n, count_scope="New Testament") == "S"


def test_occurrence_count_false():
    n = _count("βοηθέω")
    assert lab(MARK9, type="occurrence", lemma="βοηθέω", count=n + 25, count_scope="NT") == "C"


def test_occurrence_in_book():
    n = _count("πιστεύω", "Mark")
    assert lab(MARK9, type="occurrence", lemma="πιστεύω", count=n, count_scope="Mark") == "S"


def test_occurrence_in_verse():
    assert lab(MARK9, type="occurrence", lemma="πίστις", ref="Mark 11:22") == "S"
    assert lab(MARK9, type="occurrence", lemma="βοηθέω", ref="Mark 11:22") == "C"


def test_crossref_nonexistent_verse():
    assert lab(MARK9, type="crossref", ref="Mark 9:51") == "C"
    assert lab(MARK9, type="crossref", ref="Mark 17:3") == "C"


def test_crossref_existing_is_left_to_judge():
    assert lab(MARK9, type="crossref", ref="Mark 10:27") is None


def test_crossref_content():
    # v1.1: presence of the word does not verify the claim's content -> judge; absence still contradicts
    assert lab(MARK9, type="crossref", ref="Mark 6:5", orig="δύναμαι") is None
    assert lab(MARK9, type="crossref", ref="Mark 6:5", orig="βοηθέω") == "C"


# ---------------------------------------------------------------- v1.1 audit-driven cases (J24)
def test_count_off_by_one_goes_to_judge():
    n = _count("βοηθέω")
    assert lab(MARK9, type="occurrence", lemma="βοηθέω", count=n + 1, count_scope="NT") is None


def test_negated_occurrence_goes_to_judge():
    assert lab(MARK9, type="occurrence", lemma="πίστις", ref="Mark 11:22",
               claim="πίστις does not occur in Mark 11:22") is None


def test_multi_ref_occurrence_goes_to_judge():
    assert lab(MARK9, type="occurrence", lemma="πίστις", ref="Mark 11:22",
               claim="πίστις in Mark 11:22; 4:40; 10:52") is None


def test_edition_unstated_rp_only_goes_to_judge():
    assert lab(MARK9, type="quote", ref="Mark 9:23", orig="τὸ εἰ δύνασαι πιστεῦσαι", edition="unspecified") is None


def test_hebrew_person_on_participle_goes_to_judge():
    assert lab(GEN1, type="morphology", ref="Gen 1:2", orig="מְרַחֶפֶת", parse="Piel participle 3fs") is None


def test_hebrew_letter_difference_goes_to_judge():
    # one consonant dropped from a 3-word quote: not orthography in a consonantal comparison
    assert lab(GEN1, type="quote", ref="Genesis 1:1", orig="בְּרֵאשִׁית בָּרָא אֱלֹהִם") is None


# ---------------------------------------------------------------- Hebrew
def test_hebrew_quote_ok():
    assert lab(GEN1, type="quote", ref="Genesis 1:1", orig="בְּרֵאשִׁית בָּרָא אֱלֹהִים") == "S"


def test_hebrew_quote_bad():
    assert lab(GEN1, type="quote", ref="Genesis 1:1", orig="בְּרֵאשִׁית עָשָׂה אֱלֹהִים") == "C"


@pytest.mark.parametrize("form,ref,parse,exp", [
    ("בָּרָא", "Gen 1:1", "Qal perfect 3ms", "S"),
    ("בָּרָא", "Gen 1:1", "Piel perfect 3rd masculine singular", "C"),
    ("בָּרָא", "Gen 1:1", "Qal imperfect", "C"),
    ("וַיֹּאמֶר", "Gen 1:3", "Qal wayyiqtol 3ms", "S"),
    ("וַיֹּאמֶר", "Gen 1:3", "Qal perfect 3ms", "C"),
    ("יְהִי", "Gen 1:3", "Qal jussive 3rd masculine singular", "S"),
    ("יְהִי", "Gen 1:3", "Hiphil imperative", "C"),
    ("מְרַחֶפֶת", "Gen 1:2", "Piel participle feminine singular", "S"),
    ("מְרַחֶפֶת", "Gen 1:2", "Qal participle", "C"),
])
def test_hebrew_morph(form, ref, parse, exp):
    assert lab(GEN1, type="morphology", ref=ref, orig=form, parse=parse) == exp


def test_hebrew_occurrence():
    n = len([k for k in C.lemma_index["hebrew"]["H1254"] if k[0] == "Gen"])
    assert lab(GEN1, type="occurrence", lemma="בָּרָא", count=n, count_scope="Genesis") == "S"
    assert lab(GEN1, type="occurrence", lemma="בָּרָא", count=n + 30, count_scope="Genesis") == "C"


def test_lxx_quote():
    assert lab(GEN1, type="lxx", ref="Genesis 1:1", orig="ἐν ἀρχῇ ἐποίησεν ὁ θεὸς τὸν οὐρανὸν καὶ τὴν γῆν",
               edition="lxx") == "S"
    assert lab(GEN1, type="lxx", ref="Genesis 1:1", orig="ἐν τῷ πνεύματι ἁγίῳ κατέβη ὁ κύριος", edition="lxx") == "C"


def test_no_rule_is_none():
    assert lab(MARK9, type="attribution", claim="Metzger prefers δύνῃ") is None
