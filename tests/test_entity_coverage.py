"""Tests for `abstention_features.entity_coverage`.

Fast tests use a fake NER callable (just needs `nlp(text) -> object with
.ents`, each entity exposing `.text`, matching spaCy's `Language`
interface) with fixed, hand-picked entity lists, so every assertion
checks an actual expected coverage fraction, not just "runs without
crashing". One @pytest.mark.slow test at the bottom loads the real
`en_core_web_sm` pipeline -- run explicitly with `pytest -m slow`
(requires `python -m spacy download en_core_web_sm` to have been run
once, see README.md).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from types import SimpleNamespace

import pytest

from abstention_features.entity_coverage import (
    FEATURE_NAMES,
    FUZZY_MATCH_THRESHOLD,
    NO_QUESTION_ENTITIES_SENTINEL,
    _entity_is_covered,
    _is_fuzzy_match,
    _normalize_entity,
    compute_entity_coverage,
    extract_entities,
    extract_entity_coverage_features,
)


def _fake_doc(entity_texts: list[str]):
    return SimpleNamespace(ents=[SimpleNamespace(text=t) for t in entity_texts])


class FakeNLP:
    """Stand-in for a loaded spaCy `Language` pipeline: returns a fixed
    list of entity strings for each exact input text, and records every
    call it received (by input text, in order)."""

    def __init__(self, responses: dict[str, list[str]]):
        self._responses = responses
        self.calls: list[str] = []

    def __call__(self, text: str):
        self.calls.append(text)
        return _fake_doc(self._responses.get(text, []))


class ExplodingNLP:
    """Fails the test if ever invoked -- proves the empty-chunks path
    never reaches the NER pipeline."""

    def __call__(self, *args, **kwargs):
        raise AssertionError("nlp() should not be called for empty chunks")


# ---------------------------------------------------------------------------
# _normalize_entity
# ---------------------------------------------------------------------------


def test_normalize_entity_lowercases_strips_and_drops_periods():
    assert _normalize_entity("  U.S. ") == "us"
    assert _normalize_entity("US") == "us"
    assert _normalize_entity("Marseille") == "marseille"


# ---------------------------------------------------------------------------
# _is_fuzzy_match / fuzzy boundary
# ---------------------------------------------------------------------------


def test_fuzzy_match_close_spelling_variant_is_above_threshold():
    # "marseille" vs "marseilles" -- ratio ~0.947, comfortably above 0.8.
    assert _is_fuzzy_match("marseille", "marseilles") is True


def test_fuzzy_match_meaningfully_different_strings_below_threshold():
    # "france" vs "french" -- ratio ~0.667, below 0.8 despite sharing
    # most letters; these are different entities and should not match.
    assert _is_fuzzy_match("france", "french") is False


def test_fuzzy_match_boundary_is_strictly_greater_than_not_gte():
    # "mali" vs "malibu" sits at *exactly* FUZZY_MATCH_THRESHOLD (0.8).
    # The comparison is strict (`>`, not `>=`), so this must NOT count
    # as a match -- same word-boundary caution as Phase 1's answer-span
    # substring handling, applied here to entity matching.
    ratio = 4 / 5  # len("mali")=4 matching chars, total len 4+6=10 -> 8/10
    assert ratio == pytest.approx(0.8)
    assert _is_fuzzy_match("mali", "malibu", threshold=FUZZY_MATCH_THRESHOLD) is False


def test_entity_is_covered_true_for_exact_match_after_normalization():
    # "U.S." vs "US" -- not close enough by raw fuzzy ratio (~0.667) but
    # normalizes to an exact match ("us" == "us").
    assert _entity_is_covered("U.S.", ["US"]) is True


def test_entity_is_covered_true_for_fuzzy_spelling_variant():
    assert _entity_is_covered("Marseille", ["Marseilles"]) is True


def test_entity_is_covered_false_for_meaningfully_different_entity():
    assert _entity_is_covered("France", ["French"]) is False


def test_entity_is_covered_false_for_boundary_substring_case():
    assert _entity_is_covered("Mali", ["Malibu"]) is False


def test_entity_is_covered_false_when_no_candidates():
    assert _entity_is_covered("Paris", []) is False


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_empty_text_returns_empty_without_touching_model():
    assert extract_entities("", nlp=ExplodingNLP()) == []
    assert extract_entities("   ", nlp=ExplodingNLP()) == []


def test_extract_entities_returns_entity_surface_text_in_order():
    nlp = FakeNLP({"Some text about Paris and Beta.": ["Paris", "Beta"]})
    result = extract_entities("Some text about Paris and Beta.", nlp=nlp)
    assert result == ["Paris", "Beta"]
    assert nlp.calls == ["Some text about Paris and Beta."]


def test_extract_entities_no_entities_detected_returns_empty_list():
    nlp = FakeNLP({"How does photosynthesis work?": []})
    assert extract_entities("How does photosynthesis work?", nlp=nlp) == []


# ---------------------------------------------------------------------------
# compute_entity_coverage
# ---------------------------------------------------------------------------


def test_compute_entity_coverage_zero_question_entities_returns_sentinel():
    assert compute_entity_coverage([], ["Paris", "France"]) == NO_QUESTION_ENTITIES_SENTINEL


def test_compute_entity_coverage_zero_question_entities_returns_sentinel_even_with_no_chunk_entities():
    assert compute_entity_coverage([], []) == NO_QUESTION_ENTITIES_SENTINEL


def test_compute_entity_coverage_zero_chunk_entities_is_well_defined_zero():
    # Question has entities to check; chunks just don't contain any --
    # this is a real 0.0, not the no-question-entities sentinel.
    result = compute_entity_coverage(["Paris"], [])
    assert result == 0.0
    assert result != NO_QUESTION_ENTITIES_SENTINEL


def test_compute_entity_coverage_hand_worked_half_covered():
    # "Paris" exact-matches; "Director Beta" matches nothing in the
    # candidate list (too different from both "Paris" and "Some Other
    # Entity" under both exact-normalized and fuzzy matching).
    question_entities = ["Paris", "Director Beta"]
    chunk_entities = ["Paris", "Some Other Entity"]
    assert compute_entity_coverage(question_entities, chunk_entities) == pytest.approx(0.5)


def test_compute_entity_coverage_hand_worked_full_coverage_via_fuzzy_and_exact():
    # "Marseille" covered fuzzily by "Marseilles"; "U.S." covered
    # exactly (after normalization) by "US".
    question_entities = ["Marseille", "U.S."]
    chunk_entities = ["Marseilles", "US"]
    assert compute_entity_coverage(question_entities, chunk_entities) == pytest.approx(1.0)


def test_compute_entity_coverage_hand_worked_zero_coverage():
    question_entities = ["France", "Mali"]
    chunk_entities = ["French", "Malibu"]  # both near-misses, neither counts
    assert compute_entity_coverage(question_entities, chunk_entities) == pytest.approx(0.0)


def test_compute_entity_coverage_duplicate_question_entities_counted_per_occurrence():
    # "Paris" appears twice in the question entity list (e.g. mentioned
    # twice); each occurrence is checked independently, not deduplicated.
    question_entities = ["Paris", "Paris", "Atlantis"]
    chunk_entities = ["Paris"]
    assert compute_entity_coverage(question_entities, chunk_entities) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# extract_entity_coverage_features
# ---------------------------------------------------------------------------


def test_extract_entity_coverage_features_empty_chunks_never_touches_model():
    result = extract_entity_coverage_features("Any question?", [], nlp=ExplodingNLP())
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_extract_entity_coverage_features_empty_chunks_no_nlp_override_still_short_circuits():
    # No `nlp=` override at all -- if this ever tried to reach
    # `_get_nlp()` it would attempt to load the real spaCy model.
    result = extract_entity_coverage_features("Any question?", [])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_extract_entity_coverage_features_zero_question_entities_returns_sentinel():
    nlp = FakeNLP(
        {
            "How does photosynthesis work?": [],
            "Plants use sunlight to make energy. Chlorophyll absorbs light.": ["Chlorophyll"],
        }
    )
    result = extract_entity_coverage_features(
        "How does photosynthesis work?",
        ["Plants use sunlight to make energy.", "Chlorophyll absorbs light."],
        nlp=nlp,
    )
    assert result == {"entity_coverage_fraction": NO_QUESTION_ENTITIES_SENTINEL}


def test_extract_entity_coverage_features_end_to_end_with_fake_nlp():
    question = "What nationality is the director of the film that won the award?"
    chunks = [
        "Film Alpha is a 2004 movie that won the Golden Prize.",
        "Director Beta directed Film Alpha.",
        "Beta is a French filmmaker born in Marseille.",
    ]
    joined_chunks = " ".join(chunks)
    nlp = FakeNLP(
        {
            question: ["Director Beta"],
            joined_chunks: ["Film Alpha", "Director Beta", "Beta", "French", "Marseille"],
        }
    )
    result = extract_entity_coverage_features(question, chunks, nlp=nlp)
    assert set(result.keys()) == set(FEATURE_NAMES)
    # "Director Beta" exact-matches "Director Beta" in the chunk entities.
    assert result["entity_coverage_fraction"] == pytest.approx(1.0)


def test_extract_entity_coverage_features_concatenates_chunks_for_single_nlp_call():
    nlp = FakeNLP(
        {
            "Q?": ["Paris"],
            "chunk one chunk two": ["Paris"],
        }
    )
    extract_entity_coverage_features("Q?", ["chunk one", "chunk two"], nlp=nlp)
    assert nlp.calls == ["Q?", "chunk one chunk two"]


# ---------------------------------------------------------------------------
# Real-model integration test (slow, requires the en_core_web_sm download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_spacy_model_entity_coverage_and_no_entity_sentinel():
    """Loads the real en_core_web_sm pipeline. Requires
    `python -m spacy download en_core_web_sm` to have been run once."""
    question = "Where is the Eiffel Tower located?"
    chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Bananas are a good source of potassium and dietary fiber.",
    ]

    question_entities = extract_entities(question)
    # spaCy's entity span boundaries aren't guaranteed to exclude leading
    # determiners -- en_core_web_sm has been observed tagging this as
    # "the Eiffel Tower" rather than "Eiffel Tower". Check containment,
    # not an exact span match, so this test asserts what entity_coverage.py
    # actually relies on (detection) without hardcoding a boundary the
    # real model doesn't promise.
    assert any("Eiffel Tower" in e for e in question_entities)

    features = extract_entity_coverage_features(question, chunks)
    assert set(features.keys()) == set(FEATURE_NAMES)
    # "Eiffel Tower" appears verbatim in the first chunk, so it must be
    # covered -- coverage should be a real, positive value, not the
    # zero-question-entities sentinel.
    assert features["entity_coverage_fraction"] > 0.0
    assert features["entity_coverage_fraction"] <= 1.0

    # A question with no proper-noun-style entities at all should hit
    # the documented sentinel, not a divide-by-zero or a bogus 0.0.
    no_entity_question = "How does photosynthesis work?"
    assert extract_entities(no_entity_question) == []
    no_entity_features = extract_entity_coverage_features(no_entity_question, chunks)
    assert no_entity_features == {"entity_coverage_fraction": NO_QUESTION_ENTITIES_SENTINEL}


@pytest.mark.slow
def test_real_spacy_model_article_only_mismatch_still_fuzzy_matches():
    """General case behind the "the Eiffel Tower" vs "Eiffel Tower"
    boundary issue: an entity span that only differs from another by a
    leading article should still be covered via fuzzy matching, without
    entity_coverage.py needing any article-stripping logic of its own.

    difflib.SequenceMatcher(None, "the eiffel tower", "eiffel tower").ratio()
    is ~0.857 (computed and confirmed before writing this test), which
    clears FUZZY_MATCH_THRESHOLD (0.8) with room to spare -- so this is
    checking real, already-correct behavior, not proposing a threshold
    change.
    """
    ratio = SequenceMatcher(None, "the eiffel tower", "eiffel tower").ratio()
    assert ratio > FUZZY_MATCH_THRESHOLD

    question = "Where is the Eiffel Tower located?"
    chunks = ["Eiffel Tower is a wrought-iron lattice tower in Paris, France."]

    question_entities = extract_entities(question)
    assert any("Eiffel Tower" in e for e in question_entities)
    # The entity as spaCy actually tagged it (e.g. "the Eiffel Tower").
    question_entity = next(e for e in question_entities if "Eiffel Tower" in e)

    chunk_entities = extract_entities(chunks[0])
    assert any("Eiffel Tower" in e for e in chunk_entities)

    assert _entity_is_covered(question_entity, chunk_entities) is True

    features = extract_entity_coverage_features(question, chunks)
    assert features["entity_coverage_fraction"] > 0.0
