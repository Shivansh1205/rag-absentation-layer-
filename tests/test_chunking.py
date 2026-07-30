"""Tests for `abstention_data.chunking`, with a focus on the
word-boundary-safety of `contains_answer_span` / `remove_answer_bearing_chunks`.

These are the tests to eyeball closely before `corruption.py` is built on
top of this module: everything downstream assumes "answer not present"
means the answer is truly absent, not just absent-as-a-raw-substring.
"""

from __future__ import annotations

from abstention_data.chunking import (
    chunk_passage,
    chunk_sentences,
    contains_answer_span,
    remove_answer_bearing_chunks,
    split_into_sentences,
)


# ---------------------------------------------------------------------------
# split_into_sentences / chunk_sentences / chunk_passage
# ---------------------------------------------------------------------------


def test_split_into_sentences_basic():
    passage = "Film Alpha is a 2004 movie. It was shot mostly in Lyon. It won a prize."
    sentences = split_into_sentences(passage)
    assert sentences == [
        "Film Alpha is a 2004 movie.",
        "It was shot mostly in Lyon.",
        "It won a prize.",
    ]


def test_split_into_sentences_empty_string():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []


def test_chunk_sentences_groups_by_window_size():
    sentences = ["S1.", "S2.", "S3.", "S4.", "S5."]
    chunks = chunk_sentences(sentences, max_sentences_per_chunk=2)
    assert chunks == ["S1. S2.", "S3. S4.", "S5."]


def test_chunk_sentences_rejects_invalid_window_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_sentences(["S1."], max_sentences_per_chunk=0)


def test_chunk_passage_end_to_end():
    passage = "Sentence one is here. Sentence two follows. Sentence three ends it."
    chunks = chunk_passage(passage, max_sentences_per_chunk=2)
    assert chunks == [
        "Sentence one is here. Sentence two follows.",
        "Sentence three ends it.",
    ]


# ---------------------------------------------------------------------------
# contains_answer_span -- the adversarial cases
# ---------------------------------------------------------------------------


def test_contains_answer_span_true_for_genuine_standalone_mention():
    text = "Gamma was born in Mali in 1978."
    assert contains_answer_span(text, "Mali") is True


def test_contains_answer_span_false_when_answer_is_substring_of_another_word():
    # "Mali" is a substring of "Malibu" -- must NOT count as a match.
    text = "Malibu Beach is a well known stretch of coastline in California."
    assert contains_answer_span(text, "Mali") is False


def test_contains_answer_span_false_for_lowercase_substring_inside_longer_word():
    # "mali" (case-insensitively) is a substring of "formalize" -- must
    # NOT count as a match even with case-insensitive comparison.
    text = "Lawyers sometimes formalize an agreement before signing."
    assert contains_answer_span(text, "Mali") is False


def test_contains_answer_span_true_for_separate_genuine_mention_elsewhere():
    # A second, unrelated-looking sentence that genuinely re-mentions the
    # answer as a standalone word must still be flagged -- this is real
    # signal leakage, not a false alarm to suppress.
    text = "This sentence is about something else. But Mali is mentioned again here."
    assert contains_answer_span(text, "Mali") is True


def test_contains_answer_span_is_case_insensitive_for_genuine_matches():
    assert contains_answer_span("She identifies as FRENCH.", "French") is True
    assert contains_answer_span("she identifies as french.", "French") is True


def test_contains_answer_span_handles_multi_word_answer():
    text = "The film was part of the French New Wave movement."
    assert contains_answer_span(text, "French New Wave") is True
    # Partial/reordered overlap should not match.
    assert contains_answer_span("The wave was new and French.", "French New Wave") is False


def test_contains_answer_span_handles_trailing_punctuation_boundary():
    # Answer immediately followed by a period/comma should still match --
    # punctuation is not a word character, so (?!\w) is satisfied.
    assert contains_answer_span("Her nationality is French.", "French") is True
    assert contains_answer_span("Nationality: French, confirmed.", "French") is True


def test_contains_answer_span_false_for_empty_answer():
    assert contains_answer_span("Any text at all.", "") is False
    assert contains_answer_span("Any text at all.", "   ") is False


# ---------------------------------------------------------------------------
# remove_answer_bearing_chunks
# ---------------------------------------------------------------------------


def test_remove_answer_bearing_chunks_drops_only_genuine_mentions():
    chunks = [
        "Musician Gamma is known for blending traditional and modern styles.",
        "Gamma was born in Mali in 1978.",
        "Malibu Beach is a well known stretch of coastline in California.",
        "Lawyers sometimes formalize an agreement before signing.",
    ]
    remaining, did_remove_any = remove_answer_bearing_chunks(chunks, "Mali")

    assert did_remove_any is True
    # Only the genuine-mention chunk is dropped.
    assert "Gamma was born in Mali in 1978." not in remaining
    # The two adversarial substring chunks must survive untouched.
    assert "Malibu Beach is a well known stretch of coastline in California." in remaining
    assert "Lawyers sometimes formalize an agreement before signing." in remaining
    assert "Musician Gamma is known for blending traditional and modern styles." in remaining
    assert len(remaining) == 3


def test_remove_answer_bearing_chunks_reports_false_when_nothing_removed():
    chunks = [
        "Malibu Beach is a well known stretch of coastline in California.",
        "Lawyers sometimes formalize an agreement before signing.",
    ]
    remaining, did_remove_any = remove_answer_bearing_chunks(chunks, "Mali")
    assert did_remove_any is False
    assert remaining == chunks


def test_remove_answer_bearing_chunks_never_leaves_answer_span_behind():
    # Property-style check across several adversarial chunk sets: after
    # removal, no remaining chunk should satisfy contains_answer_span.
    cases = [
        (
            ["Gamma was born in Mali in 1978.", "Malibu Beach is sunny."],
            "Mali",
        ),
        (
            ["The director is French.", "This is a French New Wave film."],
            "French",
        ),
        (
            ["Nothing relevant here.", "Also nothing relevant."],
            "Unrelated Answer",
        ),
    ]
    for chunks, answer in cases:
        remaining, _ = remove_answer_bearing_chunks(chunks, answer)
        assert all(not contains_answer_span(c, answer) for c in remaining)
