"""Tests for `abstention_data.loader`.

These exercise `_examples_from_raw` — the pure normalization function —
against hand-written fixtures shaped exactly like HF `datasets`'
hotpot_qa/distractor records. No network access, no `datasets` import
required at all. This is intentional: real-hub loading (`load_hotpotqa`)
is a thin, mostly-untestable wrapper around this function, and keeping the
network out of the test suite keeps CI from being flaky.
"""

from __future__ import annotations

from abstention_data.loader import HotpotExample, _examples_from_raw
from tests.fixtures import (
    MALFORMED_RAW_EXAMPLE,
    NO_LITERAL_ANSWER_RAW_EXAMPLE,
    SIMPLE_RAW_EXAMPLE,
    SUBSTRING_ADVERSARIAL_RAW_EXAMPLE,
)


def test_simple_example_normalizes_correctly():
    [example] = list(_examples_from_raw([SIMPLE_RAW_EXAMPLE]))

    assert isinstance(example, HotpotExample)
    assert example.source_id == "hotpot_ex_0001"
    assert example.question == SIMPLE_RAW_EXAMPLE["question"]
    assert example.answer == "French"

    # Gold passage is the concatenation of both gold titles' sentences, in
    # context order.
    assert "Film Alpha is a 2004 movie" in example.gold_passage
    assert "Director Beta directed Film Alpha" in example.gold_passage
    assert "Beta is a French filmmaker" in example.gold_passage

    # Exactly the 3 non-gold titles become distractor passages.
    assert len(example.distractor_passages) == 3
    joined_distractors = " ".join(example.distractor_passages)
    assert "steel production" in joined_distractors
    assert "freshwater fish" in joined_distractors
    assert "rail infrastructure" in joined_distractors

    # Gold content must not leak into distractors, and vice versa.
    assert "Golden Prize" not in joined_distractors
    assert "steel production" not in example.gold_passage


def test_multiple_examples_yield_in_order():
    examples = list(
        _examples_from_raw([SIMPLE_RAW_EXAMPLE, SUBSTRING_ADVERSARIAL_RAW_EXAMPLE])
    )
    assert [e.source_id for e in examples] == ["hotpot_ex_0001", "hotpot_ex_0002"]


def test_malformed_example_with_dangling_supporting_fact_is_skipped():
    # supporting_facts references a title absent from context -> no gold
    # paragraphs can be resolved -> the loader should skip it silently
    # rather than emit an example with an empty gold_passage.
    examples = list(_examples_from_raw([MALFORMED_RAW_EXAMPLE]))
    assert examples == []


def test_skipped_malformed_example_does_not_break_surrounding_valid_ones():
    examples = list(
        _examples_from_raw(
            [SIMPLE_RAW_EXAMPLE, MALFORMED_RAW_EXAMPLE, SUBSTRING_ADVERSARIAL_RAW_EXAMPLE]
        )
    )
    assert [e.source_id for e in examples] == ["hotpot_ex_0001", "hotpot_ex_0002"]


def test_example_with_no_literal_answer_still_loads():
    # The loader itself doesn't filter on "answer appears literally in
    # gold_passage" -- that's a downstream (builder/corruption) concern.
    # It should faithfully pass through examples like HotpotQA's
    # comparison-type yes/no questions.
    [example] = list(_examples_from_raw([NO_LITERAL_ANSWER_RAW_EXAMPLE]))
    assert example.answer == "yes"
    assert "yes" not in example.gold_passage.lower().split()
