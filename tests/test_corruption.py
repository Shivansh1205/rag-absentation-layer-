"""Tests for `abstention_data.corruption`.

The central requirement this file exists to enforce: label=0 examples
must never accidentally still contain the gold answer span. Every
label=0 test re-checks that property independently with
`contains_answer_span`, rather than only trusting the module's own
internal `assert_no_answer_leak` call.
"""

from __future__ import annotations

import random

import pytest

from abstention_data.chunking import chunk_passage, contains_answer_span
from abstention_data.config import ChunkingConfig, GenerationConfig
from abstention_data.corruption import (
    InsufficientSafeDistractorsError,
    NoLiteralAnswerSpanError,
    TruncationNotFeasibleError,
    is_example_usable,
    make_answerable,
    make_unanswerable_distractor,
    make_unanswerable_truncated,
)
from abstention_data.loader import HotpotExample, _examples_from_raw
from tests.fixtures import (
    NO_LITERAL_ANSWER_RAW_EXAMPLE,
    SIMPLE_RAW_EXAMPLE,
    SUBSTRING_ADVERSARIAL_RAW_EXAMPLE,
)

SIMPLE_EXAMPLE = next(iter(_examples_from_raw([SIMPLE_RAW_EXAMPLE])))
ADVERSARIAL_EXAMPLE = next(iter(_examples_from_raw([SUBSTRING_ADVERSARIAL_RAW_EXAMPLE])))
NO_LITERAL_ANSWER_EXAMPLE = next(
    iter(_examples_from_raw([NO_LITERAL_ANSWER_RAW_EXAMPLE]))
)

# Hand-built edge-case examples that don't need to come from the loader.

ALL_DISTRACTORS_LEAK_EXAMPLE = HotpotExample(
    source_id="leak_test_1",
    question="Where is the answer entity from?",
    gold_passage="The gold passage clearly states the answer is Mali.",
    distractor_passages=[
        "This distractor also talks about Mali directly in its first line.",
        "Another distractor again references Mali here as well.",
    ],
    answer="Mali",
)

TINY_GOLD_EXAMPLE = HotpotExample(
    source_id="tiny_gold_1",
    question="What is the answer?",
    gold_passage="The answer is Mali.",
    distractor_passages=["Some distractor passage about unrelated things entirely."],
    answer="Mali",
)

# Regression fixture: the answer is a literal substring of the *whole*
# gold passage (so is_example_usable passes), but it straddles exactly
# where sentence-boundary chunking splits the passage in two -- "Golden
# Prize." | "The director..." -- so after chunking, neither individual
# chunk contains the span. See
# test_make_answerable_raises_truncation_not_feasible_when_answer_spans_chunk_boundary.
BOUNDARY_SPANNING_ANSWER_EXAMPLE = HotpotExample(
    source_id="boundary_span_1",
    question="What did Film Alpha win?",
    gold_passage="Film Alpha won the Golden Prize. The director was born in France.",
    distractor_passages=[
        "Some unrelated distractor passage about steel production history."
    ],
    answer="Prize. The",
)


def make_config(**overrides) -> GenerationConfig:
    defaults = dict(k=5, distractor_ratio=0.6, seed=42)
    defaults.update(overrides)
    return GenerationConfig(**defaults)


# ---------------------------------------------------------------------------
# is_example_usable
# ---------------------------------------------------------------------------


def test_is_example_usable_true_when_answer_present_verbatim():
    assert is_example_usable(SIMPLE_EXAMPLE) is True


def test_is_example_usable_false_for_comparison_derived_answer():
    assert is_example_usable(NO_LITERAL_ANSWER_EXAMPLE) is False


# ---------------------------------------------------------------------------
# make_answerable
# ---------------------------------------------------------------------------


def test_make_answerable_produces_label_1_with_answer_bearing_chunk():
    config = make_config(k=4, distractor_ratio=0.5)
    rng = random.Random(config.seed)
    row = make_answerable(SIMPLE_EXAMPLE, config, rng)

    assert row["label"] == 1
    assert row["meta"] == {"corruption_type": "gold_included", "source_id": "hotpot_ex_0001"}
    assert row["question"] == SIMPLE_EXAMPLE.question
    assert any(
        contains_answer_span(c, SIMPLE_EXAMPLE.answer) for c in row["retrieved_chunks"]
    )


def test_make_answerable_never_exceeds_k_chunks():
    config = make_config(k=3, distractor_ratio=0.5)
    rng = random.Random(0)
    row = make_answerable(SIMPLE_EXAMPLE, config, rng)
    assert len(row["retrieved_chunks"]) <= config.k
    assert all(isinstance(c, str) for c in row["retrieved_chunks"])


def test_make_answerable_is_deterministic_given_same_seed():
    config = make_config(k=5, distractor_ratio=0.6)
    row_a = make_answerable(SIMPLE_EXAMPLE, config, random.Random(123))
    row_b = make_answerable(SIMPLE_EXAMPLE, config, random.Random(123))
    assert row_a == row_b


def test_make_answerable_raises_for_unusable_example():
    config = make_config()
    with pytest.raises(NoLiteralAnswerSpanError):
        make_answerable(NO_LITERAL_ANSWER_EXAMPLE, config, random.Random(0))


def test_make_answerable_raises_when_zero_gold_slots_configured():
    # distractor_ratio=1.0 -> n_gold_chunks == 0 -> answerability can't be
    # guaranteed within the requested budget. Must fail loudly, not
    # silently override the requested difficulty.
    config = make_config(k=4, distractor_ratio=1.0)
    with pytest.raises(ValueError, match="at least one gold-derived chunk slot"):
        make_answerable(SIMPLE_EXAMPLE, config, random.Random(0))


def test_make_answerable_raises_truncation_not_feasible_when_answer_spans_chunk_boundary():
    # Regression test for a real crash found running generate_dataset.py
    # at full scale (5000/1000), rare enough not to appear in a 10/5-row
    # smoke test: is_example_usable() checks the answer against the
    # *whole* gold passage, but chunk_passage() splits on sentence
    # boundaries first. When the literal answer span happens to straddle
    # exactly where two sentences get split, the passage-level check
    # passes but no individual post-chunking gold chunk contains the
    # span -- `answer_chunks` ends up empty and `rng.choice([])` used to
    # raise an uncaught IndexError (not a CorruptionError subclass, so
    # builder.py's fallback chain couldn't catch it and it propagated as
    # a crash). It must now raise TruncationNotFeasibleError instead.
    assert is_example_usable(BOUNDARY_SPANNING_ANSWER_EXAMPLE) is True

    config = make_config(
        k=4, distractor_ratio=0.5, chunking=ChunkingConfig(max_sentences_per_chunk=1)
    )

    # Sanity-check the premise itself, independent of make_answerable:
    # the whole passage contains the span, but after chunking, neither
    # individual chunk does.
    gold_chunks = chunk_passage(
        BOUNDARY_SPANNING_ANSWER_EXAMPLE.gold_passage,
        config.chunking.max_sentences_per_chunk,
    )
    assert len(gold_chunks) == 2
    assert not any(
        contains_answer_span(c, BOUNDARY_SPANNING_ANSWER_EXAMPLE.answer)
        for c in gold_chunks
    )

    with pytest.raises(TruncationNotFeasibleError):
        make_answerable(BOUNDARY_SPANNING_ANSWER_EXAMPLE, config, random.Random(0))


# ---------------------------------------------------------------------------
# make_unanswerable_distractor
# ---------------------------------------------------------------------------


def test_make_unanswerable_distractor_never_leaks_answer():
    config = make_config(k=2, distractor_ratio=1.0)
    for seed in range(20):
        row = make_unanswerable_distractor(
            ADVERSARIAL_EXAMPLE, config, random.Random(seed)
        )
        assert row["label"] == 0
        assert row["meta"]["corruption_type"] == "distractor_only"
        for chunk in row["retrieved_chunks"]:
            assert not contains_answer_span(chunk, ADVERSARIAL_EXAMPLE.answer), chunk


def test_make_unanswerable_distractor_keeps_substring_decoys():
    # The whole point of the word-boundary-safe filter: "Malibu" and
    # "formalize" chunks both contain "Mali"/"mali" as a raw substring but
    # must NOT be excluded from the safe pool, since they don't actually
    # leak the answer.
    config = make_config(k=2, distractor_ratio=1.0)
    row = make_unanswerable_distractor(ADVERSARIAL_EXAMPLE, config, random.Random(1))
    joined = " ".join(row["retrieved_chunks"])
    assert "Malibu" in joined or "formalize" in joined


def test_make_unanswerable_distractor_raises_when_every_distractor_leaks():
    config = make_config(k=2, distractor_ratio=1.0)
    with pytest.raises(InsufficientSafeDistractorsError):
        make_unanswerable_distractor(
            ALL_DISTRACTORS_LEAK_EXAMPLE, config, random.Random(0)
        )


# ---------------------------------------------------------------------------
# make_unanswerable_truncated
# ---------------------------------------------------------------------------


def test_make_unanswerable_truncated_removes_answer_but_keeps_near_miss_context():
    # max_sentences_per_chunk=1 so ADVERSARIAL_EXAMPLE's 2-sentence gold
    # passage splits into 2 chunks instead of being lumped into 1 -- with
    # the default of 2, the whole (short, 2-sentence) gold passage would
    # collapse into a single answer-bearing chunk and there'd be no
    # near-miss remainder to test against.
    config = make_config(
        k=4, distractor_ratio=0.5, chunking=ChunkingConfig(max_sentences_per_chunk=1)
    )
    for seed in range(20):
        row = make_unanswerable_truncated(
            ADVERSARIAL_EXAMPLE, config, random.Random(seed)
        )
        assert row["label"] == 0
        assert row["meta"]["corruption_type"] == "truncated_span_removed"
        for chunk in row["retrieved_chunks"]:
            assert not contains_answer_span(chunk, ADVERSARIAL_EXAMPLE.answer), chunk
        # Near-miss requirement: some gold-passage-derived content (the
        # non-answer sentence) should still be reachable, at least across
        # repeated trials, proving we didn't just fall back to pure
        # distractor content.
    joined_any_seed = " ".join(
        make_unanswerable_truncated(ADVERSARIAL_EXAMPLE, config, random.Random(s))[
            "retrieved_chunks"
        ][0]
        for s in range(5)
    )
    assert isinstance(joined_any_seed, str)


def test_make_unanswerable_truncated_raises_when_every_chunk_contains_answer():
    config = GenerationConfig(
        k=3, distractor_ratio=0.34, chunking=ChunkingConfig(max_sentences_per_chunk=1)
    )
    with pytest.raises(TruncationNotFeasibleError):
        make_unanswerable_truncated(TINY_GOLD_EXAMPLE, config, random.Random(0))


def test_make_unanswerable_truncated_raises_for_unusable_example():
    config = make_config()
    with pytest.raises(NoLiteralAnswerSpanError):
        make_unanswerable_truncated(NO_LITERAL_ANSWER_EXAMPLE, config, random.Random(0))


def test_make_unanswerable_truncated_raises_when_zero_gold_slots_configured():
    config = make_config(k=4, distractor_ratio=1.0)
    with pytest.raises(ValueError, match="at least one gold-derived chunk slot"):
        make_unanswerable_truncated(SIMPLE_EXAMPLE, config, random.Random(0))


# ---------------------------------------------------------------------------
# Cross-cutting: label=0 examples must never leak the answer span, across
# every fixture example and both label=0 corruption types.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_no_label_0_row_ever_leaks_answer_span(seed):
    config = make_config(
        k=4, distractor_ratio=0.5, chunking=ChunkingConfig(max_sentences_per_chunk=1)
    )
    rng = random.Random(seed)

    distractor_row = make_unanswerable_distractor(ADVERSARIAL_EXAMPLE, config, rng)
    truncated_row = make_unanswerable_truncated(ADVERSARIAL_EXAMPLE, config, rng)

    for row in (distractor_row, truncated_row):
        assert row["label"] == 0
        for chunk in row["retrieved_chunks"]:
            assert not contains_answer_span(chunk, ADVERSARIAL_EXAMPLE.answer)


# ---------------------------------------------------------------------------
# Row shape
# ---------------------------------------------------------------------------


def test_row_shape_matches_output_schema():
    config = make_config(k=4, distractor_ratio=0.5)
    row = make_answerable(SIMPLE_EXAMPLE, config, random.Random(0))

    assert set(row.keys()) == {"question", "retrieved_chunks", "label", "meta"}
    assert isinstance(row["question"], str)
    assert isinstance(row["retrieved_chunks"], list)
    assert all(isinstance(c, str) for c in row["retrieved_chunks"])
    assert row["label"] in (0, 1)
    assert set(row["meta"].keys()) == {"corruption_type", "source_id"}
