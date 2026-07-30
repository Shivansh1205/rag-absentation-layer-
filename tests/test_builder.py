"""Tests for `abstention_data.builder`, focused on the fallback policy:
unusable examples are skipped, infeasible sampled corruption types fall
back to a working one (and always report the type that actually
succeeded, never the one that was sampled but failed), and a structurally
broken config is rejected immediately rather than per-example.
"""

from __future__ import annotations

import dataclasses

import pytest

from abstention_data.builder import build_dataset
from abstention_data.config import ChunkingConfig, GenerationConfig
from abstention_data.loader import _examples_from_raw
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


def _replicate(example, n: int):
    """Make `n` distinct-source_id copies of `example` for volume tests."""
    return [
        dataclasses.replace(example, source_id=f"{example.source_id}_{i}")
        for i in range(n)
    ]


def make_config(**overrides) -> GenerationConfig:
    defaults = dict(k=4, distractor_ratio=0.5, seed=7)
    defaults.update(overrides)
    return GenerationConfig(**defaults)


def test_build_dataset_produces_requested_rows_when_pool_sufficient():
    examples = _replicate(SIMPLE_EXAMPLE, 15)
    config = make_config()
    rows, stats = build_dataset(examples, config, n_rows=10)

    assert len(rows) == 10
    assert stats.n_produced == 10
    assert stats.n_requested == 10
    assert stats.n_input_examples == 15
    assert sum(stats.corruption_type_counts.values()) == 10


def test_build_dataset_stops_at_n_rows_even_with_more_examples_available():
    examples = _replicate(SIMPLE_EXAMPLE, 50)
    rows, stats = build_dataset(examples, make_config(), n_rows=5)
    assert len(rows) == 5
    assert stats.n_produced == 5


def test_build_dataset_skips_unusable_examples():
    usable = _replicate(SIMPLE_EXAMPLE, 5)
    unusable = _replicate(NO_LITERAL_ANSWER_EXAMPLE, 5)
    # Interleave so skips don't just happen to land at the end.
    examples = [x for pair in zip(usable, unusable) for x in pair]

    rows, stats = build_dataset(examples, make_config(), n_rows=5)

    assert len(rows) == 5
    # Iteration stops as soon as n_rows is hit, so with usable/unusable
    # interleaved as [u0, x0, u1, x1, ..., u4, x4], the 5th usable example
    # (index 8) satisfies the request before x4 (index 9) is ever looked
    # at -- so only 4 unusable examples get counted, not 5.
    assert stats.n_skipped_unusable == 4
    for row in rows:
        assert row["meta"]["source_id"].startswith(SIMPLE_EXAMPLE.source_id)


def test_build_dataset_rejects_structurally_broken_config_immediately():
    # distractor_ratio=1.0 -> n_gold_chunks == 0, but the default weights
    # give nonzero weight to gold_included and truncated_span_removed.
    config = make_config(distractor_ratio=1.0)
    with pytest.raises(ValueError, match="0 gold-derived chunk slots"):
        build_dataset([], config, n_rows=1)


def test_build_dataset_allows_distractor_ratio_1_if_weights_only_use_distractor_only():
    config = make_config(
        distractor_ratio=1.0,
        corruption_type_weights={
            "gold_included": 0.0,
            "distractor_only": 1.0,
            "truncated_span_removed": 0.0,
        },
    )
    examples = _replicate(ADVERSARIAL_EXAMPLE, 5)
    rows, stats = build_dataset(examples, config, n_rows=3)
    assert len(rows) == 3
    assert all(r["meta"]["corruption_type"] == "distractor_only" for r in rows)


def test_build_dataset_falls_back_when_sampled_type_infeasible():
    # Force every example to sample "truncated_span_removed" (weight 1.0),
    # but with the default chunk size (2 sentences/chunk) ADVERSARIAL_EXAMPLE's
    # 2-sentence gold passage collapses into a single answer-bearing
    # chunk, making truncation infeasible -> must fall back to
    # distractor_only (next in _FALLBACK_ORDER) rather than being skipped.
    config = make_config(
        distractor_ratio=0.5,
        corruption_type_weights={
            "gold_included": 0.0,
            "distractor_only": 0.0,
            "truncated_span_removed": 1.0,
        },
    )
    examples = _replicate(ADVERSARIAL_EXAMPLE, 5)
    rows, stats = build_dataset(examples, config, n_rows=5)

    assert len(rows) == 5
    assert stats.n_fallback == 5
    assert stats.n_skipped_all_failed == 0
    # The type that actually succeeded must be reported truthfully, never
    # the sampled-but-failed "truncated_span_removed".
    assert all(r["meta"]["corruption_type"] == "distractor_only" for r in rows)


def test_build_dataset_with_smaller_chunk_size_makes_truncated_feasible():
    # Same scenario as above, but max_sentences_per_chunk=1 means
    # ADVERSARIAL_EXAMPLE's gold passage splits into 2 chunks, so
    # truncation *is* feasible and no fallback should be needed.
    config = make_config(
        distractor_ratio=0.5,
        chunking=ChunkingConfig(max_sentences_per_chunk=1),
        corruption_type_weights={
            "gold_included": 0.0,
            "distractor_only": 0.0,
            "truncated_span_removed": 1.0,
        },
    )
    examples = _replicate(ADVERSARIAL_EXAMPLE, 5)
    rows, stats = build_dataset(examples, config, n_rows=5)

    assert len(rows) == 5
    assert stats.n_fallback == 0
    assert all(r["meta"]["corruption_type"] == "truncated_span_removed" for r in rows)


def test_build_dataset_is_deterministic():
    examples = _replicate(SIMPLE_EXAMPLE, 20)
    config = make_config()
    rows_a, _ = build_dataset(examples, config, n_rows=12)
    rows_b, _ = build_dataset(examples, config, n_rows=12)
    assert rows_a == rows_b


def test_build_dataset_returns_fewer_rows_when_pool_exhausted():
    examples = _replicate(SIMPLE_EXAMPLE, 3)
    rows, stats = build_dataset(examples, make_config(), n_rows=10)
    assert len(rows) == 3
    assert stats.n_produced == 3
