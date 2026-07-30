"""Orchestrates loader examples + corruption functions into a labeled
dataset of (question, retrieved_chunks, label, meta) rows.

Fallback policy
----------------
`corruption.py` deliberately raises rather than silently degrading when a
specific corruption type isn't feasible for a specific example (see its
module docstring). This module is where that's turned into a concrete,
visible policy:

1. An example whose answer never appears verbatim in the gold passage
   (`is_example_usable` is False) is skipped entirely, before any
   corruption type is attempted, and counted in `BuildStats.n_skipped_unusable`.
2. A config where a requested corruption type structurally can't work
   (``n_gold_chunks == 0`` while ``gold_included`` or
   ``truncated_span_removed`` have nonzero weight) is rejected immediately
   at `build_dataset` entry, rather than discovered example-by-example.
3. For each usable example, a corruption type is sampled per
   `config.corruption_type_weights` using a seed derived from
   (``config.seed``, ``example.source_id``) — reproducible regardless of
   iteration order or how many other examples are processed.
4. If the sampled corruption type fails for that specific example (e.g.
   `TruncationNotFeasibleError`, `InsufficientSafeDistractorsError`), we
   fall back through `_FALLBACK_ORDER` for that type rather than dropping
   the example outright -- this keeps dataset size targets achievable
   without silently producing degenerate rows. Every fallback is counted
   in `BuildStats.n_fallback` and which corruption type a row actually
   ended up as is always recorded truthfully in `meta.corruption_type`
   (never the originally-sampled type if a fallback occurred).
5. If every corruption type in the fallback chain fails for an example
   (should be rare -- `gold_included` is the last resort and only fails
   for reasons already ruled out by steps 1-2), the example is skipped
   and counted in `BuildStats.n_skipped_all_failed`.

Seeding note: per-example RNGs are seeded from an f-string
(``f"{seed}:{source_id}:..."``), not a tuple. `random.Random(a)` hashes
non-str/bytes/int seeds with the process's `hash()`, which is randomized
per-run for strings inside tuples (PYTHONHASHSEED) -- that would silently
break cross-run reproducibility. Seeding with a plain string avoids this,
since `random` hashes str seeds through its own deterministic algorithm.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from abstention_data.config import GenerationConfig
from abstention_data.corruption import (
    CorruptionError,
    is_example_usable,
    make_answerable,
    make_unanswerable_distractor,
    make_unanswerable_truncated,
)
from abstention_data.loader import HotpotExample

_CORRUPTION_FUNCS = {
    "gold_included": make_answerable,
    "distractor_only": make_unanswerable_distractor,
    "truncated_span_removed": make_unanswerable_truncated,
}

# Deterministic fallback chain per originally-sampled corruption type.
# `gold_included` has no fallback: once an example passes
# `is_example_usable` and the config has been validated to have
# `n_gold_chunks >= 1`, it should always succeed.
_FALLBACK_ORDER: dict[str, list[str]] = {
    "gold_included": [],
    "distractor_only": ["truncated_span_removed", "gold_included"],
    "truncated_span_removed": ["distractor_only", "gold_included"],
}


@dataclass
class BuildStats:
    """Bookkeeping for one `build_dataset` call, useful for logging and
    for sanity-checking a generation run in the CLI script."""

    n_input_examples: int = 0
    n_requested: int = 0
    n_produced: int = 0
    n_skipped_unusable: int = 0
    n_fallback: int = 0
    n_skipped_all_failed: int = 0
    corruption_type_counts: dict[str, int] = field(default_factory=dict)

    def record(self, corruption_type: str) -> None:
        self.corruption_type_counts[corruption_type] = (
            self.corruption_type_counts.get(corruption_type, 0) + 1
        )


def _validate_config_for_weights(config: GenerationConfig) -> None:
    """Fail fast, once, at the start of a build -- rather than discovering
    a structural config problem on the first example and then re-deriving
    the same failure on every subsequent one.
    """
    needs_gold_slot = (
        config.corruption_type_weights.get("gold_included", 0.0) > 0.0
        or config.corruption_type_weights.get("truncated_span_removed", 0.0) > 0.0
    )
    if needs_gold_slot and config.n_gold_chunks < 1:
        raise ValueError(
            "Config requests 'gold_included' and/or 'truncated_span_removed' "
            f"examples (nonzero weight) but distractor_ratio={config.distractor_ratio} "
            f"at k={config.k} allocates 0 gold-derived chunk slots. Lower "
            "distractor_ratio, raise k, or zero out those weights."
        )


def _sample_corruption_type(config: GenerationConfig, source_id: str) -> str:
    rng = random.Random(f"{config.seed}:{source_id}:corruption_type")
    types = list(config.corruption_type_weights.keys())
    weights = [config.corruption_type_weights[t] for t in types]
    return rng.choices(types, weights=weights, k=1)[0]


def build_dataset(
    examples: list[HotpotExample],
    config: GenerationConfig,
    n_rows: int,
) -> tuple[list[dict], BuildStats]:
    """Build up to `n_rows` labeled rows from `examples`.

    Consumes at most one row per input example (no reuse/duplication of a
    single HotpotQA example across rows), so `n_rows` should be <=
    `len(examples)` after accounting for expected skips; pass a larger
    example pool than `n_rows` to comfortably absorb `is_example_usable`
    skips and rare all-fallbacks-failed skips.

    Returns `(rows, stats)`. `rows` may have fewer than `n_rows` entries
    if `examples` is exhausted first -- check `stats` to see why.
    """
    _validate_config_for_weights(config)

    stats = BuildStats(n_input_examples=len(examples), n_requested=n_rows)
    rows: list[dict] = []

    for example in examples:
        if len(rows) >= n_rows:
            break

        if not is_example_usable(example):
            stats.n_skipped_unusable += 1
            continue

        sampled_type = _sample_corruption_type(config, example.source_id)
        attempt_order = [sampled_type] + _FALLBACK_ORDER[sampled_type]

        row = None
        for attempt_i, corruption_type in enumerate(attempt_order):
            example_rng = random.Random(
                f"{config.seed}:{example.source_id}:{corruption_type}"
            )
            try:
                row = _CORRUPTION_FUNCS[corruption_type](example, config, example_rng)
            except CorruptionError:
                continue
            if attempt_i > 0:
                stats.n_fallback += 1
            break

        if row is None:
            stats.n_skipped_all_failed += 1
            continue

        rows.append(row)
        stats.record(row["meta"]["corruption_type"])

    stats.n_produced = len(rows)
    return rows, stats
