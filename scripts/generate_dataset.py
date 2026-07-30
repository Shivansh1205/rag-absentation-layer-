#!/usr/bin/env python
"""CLI: generate the Phase 1 synthetic train/eval datasets for the RAG
abstention classifier.

Usage:
    python scripts/generate_dataset.py --n-train 5000 --n-eval 1000 --seed 42

Draws the training pool from HotpotQA's `train` split and the eval pool
from its `validation` split (rather than manually partitioning one pool),
so there is no possibility of a source question leaking between the two
generated sets. Requires network access on first run to download
HotpotQA via `datasets` (cached locally afterwards).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

from abstention_data.builder import BuildStats, build_dataset
from abstention_data.config import ChunkingConfig, GenerationConfig
from abstention_data.io import write_parquet
from abstention_data.loader import HotpotExample, load_hotpotqa

DEFAULT_POOL_MULTIPLIER = 3
DEFAULT_MIN_POOL = 2000


def _report(name: str, stats: BuildStats) -> None:
    print(f"--- {name} ---")
    print(f"  requested:            {stats.n_requested}")
    print(f"  produced:             {stats.n_produced}")
    print(f"  input pool size:      {stats.n_input_examples}")
    print(f"  skipped (unusable):   {stats.n_skipped_unusable}")
    print(f"  skipped (all failed): {stats.n_skipped_all_failed}")
    print(f"  fallback count:       {stats.n_fallback}")
    print("  corruption type counts:")
    for corruption_type, count in sorted(stats.corruption_type_counts.items()):
        print(f"    {corruption_type}: {count}")
    if stats.n_produced < stats.n_requested:
        print(
            f"  WARNING: produced {stats.n_produced} < requested {stats.n_requested}. "
            "Increase --pool-multiplier, or the source HotpotQA split doesn't have "
            "enough usable examples left.",
            file=sys.stderr,
        )


def generate_split(
    *,
    hf_split: str,
    n_rows: int,
    config: GenerationConfig,
    pool_multiplier: int,
    min_pool: int,
    load_fn=None,
) -> tuple[list[dict], BuildStats]:
    """Load a raw example pool from `hf_split`, shuffle deterministically,
    and build up to `n_rows` labeled rows from it.

    `load_fn` is injectable (defaults to the real `load_hotpotqa`, resolved
    at call time -- not bound as a function-signature default -- so that
    monkeypatching this module's `load_hotpotqa` name in tests affects
    `main()` too) so this function, and the whole CLI wiring below it, can
    be exercised in tests against a local fixture, with no network access.
    """
    load_fn = load_fn or load_hotpotqa
    pool_size = max(n_rows * pool_multiplier, min_pool)
    examples: list[HotpotExample] = load_fn(split=hf_split, limit=pool_size)
    random.Random(config.seed).shuffle(examples)
    return build_dataset(examples, config, n_rows=n_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5, help="Retrieved chunks per example")
    parser.add_argument(
        "--distractor-ratio",
        type=float,
        default=0.6,
        help=(
            "Fraction of k chunks that are distractor-derived, for "
            "gold_included/truncated_span_removed examples"
        ),
    )
    parser.add_argument(
        "--max-sentences-per-chunk",
        type=int,
        default=2,
        help="Sentence window size used to build retrieval chunks",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=DEFAULT_POOL_MULTIPLIER,
        help="Load pool_multiplier * n_rows raw HotpotQA examples per split, to absorb skips/fallbacks",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = GenerationConfig(
        k=args.k,
        distractor_ratio=args.distractor_ratio,
        seed=args.seed,
        chunking=ChunkingConfig(max_sentences_per_chunk=args.max_sentences_per_chunk),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_stats = generate_split(
        hf_split="train",
        n_rows=args.n_train,
        config=config,
        pool_multiplier=args.pool_multiplier,
        min_pool=DEFAULT_MIN_POOL,
    )
    _report("train", train_stats)
    write_parquet(train_rows, str(args.out_dir / "train.parquet"))

    eval_rows, eval_stats = generate_split(
        hf_split="validation",
        n_rows=args.n_eval,
        config=config,
        pool_multiplier=args.pool_multiplier,
        min_pool=DEFAULT_MIN_POOL,
    )
    _report("eval", eval_stats)
    write_parquet(eval_rows, str(args.out_dir / "eval.parquet"))

    print(
        f"\nWrote {len(train_rows)} train rows and {len(eval_rows)} eval rows "
        f"to {args.out_dir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
