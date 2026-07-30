#!/usr/bin/env python
"""CLI: run Phase 2/4 feature extraction over Phase 1's `train.parquet` /
`eval.parquet` outputs, writing one row of the fixed 5-key feature
vector (see `abstention_features.pipeline.feature_names`) per input row,
alongside `label` and `meta.corruption_type`.

Phase 4: this now loads `reranker.py`'s
`cross-encoder/ms-marco-MiniLM-L6-v2`, not `entailment.py`'s
`cross-encoder/nli-deberta-v3-xsmall` -- see `pipeline.py`'s module
docstring for why (Phase 4's clean-subset AUC diagnostic found the prior
feature set couldn't distinguish "passage with the answer" from "same
passage without it," AUC 0.5495, ~chance).

**Re-running this script re-extracts all rows with the new feature set
-- any existing `*_features.parquet` written by the pre-Phase-4 pipeline
is now stale (wrong columns) and must be regenerated before retraining
the classifier.**

All three real models (`cross-encoder/ms-marco-MiniLM-L6-v2`,
`sentence-transformers/all-MiniLM-L6-v2`, `en_core_web_sm`) are loaded
exactly ONCE at startup and reused for every row via
`pipeline.extract_features`'s dependency-injection parameters --
reloading any of them per row would dominate total runtime.

Requires network access on first run to download the first two models
(cached afterwards), plus a one-time `python -m spacy download
en_core_web_sm` for the third -- see README.md.

Usage:
    uv run python scripts/extract_features.py
    uv run python scripts/extract_features.py --limit 20   # smoke test first
    uv run python scripts/extract_features.py --data-dir data --out-dir data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from abstention_data.io import read_parquet
from abstention_features import diversity, entity_coverage, reranker
from abstention_features.pipeline import extract_features, feature_names

OUTPUT_COLUMNS = list(feature_names) + ["label", "corruption_type"]

# (input filename, output filename) per split.
SPLITS = (
    ("train", "train_features.parquet"),
    ("eval", "eval_features.parquet"),
)


def load_models():
    """Load all three real models once. Returns a
    `(reranker_model, diversity_model, entity_coverage_nlp)` tuple
    ready to hand to every `pipeline.extract_features` call.

    This is the function `main()` calls by default; tests inject a fake
    replacement (same `load_models_fn=` pattern `generate_dataset.py`
    uses for `load_fn=`) so the CLI's row-processing / output-schema
    logic can be exercised with no network access and no ~GB of real
    model downloads.
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer

    import spacy

    print(f"Loading reranker model ({reranker.MODEL_NAME})...", file=sys.stderr)
    reranker_model = CrossEncoder(reranker.MODEL_NAME)

    print(f"Loading diversity model ({diversity.MODEL_NAME})...", file=sys.stderr)
    diversity_model = SentenceTransformer(diversity.MODEL_NAME)

    print(f"Loading entity_coverage model ({entity_coverage.MODEL_NAME})...", file=sys.stderr)
    entity_coverage_nlp = spacy.load(entity_coverage.MODEL_NAME)

    return reranker_model, diversity_model, entity_coverage_nlp


def extract_split(
    df: pd.DataFrame,
    *,
    reranker_model,
    diversity_model,
    entity_coverage_nlp,
    limit: Optional[int] = None,
    desc: str = "extracting features",
) -> tuple[pd.DataFrame, float]:
    """Run `pipeline.extract_features` over every row of `df` (optionally
    truncated to the first `limit` rows), with a tqdm progress bar.

    Returns `(output_dataframe, elapsed_seconds)` -- elapsed time covers
    only the extraction loop, not model loading, so it directly answers
    "how long would the full run take" independent of one-time startup
    cost.
    """
    if limit is not None:
        df = df.head(limit)

    records: list[dict] = []
    start = time.perf_counter()
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=desc):
        # Parquet round-trips a list[str] column back as a numpy array
        # (confirmed by abstention_data.io's own tests, which wrap it in
        # `list(...)` before comparing), not a plain Python list. All
        # three feature modules' empty-chunks short-circuits do
        # `if not chunks:`, which raises "the truth value of an array
        # with more than one element is ambiguous" on a >1-element
        # ndarray -- so this conversion has to happen here, once, before
        # a row ever reaches `extract_features`, rather than trusting
        # each module to defend against a non-list input.
        retrieved_chunks = list(row.retrieved_chunks)
        features = extract_features(
            row.question,
            retrieved_chunks,
            reranker_model=reranker_model,
            diversity_model=diversity_model,
            entity_coverage_nlp=entity_coverage_nlp,
        )
        record = dict(features)
        record["label"] = int(row.label)
        record["corruption_type"] = row.meta["corruption_type"]
        records.append(record)
    elapsed = time.perf_counter() - start

    out_df = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    return out_df, elapsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Directory containing train.parquet / eval.parquet"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write *_features.parquet to (defaults to --data-dir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows of each split -- e.g. --limit 20 for a quick smoke run before the full 5000/1000 job",
    )
    return parser


def main(argv: Optional[list[str]] = None, load_models_fn=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    load_models_fn = load_models_fn or load_models
    load_start = time.perf_counter()
    reranker_model, diversity_model, entity_coverage_nlp = load_models_fn()
    load_elapsed = time.perf_counter() - load_start
    print(f"Models loaded in {load_elapsed:.1f}s.\n", file=sys.stderr)

    total_rows = 0
    total_elapsed = 0.0

    for split, out_name in SPLITS:
        in_path = args.data_dir / f"{split}.parquet"
        if not in_path.exists():
            print(f"skip: {in_path} not found", file=sys.stderr)
            continue

        df = read_parquet(str(in_path))
        out_df, elapsed = extract_split(
            df,
            reranker_model=reranker_model,
            diversity_model=diversity_model,
            entity_coverage_nlp=entity_coverage_nlp,
            limit=args.limit,
            desc=f"{split} features",
        )

        out_path = out_dir / out_name
        out_df.to_parquet(out_path, engine="pyarrow", index=False)

        rows_per_sec = (len(out_df) / elapsed) if elapsed > 0 else float("inf")
        print(
            f"{split}: {len(out_df)} rows in {elapsed:.1f}s "
            f"({rows_per_sec:.2f} rows/sec) -> {out_path}"
        )
        total_rows += len(out_df)
        total_elapsed += elapsed

    if total_rows:
        overall_rate = total_rows / total_elapsed if total_elapsed > 0 else float("inf")
        print(
            f"\nTotal: {total_rows} rows in {total_elapsed:.1f}s "
            f"({overall_rate:.2f} rows/sec, extraction time only -- "
            f"model loading took an additional {load_elapsed:.1f}s)"
        )
    else:
        print(
            "\nNo input parquet files found -- nothing written. "
            "Run scripts/generate_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
