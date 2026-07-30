#!/usr/bin/env python
"""Sanity-check a generated dataset by hand: print a sample of rows per
corruption type, alongside the original HotpotQA answer, and
automatically flag any label=0 row whose `retrieved_chunks` still
contains that answer span.

Usage:
    uv run python scripts/inspect_sample.py --data-dir data --n-per-type 3

Requires network access: the output parquet intentionally does not store
the answer text (see README schema), so this re-loads the relevant
HotpotQA split(s) to resolve each sampled row's answer by
`meta.source_id`. Exits with code 1 if any leak is found.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

from abstention_data.chunking import contains_answer_span
from abstention_data.io import read_parquet
from abstention_data.loader import load_hotpotqa

# Mirrors the split mapping in generate_dataset.py: our "train" rows were
# drawn from HotpotQA's "train" split, our "eval" rows from "validation".
SPLIT_TO_HF_SPLIT = {"train": "train", "eval": "validation"}


def load_answer_lookup(hf_split: str, load_fn=None) -> dict:
    """Build a source_id -> answer map by loading the full HotpotQA split.

    `load_fn` is injectable for testing against a local fixture, same
    pattern as `scripts/generate_dataset.py`.
    """
    load_fn = load_fn or load_hotpotqa
    print(
        f"Loading HotpotQA[{hf_split}] to resolve answers by source_id "
        "(one-time cost for this check)...",
        file=sys.stderr,
    )
    examples = load_fn(split=hf_split, limit=None)
    return {ex.source_id: ex.answer for ex in examples}


def inspect_split(
    data_dir: Path, split: str, n_per_type: int, seed: int, load_fn=None
) -> bool:
    """Print a sample of `split`'s rows and return False if any label=0
    row is found to leak its answer."""
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        print(f"skip: {path} not found", file=sys.stderr)
        return True

    df = read_parquet(str(path))
    answers = load_answer_lookup(SPLIT_TO_HF_SPLIT[split], load_fn=load_fn)

    corruption_types = sorted({m["corruption_type"] for m in df["meta"]})
    rng = random.Random(seed)
    all_ok = True

    print(f"\n===== {split} ({len(df)} rows) =====")
    print("corruption_type counts: " + ", ".join(
        f"{ct}={sum(1 for m in df['meta'] if m['corruption_type'] == ct)}"
        for ct in corruption_types
    ))

    for corruption_type in corruption_types:
        subset_idx = [i for i, m in enumerate(df["meta"]) if m["corruption_type"] == corruption_type]
        n_sample = min(n_per_type, len(subset_idx))
        sample_idx = rng.sample(subset_idx, n_sample)

        print(f"\n--- corruption_type = {corruption_type!r} ({len(subset_idx)} rows total, showing {n_sample}) ---")
        for idx in sample_idx:
            row = df.iloc[idx]
            source_id = row["meta"]["source_id"]
            answer = answers.get(source_id)
            answer_display = repr(answer) if answer is not None else "<source_id not found>"

            print(f"\n  question: {row['question']}")
            print(f"  label: {row['label']}   answer: {answer_display}")
            for i, chunk in enumerate(row["retrieved_chunks"]):
                leaks = bool(answer) and contains_answer_span(chunk, answer)
                flag = "  <-- CONTAINS ANSWER" if leaks else ""
                print(f"    [{i}] {chunk}{flag}")
                if row["label"] == 0 and leaks:
                    all_ok = False

    return all_ok


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-per-type", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    all_ok = True
    for split in ("train", "eval"):
        ok = inspect_split(args.data_dir, split, args.n_per_type, args.seed)
        all_ok = all_ok and ok

    print(
        "\n"
        + (
            "PASS: no label=0 row leaked its answer."
            if all_ok
            else "FAIL: see 'CONTAINS ANSWER' flags above."
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
