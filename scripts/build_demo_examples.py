#!/usr/bin/env python
"""Build the demo website's `site/src/data/examples.json` from real eval-set
rows.

Selects 8 real (question, retrieved_chunks) pairs from `data/eval.parquet`
/ `data/eval_features.parquet` where the Phase 4 clean-subset classifier
(same methodology as `scripts/train_clean_subset.py`: `gold_included` vs
`distractor_only` only, `truncated_span_removed` excluded) is clearly and
correctly confident:

    - 4 highest-confidence correct `gold_included` (label=1) rows --
      "the middleware lets good retrievals through."
    - 4 lowest-confidence correct `distractor_only` (label=0) rows -- "the
      middleware catches bad retrievals and abstains."

The model here is fit fresh, in-process, purely to produce these
confidence numbers -- it is NOT saved to disk anywhere (no
`artifacts/eval_clean_subset/model.joblib` overwrite, no new pickle
committed to the repo). Same reasoning as the sklearn-version incident
documented in `docs/phase4_diagnosis.md`: a pickle written by this
sandbox's scikit-learn (1.7.2) is not safely loadable by the real venv's
(1.9.0). A transient in-memory fit has no such problem -- only the
resulting JSON numbers leave this process.

What this script does NOT fill in
------------------------------------
`question` and `chunks` (truncated to 100 chars each) are the real text
from `data/eval.parquet`. `confidence_score` is the real calibrated
`predict_proba` output. But neither `correct_answer` nor
`hallucinated_answer` can be derived from the dataset: Phase 1's final
parquet schema (`question`, `retrieved_chunks`, `label`, `meta`) never
persisted the original HotpotQA answer string past dataset generation
(see `abstention_data/loader.py` / `corruption.py` -- it's consumed
during corruption, not carried into the output row). Both fields are
written here as an explicit `"FILL_IN_MANUALLY: ..."` placeholder,
authored by hand afterward by reading the actual selected question and
chunks -- per the original request's instruction for
`hallucinated_answer`, extended to `correct_answer` for the same reason.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_demo_examples.py
    .venv\\Scripts\\python.exe scripts\\build_demo_examples.py --out site/src/data/examples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from train_clean_subset import EXCLUDED_CORRUPTION_TYPE  # noqa: E402

from abstention_data.io import read_parquet  # noqa: E402
from abstention_model.calibrate import calibrate_classifier  # noqa: E402
from abstention_model.features import load_features  # noqa: E402
from abstention_model.train import train_classifier  # noqa: E402

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_PATH = Path("site/src/data/examples.json")

N_PER_CLASS = 4
CHUNK_TRUNCATE_CHARS = 100
DEFAULT_THRESHOLD = 0.5
PLACEHOLDER = "FILL_IN_MANUALLY: read the question + chunks above and write this by hand"


def _truncate_chunk(chunk: str) -> str:
    if len(chunk) <= CHUNK_TRUNCATE_CHARS:
        return chunk
    return chunk[:CHUNK_TRUNCATE_CHARS].rstrip() + "..."


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--calibration-cv", type=int, default=5)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    train_path = args.data_dir / "train_features.parquet"
    eval_features_path = args.data_dir / "eval_features.parquet"
    eval_raw_path = args.data_dir / "eval.parquet"

    print(f"Loading {train_path} ...")
    X_train_full, y_train_full, meta_train_full = load_features(str(train_path))
    train_mask = (meta_train_full != EXCLUDED_CORRUPTION_TYPE).to_numpy()
    X_train = X_train_full.loc[train_mask].reset_index(drop=True)
    y_train = y_train_full[train_mask]
    print(f"  clean-subset train rows: {len(y_train)} (of {len(y_train_full)} total)")

    print("Fitting classifier + calibration (transient -- not saved to disk)...")
    train_classifier(X_train, y_train)
    model = calibrate_classifier(X_train, y_train, cv=args.calibration_cv)

    print(f"Loading {eval_features_path} / {eval_raw_path} ...")
    X_eval_full, y_eval_full, meta_eval_full = load_features(str(eval_features_path))
    raw_eval_df = read_parquet(str(eval_raw_path))

    if len(raw_eval_df) != len(X_eval_full):
        raise ValueError(
            f"Row count mismatch: {eval_raw_path} has {len(raw_eval_df)} rows but "
            f"{eval_features_path} has {len(X_eval_full)} -- they must be positionally "
            "aligned (same extraction run, no reordering) for this script's question/chunk "
            "lookup by index to be valid. Refusing to guess."
        )

    eval_mask = (meta_eval_full != EXCLUDED_CORRUPTION_TYPE).to_numpy()
    original_positions = np.arange(len(eval_mask))[eval_mask]
    X_eval = X_eval_full.loc[eval_mask].reset_index(drop=True)
    y_eval = y_eval_full[eval_mask]
    meta_eval = meta_eval_full.loc[eval_mask].reset_index(drop=True)
    print(f"  clean-subset eval rows: {len(y_eval)} (of {len(y_eval_full)} total)")

    p_hat = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=float)
    predicted_label = (p_hat >= DEFAULT_THRESHOLD).astype(int)
    correct = predicted_label == y_eval

    gold_correct_idx = np.where((y_eval == 1) & correct)[0]
    distractor_correct_idx = np.where((y_eval == 0) & correct)[0]
    print(
        f"  correct gold_included: {len(gold_correct_idx)} / {(y_eval == 1).sum()}, "
        f"correct distractor_only: {len(distractor_correct_idx)} / {(y_eval == 0).sum()}"
    )

    # 4 highest-confidence correct gold_included rows.
    top_gold = gold_correct_idx[np.argsort(-p_hat[gold_correct_idx])[:N_PER_CLASS]]
    # 4 lowest-confidence correct distractor_only rows (most confidently
    # correctly abstained-on).
    bottom_distractor = distractor_correct_idx[np.argsort(p_hat[distractor_correct_idx])[:N_PER_CLASS]]

    selected = list(top_gold) + list(bottom_distractor)

    examples = []
    print()
    print("=== Selected examples (review before hand-authoring answers) ===")
    for n, i in enumerate(selected, start=1):
        raw_row = raw_eval_df.iloc[original_positions[i]]
        question = str(raw_row["question"])
        chunks = [_truncate_chunk(str(c)) for c in list(raw_row["retrieved_chunks"])]
        label = int(y_eval[i])
        corruption_type = str(meta_eval.iloc[i])
        confidence = round(float(p_hat[i]), 2)

        example = {
            "id": f"ex_{n}",
            "question": question,
            "chunks": chunks,
            "label": label,
            "corruption_type": corruption_type,
            "confidence_score": confidence,
            "correct_answer": PLACEHOLDER if label == 1 else None,
            "hallucinated_answer": PLACEHOLDER if label == 0 else None,
        }
        examples.append(example)

        print(f"\n--- {example['id']} ({corruption_type}, label={label}, confidence={confidence}) ---")
        print(f"Q: {question}")
        for c in chunks:
            print(f"  chunk: {c}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nWrote {len(examples)} examples -> {args.out}")
    print(
        "\nNOTE: correct_answer / hallucinated_answer are placeholders -- "
        "the raw dataset doesn't retain HotpotQA's original answer string "
        "past Phase 1 corruption, so these must be hand-authored from the "
        "printed question/chunks above (see module docstring)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
