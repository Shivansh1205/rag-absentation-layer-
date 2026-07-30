#!/usr/bin/env python
"""Phase 4, Step 1 diagnostic: clean-subset AUC.

Re-evaluates the EXISTING trained/calibrated model (loaded from
`artifacts/eval/model.joblib` -- not retrained) on eval-set subsets sliced
by `corruption_type`, to localize whether Phase 3's ~0.6735 ROC-AUC
ceiling is a feature problem or a label-noise problem.

Phase 1's corruption types each map to exactly one label (documented in
`evaluate.py`'s `_error_by_corruption_type`):

    gold_included            -> label=1 (answerable)
    distractor_only          -> label=0 (unanswerable, easy/unrelated)
    truncated_span_removed   -> label=0 (unanswerable, hard/near-miss)

Three subsets:

  a. "clean":  gold_included vs distractor_only only (excludes
     truncated_span_removed) -- the least ambiguous label pair: answer
     clearly present vs. clearly, topically absent.
  b. "full":   the entire eval set (Phase 3's baseline, ~0.6735).
  c. "hard":   gold_included vs truncated_span_removed only -- the
     near-miss pair `error_by_corruption_type` already flagged as the
     likely-harder case (32.2% false-answer rate vs. distractor_only's
     8.2%, at t=0.5).

Interpretation (see printed output for which one this run matched):

  - clean AUC jumps well above full AUC, hard AUC lags behind both =>
    features separate the unambiguous cases fine; truncated_span_removed
    is genuinely ambiguous/noisy-labeled and is capping the ceiling. Fix
    is data-side.
  - clean AUC stays near full AUC (~0.67) => features are the bottleneck
    regardless of label quality. Fix is richer features.

No retraining happens anywhere in this script: it loads the exact joblib
artifact `scripts/train_and_evaluate.py` already wrote and only calls
`predict_proba` + `roc_auc_score` on subsets of the existing eval set.

Usage:
    .venv\\Scripts\\python.exe scripts\\diagnose_subset_auc.py
    .venv\\Scripts\\python.exe scripts\\diagnose_subset_auc.py \\
        --model artifacts/eval/model.joblib --eval-path data/eval_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from abstention_model.features import load_features

GOLD = "gold_included"
DISTRACTOR = "distractor_only"
TRUNCATED = "truncated_span_removed"

DEFAULT_MODEL_PATH = Path("artifacts/eval/model.joblib")
DEFAULT_EVAL_PATH = Path("data/eval_features.parquet")

# AUC threshold used only to pick which interpretation message to print --
# not a scientific cutoff, just makes the printed verdict self-contained
# instead of requiring the reader to eyeball three floats.
CLEAN_AUC_HIGH_BAR = 0.80
CLEAN_AUC_LOW_BAR = 0.75


def _subset_auc(
    y: np.ndarray, p_hat: np.ndarray, corruption_type: pd.Series, keep_types: set[str]
) -> dict:
    """ROC-AUC (and row counts) restricted to rows whose corruption_type
    is in `keep_types`. Returns `auc=nan` if the subset has only one
    class present (AUC is undefined there), rather than letting
    `roc_auc_score` raise.
    """
    mask = corruption_type.isin(keep_types).to_numpy()
    y_sub = y[mask]
    p_sub = p_hat[mask]
    n_pos = int((y_sub == 1).sum())
    n_neg = int((y_sub == 0).sum())
    auc = float(roc_auc_score(y_sub, p_sub)) if n_pos > 0 and n_neg > 0 else float("nan")
    return {"n": int(mask.sum()), "n_pos": n_pos, "n_neg": n_neg, "auc": auc}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    print(f"Loading model from {args.model} (no retraining) ...")
    model = joblib.load(args.model)

    print(f"Loading eval features from {args.eval_path} ...")
    X_eval, y_eval, meta_eval = load_features(str(args.eval_path))
    y_eval = np.asarray(y_eval, dtype=int)

    print(f"Eval set: {len(y_eval)} rows")
    print("corruption_type counts:")
    print(meta_eval.value_counts().to_string())
    print()

    p_hat = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=float)

    results = {
        "a_clean_gold_vs_distractor": _subset_auc(y_eval, p_hat, meta_eval, {GOLD, DISTRACTOR}),
        "b_full_eval_set": _subset_auc(y_eval, p_hat, meta_eval, {GOLD, DISTRACTOR, TRUNCATED}),
        "c_hard_gold_vs_truncated": _subset_auc(y_eval, p_hat, meta_eval, {GOLD, TRUNCATED}),
    }

    print("=== Step 1: clean-subset AUC ===")
    for name, r in results.items():
        print(
            f"{name}: n={r['n']} (n_pos={r['n_pos']}, n_neg={r['n_neg']})  "
            f"ROC-AUC={r['auc']:.4f}"
        )
    print()

    clean_auc = results["a_clean_gold_vs_distractor"]["auc"]
    full_auc = results["b_full_eval_set"]["auc"]
    hard_auc = results["c_hard_gold_vs_truncated"]["auc"]

    print("=== Interpretation ===")
    if clean_auc >= CLEAN_AUC_HIGH_BAR and hard_auc < full_auc:
        print(
            f"Clean-subset AUC ({clean_auc:.4f}) is substantially higher than "
            f"the full-eval AUC ({full_auc:.4f}), and the hard subset "
            f"({hard_auc:.4f}) lags behind both. Features separate the "
            f"unambiguous cases well -- '{TRUNCATED}' looks like the thing "
            "capping the ceiling, not the feature set. This points at a "
            "data-side fix (revise/reweight/relabel that corruption type)."
        )
    elif clean_auc < CLEAN_AUC_LOW_BAR:
        print(
            f"Clean-subset AUC ({clean_auc:.4f}) stays close to the full-eval "
            f"AUC ({full_auc:.4f}) even on the least-ambiguous label pair "
            f"(gold_included vs. {DISTRACTOR}). Features are the bottleneck "
            "regardless of label quality -- this points at a feature-side "
            "fix (a real query-passage relevance/reranker feature)."
        )
    else:
        print(
            f"Clean-subset AUC ({clean_auc:.4f}) sits between the two clean "
            f"predicted patterns (full={full_auc:.4f}, hard={hard_auc:.4f}) "
            "-- doesn't cleanly match either hypothesis. Inspect the three "
            "numbers directly before deciding which fix to pursue."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
