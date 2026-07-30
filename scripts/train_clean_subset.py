#!/usr/bin/env python
"""Phase 4 experiment: retrain + evaluate on the "clean" subset only --
`gold_included` vs `distractor_only`, with `truncated_span_removed`
filtered out entirely.

Why this experiment exists
---------------------------
`docs/phase4_diagnosis.md`'s Step 1 diagnostic (post-reranker-swap)
found:

    full eval set (all 3 corruption types):        ROC-AUC 0.7108
    gold_included vs distractor_only (clean):       ROC-AUC 0.8413
    gold_included vs truncated_span_removed (hard):  ROC-AUC 0.5691

The clean pair is comfortably above the full-eval baseline and the hard
pair sits barely above chance -- this now reads as a label problem, not a
feature problem: `truncated_span_removed` (same passage, answer sentence
surgically removed) is capping the achievable ceiling, and the reranker
feature that fixed the entailment-era feature gap still can't reliably
tell those apart. This script asks the follow-up question directly: if
the classifier is trained AND evaluated only on the clean pair (the
scenario Step 1 already showed the features handle reasonably well), how
high does AUC actually go?

No re-extraction, no changes to existing code
-------------------------------------------------
This reuses the *already-extracted* `data/train_features.parquet` /
`data/eval_features.parquet` (the reranker-based Phase 4 features) --
`truncated_span_removed` rows are filtered out in memory via
`abstention_model.features.load_features`'s existing `meta` return value,
nothing is re-extracted or re-parsed. Training/calibration/evaluation
call the exact same `abstention_model.train.train_classifier`,
`abstention_model.calibrate.calibrate_classifier`, and
`abstention_model.evaluate.evaluate` functions `train_and_evaluate.py`
uses -- same hyperparameters, same calibration method -- nothing in
those modules is modified. Plotting/summary-serialization helpers
(`plot_tradeoff_curve`, `plot_calibration_curve`, `_json_default`) and
the output filename constants are imported directly from
`train_and_evaluate.py` rather than copy-pasted, so this script cannot
silently drift out of sync with how the full-eval run renders the same
data.

All outputs go to `--output-dir` (default `artifacts/eval_clean_subset/`)
-- a separate directory from `artifacts/eval/`, so this experiment never
overwrites the full-eval artifacts.

Usage:
    .venv\\Scripts\\python.exe scripts\\train_clean_subset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# scripts/ isn't a package -- import train_and_evaluate.py's plotting/
# summary helpers and filename constants by path, same pattern
# tests/test_extract_features_script.py uses for extract_features.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import train_and_evaluate  # noqa: E402

from abstention_model.calibrate import calibrate_classifier  # noqa: E402
from abstention_model.evaluate import evaluate  # noqa: E402
from abstention_model.features import load_features  # noqa: E402
from abstention_model.train import train_classifier  # noqa: E402

EXCLUDED_CORRUPTION_TYPE = "truncated_span_removed"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("artifacts/eval_clean_subset")

# Reference number from docs/phase4_diagnosis.md's Step 1 diagnostic
# (scripts/diagnose_subset_auc.py, run against the reranker-based
# artifacts/eval/model.joblib). Used only for the printed comparison
# table below -- this script does not recompute it, since that requires
# the full (unfiltered) eval set and the full-data model, not this
# experiment's filtered one. If artifacts/eval/summary.json exists, its
# roc_auc is used instead (more precision than the number in the docs),
# falling back to this constant if that file isn't present.
FULL_EVAL_AUC_REFERENCE = 0.7108
FULL_EVAL_SUMMARY_PATH = Path("artifacts/eval/summary.json")


def _filter_clean_subset(
    X: pd.DataFrame, y: np.ndarray, meta: pd.Series
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Drop every row whose corruption_type is `EXCLUDED_CORRUPTION_TYPE`,
    keeping `X`/`y`/`meta` aligned and re-indexed."""
    mask = (meta != EXCLUDED_CORRUPTION_TYPE).to_numpy()
    X_filtered = X.loc[mask].reset_index(drop=True)
    y_filtered = y[mask]
    meta_filtered = meta.loc[mask].reset_index(drop=True)
    return X_filtered, y_filtered, meta_filtered


def _print_split_summary(name: str, y: np.ndarray, meta: pd.Series) -> None:
    print(f"{name}: {len(y)} rows after filtering out '{EXCLUDED_CORRUPTION_TYPE}'")
    print("  corruption_type counts:")
    for corruption_type, count in meta.value_counts().items():
        print(f"    {corruption_type}: {count}")
    print("  label balance:")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n = len(y)
    pos_pct = (n_pos / n * 100) if n else float("nan")
    neg_pct = (n_neg / n * 100) if n else float("nan")
    print(f"    label=1 (answerable):   {n_pos} ({pos_pct:.1f}%)")
    print(f"    label=0 (unanswerable): {n_neg} ({neg_pct:.1f}%)")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with train_features.parquet / eval_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write all clean-subset experiment artifacts to",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=train_and_evaluate.DEFAULT_CALIBRATION_CV,
        help="Number of CV folds CalibratedClassifierCV uses internally",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train_features.parquet"
    eval_path = args.data_dir / "eval_features.parquet"

    print(f"Loading {train_path} / {eval_path} ...")
    X_train_full, y_train_full, meta_train_full = load_features(str(train_path))
    X_eval_full, y_eval_full, meta_eval_full = load_features(str(eval_path))
    print(f"  train (unfiltered): {X_train_full.shape}, eval (unfiltered): {X_eval_full.shape}")
    print()

    print(f"Filtering out '{EXCLUDED_CORRUPTION_TYPE}' from both splits...")
    X_train, y_train, meta_train = _filter_clean_subset(X_train_full, y_train_full, meta_train_full)
    X_eval, y_eval, meta_eval = _filter_clean_subset(X_eval_full, y_eval_full, meta_eval_full)
    print()
    _print_split_summary("train", y_train, meta_train)
    _print_split_summary("eval", y_eval, meta_eval)

    print("Fitting baseline classifier (train.py, same hyperparameters)...")
    t0 = time.perf_counter()
    train_classifier(X_train, y_train)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print(f"Fitting calibrated classifier (calibrate.py, cv={args.calibration_cv})...")
    t0 = time.perf_counter()
    calibrated_model = calibrate_classifier(X_train, y_train, cv=args.calibration_cv)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    model_path = args.output_dir / train_and_evaluate.MODEL_FILENAME
    joblib.dump(calibrated_model, model_path)
    print(f"  saved calibrated model -> {model_path}")

    print("Evaluating on filtered (clean-subset) eval set...")
    t0 = time.perf_counter()
    result = evaluate(calibrated_model, X_eval, y_eval, corruption_type=meta_eval)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # Calibration curve: PNG + CSV both built from evaluate()'s own dict --
    # same approach train_and_evaluate.py uses, reusing its plotting
    # function directly rather than re-deriving it here.
    calibration_df = pd.DataFrame(
        {
            "prob_pred": result["calibration_curve"]["prob_pred"],
            "prob_true": result["calibration_curve"]["prob_true"],
            "n_per_bin": result["calibration_curve"]["n_per_bin"],
        }
    )
    calibration_csv_path = args.output_dir / train_and_evaluate.CALIBRATION_CURVE_CSV_FILENAME
    calibration_df.to_csv(calibration_csv_path, index=False)
    print(f"  saved calibration curve CSV -> {calibration_csv_path}")

    calibration_png_path = args.output_dir / train_and_evaluate.CALIBRATION_CURVE_PNG_FILENAME
    train_and_evaluate.plot_calibration_curve(result["calibration_curve"], str(calibration_png_path))
    print(f"  saved calibration curve PNG -> {calibration_png_path}")

    signed_mean_deviation = float(np.mean(calibration_df["prob_true"] - calibration_df["prob_pred"]))
    abs_mean_deviation = float(np.mean(np.abs(calibration_df["prob_true"] - calibration_df["prob_pred"])))

    tradeoff_png_path = args.output_dir / train_and_evaluate.TRADEOFF_CURVE_PNG_FILENAME
    train_and_evaluate.plot_tradeoff_curve(result["threshold_sweep"], str(tradeoff_png_path))
    print(f"  saved tradeoff curve PNG -> {tradeoff_png_path}")

    threshold_sweep_path = args.output_dir / train_and_evaluate.THRESHOLD_SWEEP_CSV_FILENAME
    result["threshold_sweep"].to_csv(threshold_sweep_path, index=False)
    print(f"  saved threshold sweep CSV -> {threshold_sweep_path}")

    importance_path = args.output_dir / train_and_evaluate.FEATURE_IMPORTANCE_FILENAME
    result["permutation_importance"].to_csv(importance_path, index=False)
    print(f"  saved feature importance table -> {importance_path}")

    if result["error_by_corruption_type"] is not None:
        error_path = args.output_dir / train_and_evaluate.ERROR_BY_CORRUPTION_TYPE_FILENAME
        result["error_by_corruption_type"].to_csv(error_path, index=False)
        print(f"  saved error-by-corruption_type -> {error_path}")

    cm = result["confusion_matrix_at_default"]
    summary = {
        "experiment": "clean_subset (gold_included vs distractor_only only, "
        f"'{EXCLUDED_CORRUPTION_TYPE}' filtered out)",
        "train_rows": int(len(y_train)),
        "eval_rows": int(len(y_eval)),
        "default_threshold": cm["threshold"],
        "roc_auc": result["roc_auc"],
        "pr_auc_positive_class_1": result["pr_auc_positive_class_1"],
        "pr_auc_positive_class_0": result["pr_auc_positive_class_0"],
        "confusion_matrix": {
            "labels": cm["labels"],
            "matrix": np.asarray(cm["matrix"]).tolist(),
            "note": cm["note"],
        },
        "calibration_signed_mean_deviation": signed_mean_deviation,
        "calibration_abs_mean_deviation": abs_mean_deviation,
        "top_permutation_importance_features": result["permutation_importance"]
        .head(4)
        .to_dict(orient="records"),
    }
    summary_path = args.output_dir / train_and_evaluate.SUMMARY_FILENAME
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=train_and_evaluate._json_default)
    print(f"  saved summary -> {summary_path}")

    # Reference full-eval AUC: prefer the real artifacts/eval/summary.json
    # if present (more precision than the docs' rounded number), else fall
    # back to the documented constant.
    full_eval_auc = FULL_EVAL_AUC_REFERENCE
    full_eval_source = f"docs/phase4_diagnosis.md reference constant ({FULL_EVAL_AUC_REFERENCE})"
    if FULL_EVAL_SUMMARY_PATH.exists():
        try:
            with open(FULL_EVAL_SUMMARY_PATH) as f:
                full_eval_auc = json.load(f)["roc_auc"]
            full_eval_source = str(FULL_EVAL_SUMMARY_PATH)
        except (json.JSONDecodeError, KeyError):
            pass  # keep the fallback constant if the file is malformed/missing the key

    print()
    print("=== Summary ===")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC (positive=label 1, answerable):            {result['pr_auc_positive_class_1']:.4f}")
    print(f"PR-AUC (positive=label 0, hallucination-risk):    {result['pr_auc_positive_class_0']:.4f}")
    print(
        f"Confusion matrix at threshold={cm['threshold']} "
        f"(rows=true, cols=predicted, labels={cm['labels']}; {cm['note']}):"
    )
    print(cm["matrix"])
    print(
        f"Calibration deviation: signed={signed_mean_deviation:.4f}, "
        f"abs={abs_mean_deviation:.4f}"
    )
    print()
    print("Top permutation-importance features (clean-subset eval set):")
    print(result["permutation_importance"].head(4).to_string(index=False))
    print()
    print("=== Full eval vs. clean subset ===")
    print(f"Full eval (all corruption types):  AUC = {full_eval_auc:.4f}  (source: {full_eval_source})")
    print(f"Clean subset (gold vs distractor): AUC = {result['roc_auc']:.4f}")
    print(f"Delta: {result['roc_auc'] - full_eval_auc:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
