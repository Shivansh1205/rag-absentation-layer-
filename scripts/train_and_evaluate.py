#!/usr/bin/env python
"""CLI: run the full Phase 3 pipeline end to end.

load_features -> train_classifier -> calibrate_classifier -> evaluate,
saving every artifact to `--output-dir` (default `artifacts/eval/` --
reserved for evaluation outputs specifically; Phase 4 tuning is expected
to use its own `artifacts/tune/` or `artifacts/final/` alongside this):

    model.joblib                   the calibrated classifier
    calibration_curve.png          predicted probability vs. observed frequency
    calibration_curve.csv          the same, as a binned table (prob_pred, prob_true, n_per_bin)
    tradeoff_curve.png             abstention_rate / hallucination_rate_of_answered /
                                    coverage_of_answerable vs. threshold
    threshold_sweep.csv            the same, as a full 101-row table
    feature_importance.csv         permutation importance on the eval set
    error_by_corruption_type.csv   false-abstain / false-answer rates per corruption_type
    summary.json                   the headline scalar metrics in one machine-readable file

The model saved is `calibrate.py`'s `CalibratedClassifierCV`, not
`train.py`'s plain baseline -- that's the one whose `predict_proba`
output every threshold decision in this pipeline is actually meant to be
read from. `train.py`'s fit is still run first because it's cheap and
its own docstring frames it as a standalone diagnostic step, but this
script does not persist it separately.

`calibration_curve.png` and `calibration_curve.csv` are both built
directly from `evaluate()`'s own `"calibration_curve"` output (not from
a separate call to `calibrate.plot_reliability_diagram`, which would
recompute `sklearn.calibration.calibration_curve` a second time from raw
predictions) -- so the plot and the table are guaranteed to show the
same numbers, not two independent recomputations that could drift apart.

Usage:
    uv run python scripts/train_and_evaluate.py
    uv run python scripts/train_and_evaluate.py --data-dir data --output-dir artifacts/eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from abstention_model.calibrate import calibrate_classifier
from abstention_model.evaluate import evaluate
from abstention_model.features import load_features
from abstention_model.train import train_classifier

MODEL_FILENAME = "model.joblib"
CALIBRATION_CURVE_PNG_FILENAME = "calibration_curve.png"
CALIBRATION_CURVE_CSV_FILENAME = "calibration_curve.csv"
TRADEOFF_CURVE_PNG_FILENAME = "tradeoff_curve.png"
THRESHOLD_SWEEP_CSV_FILENAME = "threshold_sweep.csv"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.csv"
ERROR_BY_CORRUPTION_TYPE_FILENAME = "error_by_corruption_type.csv"
SUMMARY_FILENAME = "summary.json"

DEFAULT_OUTPUT_DIR = Path("artifacts/eval")
DEFAULT_CALIBRATION_CV = 5


def plot_tradeoff_curve(sweep: pd.DataFrame, path: str) -> None:
    """Save the abstention tradeoff curve -- `abstention_rate`,
    `hallucination_rate_of_answered`, and `coverage_of_answerable`, all
    vs. `threshold` -- as a single PNG. `evaluate.py` computes the data;
    this is one of two places it gets plotted (evaluate.py itself does
    no file I/O by design).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sweep["threshold"], sweep["abstention_rate"], label="abstention_rate")
    ax.plot(
        sweep["threshold"],
        sweep["hallucination_rate_of_answered"],
        label="hallucination_rate_of_answered",
    )
    ax.plot(sweep["threshold"], sweep["coverage_of_answerable"], label="coverage_of_answerable")
    ax.set_xlabel("Abstention threshold t  (system answers if p_hat >= t)")
    ax.set_ylabel("Rate")
    ax.set_title("Abstention tradeoff curve (eval set)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration_curve(calibration_curve: dict, path: str) -> None:
    """Save the calibration curve (mean predicted probability vs.
    observed label=1 frequency per quantile bin, plus the
    perfect-calibration diagonal) as a PNG, built from `evaluate()`'s own
    `"calibration_curve"` dict -- see module docstring for why this
    doesn't call `calibrate.plot_reliability_diagram` (which would
    recompute the same curve a second time from raw predictions).
    """
    prob_true = calibration_curve["prob_true"]
    prob_pred = calibration_curve["prob_pred"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability (prob_pred)")
    ax.set_ylabel("Observed frequency of label=1 (prob_true)")
    ax.set_title("Calibration curve (eval set, quantile bins)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _json_default(obj):
    """`json.dump` fallback for numpy scalar types (most, like
    `np.float64`, are already `float`/`int` subclasses and don't need
    this, but this is a cheap defensive net against any that aren't)."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with train_features.parquet / eval_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write all Phase 3 evaluation artifacts to",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=DEFAULT_CALIBRATION_CV,
        help="Number of CV folds CalibratedClassifierCV uses internally (see calibrate.py)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train_features.parquet"
    eval_path = args.data_dir / "eval_features.parquet"

    print(f"Loading {train_path} / {eval_path} ...")
    X_train, y_train, _ = load_features(str(train_path))
    X_eval, y_eval, meta_eval = load_features(str(eval_path))
    print(f"  train: {X_train.shape}, eval: {X_eval.shape}")

    print("Fitting baseline classifier (train.py)...")
    t0 = time.perf_counter()
    train_classifier(X_train, y_train)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print(f"Fitting calibrated classifier (calibrate.py, cv={args.calibration_cv})...")
    t0 = time.perf_counter()
    calibrated_model = calibrate_classifier(X_train, y_train, cv=args.calibration_cv)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    model_path = args.output_dir / MODEL_FILENAME
    joblib.dump(calibrated_model, model_path)
    print(f"  saved calibrated model -> {model_path}")

    print("Evaluating on eval set...")
    t0 = time.perf_counter()
    result = evaluate(calibrated_model, X_eval, y_eval, corruption_type=meta_eval)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    # Calibration curve: PNG + CSV both built from evaluate()'s own
    # already-computed dict -- see module + function docstrings.
    calibration_df = pd.DataFrame(
        {
            "prob_pred": result["calibration_curve"]["prob_pred"],
            "prob_true": result["calibration_curve"]["prob_true"],
            "n_per_bin": result["calibration_curve"]["n_per_bin"],
        }
    )
    calibration_csv_path = args.output_dir / CALIBRATION_CURVE_CSV_FILENAME
    calibration_df.to_csv(calibration_csv_path, index=False)
    print(f"  saved calibration curve CSV -> {calibration_csv_path}")

    calibration_png_path = args.output_dir / CALIBRATION_CURVE_PNG_FILENAME
    plot_calibration_curve(result["calibration_curve"], str(calibration_png_path))
    print(f"  saved calibration curve PNG -> {calibration_png_path}")

    signed_mean_deviation = float(np.mean(calibration_df["prob_true"] - calibration_df["prob_pred"]))
    abs_mean_deviation = float(np.mean(np.abs(calibration_df["prob_true"] - calibration_df["prob_pred"])))

    tradeoff_png_path = args.output_dir / TRADEOFF_CURVE_PNG_FILENAME
    plot_tradeoff_curve(result["threshold_sweep"], str(tradeoff_png_path))
    print(f"  saved tradeoff curve PNG -> {tradeoff_png_path}")

    threshold_sweep_path = args.output_dir / THRESHOLD_SWEEP_CSV_FILENAME
    result["threshold_sweep"].to_csv(threshold_sweep_path, index=False)
    print(f"  saved threshold sweep CSV -> {threshold_sweep_path}")

    importance_path = args.output_dir / FEATURE_IMPORTANCE_FILENAME
    result["permutation_importance"].to_csv(importance_path, index=False)
    print(f"  saved feature importance table -> {importance_path}")

    if result["error_by_corruption_type"] is not None:
        error_path = args.output_dir / ERROR_BY_CORRUPTION_TYPE_FILENAME
        result["error_by_corruption_type"].to_csv(error_path, index=False)
        print(f"  saved error-by-corruption_type -> {error_path}")

    cm = result["confusion_matrix_at_default"]
    summary = {
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
    summary_path = args.output_dir / SUMMARY_FILENAME
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"  saved summary -> {summary_path}")

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
    print("Top permutation-importance features (eval set):")
    print(result["permutation_importance"].head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
