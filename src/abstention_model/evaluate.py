"""The abstention tradeoff curve, plus the standard metrics/diagnostics
that go with it.

This module computes everything from a **calibrated** classifier's
`predict_proba(X_eval)[:, 1]` -- call it `p_hat`, the calibrated
P(label=1 | features) -- against the eval set's true binary `label`
(1=answerable, 0=unanswerable). Decision rule throughout: the system
**answers** if `p_hat >= t`, **abstains** if `p_hat < t`.

Metric definitions -- the denominators are the entire point
----------------------------------------------------------------
Every rate below was reviewed and confirmed before this module was
written; documented here so the definitions travel with the code, not
just the design conversation that produced them.

- `abstention_rate(t)` = (# rows with `p_hat < t`) / N, N = total eval
  rows. Fixed denominator, the whole eval set.
- `hallucination_rate_of_answered(t)` = (# rows with `p_hat >= t` AND
  `label==0`) / (# rows with `p_hat >= t`). Denominator is the
  *answered* subset at that threshold, NOT N -- dividing by N would
  understate real hallucination risk once abstention rate is high.
  **`NaN` (not `0.0`) when nothing is answered at that threshold** --
  "zero hallucinations because nothing was answered" is a materially
  different fact from "zero hallucinations among a real answered set",
  and collapsing them to the same number would hide small-sample noise
  at high thresholds (e.g. 1 bad prediction out of 4 answered rows is
  25%, not a stable estimate, and 0 out of 0 is not 0% at all).
- `coverage_of_answerable(t)` = (# rows with `p_hat >= t` AND
  `label==1`) / (# rows with `label==1`, total). Denominator is **fixed
  across the whole sweep** -- the total count of genuinely answerable
  eval rows, never the answered subset at that threshold. This is
  recall restricted to the label=1 slice: of the questions that were
  truly answerable, what fraction does the system still attempt at this
  threshold. A shrinking denominator here would silently turn this into
  a different metric (precision-like) that can't show coverage actually
  degrading as `t` rises, which is the entire point of plotting it.

Raw counts (`n_answered`, `n_answered_and_hallucinated`,
`n_answered_and_correct`, `n_abstained_from_answerable`) are returned
alongside every rate so the sweep table is self-diagnosable without
recomputing anything from scratch.

Threshold grid: always the fixed 101-point `np.linspace(0.0, 1.0, 101)`
-- never restricted to observed probability values (unlike an ROC
curve's unique-threshold sweep). The boundary rows (`t=0.0`: everything
answered; `t=1.0`: essentially nothing answered) are included
deliberately, not trimmed -- they're the conditions that make the rest
of the curve interpretable.

PR-AUC: which class is "positive" is reported as two separate,
explicitly labeled numbers (`pr_auc_positive_class_1`,
`pr_auc_positive_class_0`) rather than one ambiguous "PR-AUC", since
which class is of interest depends on which side of the tradeoff is
being optimized (answerable-detection vs. hallucination-risk-detection).
`pr_auc_positive_class_0` is computed by re-deriving both the true
labels and the scores for that framing (`label==0` as the positive
class, `1 - p_hat` as its score) rather than passing `pos_label=0` to
`average_precision_score` with `p_hat` unchanged -- `p_hat` is a score
for class 1, and using it un-inverted with `pos_label=0` would rank rows
backwards for that computation.

Confusion matrix and error-by-corruption_type both use a single shared
`default_threshold` (default `0.5`) rather than two independently
configurable knobs -- unifying two knobs that diverged later is
annoying; splitting one that turns out to need it is trivial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

N_THRESHOLDS = 101
THRESHOLD_GRID = np.linspace(0.0, 1.0, N_THRESHOLDS)

DEFAULT_THRESHOLD = 0.5

CALIBRATION_N_BINS = 10
CALIBRATION_STRATEGY = "quantile"

PERMUTATION_N_REPEATS = 20
PERMUTATION_RANDOM_STATE = 42
PERMUTATION_SCORING = "roc_auc"


def _threshold_sweep(p_hat: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Vectorized computation of the full tradeoff-curve table: one row
    per threshold in `THRESHOLD_GRID`, no per-threshold Python loop and
    no per-row loop. See module docstring for the exact metric/denominator
    definitions.
    """
    p_hat = np.asarray(p_hat, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    n_positive = int((y == 1).sum())

    # (n_rows, n_thresholds) boolean matrix via broadcasting:
    # answered_matrix[i, j] == True iff row i is "answered" at
    # THRESHOLD_GRID[j] (p_hat[i] >= THRESHOLD_GRID[j]).
    answered_matrix = p_hat[:, None] >= THRESHOLD_GRID[None, :]
    abstained_matrix = ~answered_matrix

    y_is_1 = (y == 1)[:, None]
    y_is_0 = (y == 0)[:, None]

    n_answered = answered_matrix.sum(axis=0)
    n_answered_and_hallucinated = (answered_matrix & y_is_0).sum(axis=0)
    n_answered_and_correct = (answered_matrix & y_is_1).sum(axis=0)
    n_abstained_from_answerable = (abstained_matrix & y_is_1).sum(axis=0)

    abstention_rate = abstained_matrix.sum(axis=0) / n

    # hallucination_rate_of_answered: NaN, not 0.0, when n_answered==0
    # for that threshold -- explicit np.where rather than relying on
    # 0/0 -> nan falling out of float division, so the "undefined when
    # nothing answered" behavior is self-documenting in the code, not
    # an IEEE-754 coincidence.
    with np.errstate(invalid="ignore", divide="ignore"):
        hallucination_rate_of_answered = np.where(
            n_answered == 0,
            np.nan,
            n_answered_and_hallucinated / np.where(n_answered == 0, 1, n_answered),
        )

    # coverage_of_answerable: denominator is n_positive, fixed across
    # every threshold -- never n_answered. NaN only in the degenerate
    # case where the eval set has zero label=1 rows at all (nothing to
    # take coverage of).
    if n_positive == 0:
        coverage_of_answerable = np.full(N_THRESHOLDS, np.nan)
    else:
        coverage_of_answerable = n_answered_and_correct / n_positive

    return pd.DataFrame(
        {
            "threshold": THRESHOLD_GRID,
            "abstention_rate": abstention_rate,
            "hallucination_rate_of_answered": hallucination_rate_of_answered,
            "coverage_of_answerable": coverage_of_answerable,
            "n_answered": n_answered.astype(int),
            "n_answered_and_hallucinated": n_answered_and_hallucinated.astype(int),
            "n_answered_and_correct": n_answered_and_correct.astype(int),
            "n_abstained_from_answerable": n_abstained_from_answerable.astype(int),
        }
    )


def _calibration_bin_counts(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """Recover each non-empty bin's sample count from the same
    quantile-binning algorithm `sklearn.calibration.calibration_curve`
    uses internally (bin edges via percentile, `np.searchsorted`
    assignment, empty bins dropped) -- `calibration_curve` itself
    doesn't return bin counts, but this must stay in lockstep with its
    binning so `n_per_bin` lines up index-for-index with the
    `prob_true`/`prob_pred` arrays computed from the same inputs.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.percentile(y_prob, quantiles * 100)
    binids = np.searchsorted(bins[1:-1], y_prob)
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0
    return bin_total[nonzero]


def _error_by_corruption_type(
    y: np.ndarray, p_hat: np.ndarray, corruption_type, threshold: float
) -> pd.DataFrame:
    """Per-`corruption_type` breakdown at a single fixed `threshold`.

    Note on this dataset's structure (worth knowing before reading the
    output): Phase 1's corruption types each map to exactly one label
    value (`gold_included` -> label=1 only; `distractor_only` and
    `truncated_span_removed` -> label=0 only). That means, for any given
    corruption_type group, one of `false_abstain_rate` /
    `false_answer_rate` is *always* `NaN` here -- there being zero
    label=1 rows in a label=0-only group makes "false abstain rate"
    undefined for that group, and vice versa. That's a real property of
    the data, not a bug: comparing whether `truncated_span_removed`
    drives more hallucination-risk errors than `distractor_only` means
    comparing their `false_answer_rate` columns specifically (their
    `false_abstain_rate` will both be `NaN`), while `gold_included`'s
    only meaningful column is `false_abstain_rate`.
    """
    y = np.asarray(y, dtype=int)
    p_hat = np.asarray(p_hat, dtype=float)
    corruption_type = pd.Series(corruption_type).reset_index(drop=True).to_numpy()

    df = pd.DataFrame({"y": y, "p_hat": p_hat, "corruption_type": corruption_type})

    rows = []
    for group_name, group in df.groupby("corruption_type", sort=True):
        y_g = group["y"].to_numpy()
        p_g = group["p_hat"].to_numpy()

        n_pos = int((y_g == 1).sum())
        n_neg = int((y_g == 0).sum())

        false_abstain = int(np.sum((y_g == 1) & (p_g < threshold)))
        false_answer = int(np.sum((y_g == 0) & (p_g >= threshold)))

        false_abstain_rate = (false_abstain / n_pos) if n_pos > 0 else np.nan
        false_answer_rate = (false_answer / n_neg) if n_neg > 0 else np.nan

        rows.append(
            {
                "corruption_type": group_name,
                "count": int(len(group)),
                "false_abstain_rate": false_abstain_rate,
                "false_answer_rate": false_answer_rate,
            }
        )

    return pd.DataFrame(rows, columns=["corruption_type", "count", "false_abstain_rate", "false_answer_rate"])


def evaluate(
    model,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    corruption_type: pd.Series | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute the full Phase 3 evaluation bundle for a calibrated
    `model` on `(X_eval, y_eval)`.

    Pure computation -- no file writes, no plotting, no side effects.
    `scripts/train_and_evaluate.py` is responsible for saving any of
    this to disk.

    Returns a dict with keys:

    - `"threshold_sweep"`: DataFrame, one row per threshold in
      `THRESHOLD_GRID` (101 rows). See module docstring for exact
      column definitions.
    - `"roc_auc"`: float, `roc_auc_score(y_eval, p_hat)`.
    - `"pr_auc_positive_class_1"`: float, average precision with
      label=1 (answerable) as the positive class.
    - `"pr_auc_positive_class_0"`: float, average precision with
      label=0 (unanswerable / hallucination-risk) as the positive class
      -- computed from `1 - p_hat` against `label==0`, not via
      `pos_label=0` on the unmodified score (see module docstring).
    - `"confusion_matrix_at_default"`: dict with `"matrix"` (2x2
      `numpy.ndarray`, `labels=[0, 1]`, rows=true/cols=predicted),
      `"labels"`, `"threshold"`, and an explicit `"note"` that this is
      one illustrative point on the tradeoff curve, not a claimed
      optimal operating point.
    - `"calibration_curve"`: dict with `"prob_true"`, `"prob_pred"`
      (from `sklearn.calibration.calibration_curve`, `n_bins=10`,
      `strategy="quantile"`) and `"n_per_bin"` (recovered separately --
      see `_calibration_bin_counts`).
    - `"permutation_importance"`: DataFrame with columns `["feature",
      "importance_mean", "importance_std"]`, sorted descending by
      `importance_mean`; computed on `(X_eval, y_eval)` (not train --
      train would partly measure memorization), `scoring="roc_auc"`,
      `n_repeats=20`, fixed `random_state=42`.
    - `"error_by_corruption_type"`: DataFrame (see
      `_error_by_corruption_type`), or `None` if `corruption_type` is
      not provided.
    """
    p_hat = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=float)
    y_eval = np.asarray(y_eval, dtype=int)

    threshold_sweep = _threshold_sweep(p_hat, y_eval)

    roc_auc = float(roc_auc_score(y_eval, p_hat))
    pr_auc_positive_class_1 = float(average_precision_score(y_eval, p_hat))
    pr_auc_positive_class_0 = float(
        average_precision_score((y_eval == 0).astype(int), 1.0 - p_hat)
    )

    predicted_at_default = (p_hat >= default_threshold).astype(int)
    confusion_matrix_at_default = {
        "matrix": confusion_matrix(y_eval, predicted_at_default, labels=[0, 1]),
        "labels": [0, 1],
        "threshold": default_threshold,
        "note": (
            f"One illustrative point on the abstention tradeoff curve at "
            f"threshold={default_threshold!r}, not a claimed optimal "
            "operating point -- see 'threshold_sweep' for the full curve."
        ),
    }

    prob_true, prob_pred = calibration_curve(
        y_eval, p_hat, n_bins=CALIBRATION_N_BINS, strategy=CALIBRATION_STRATEGY
    )
    n_per_bin = _calibration_bin_counts(y_eval, p_hat, CALIBRATION_N_BINS)
    assert len(n_per_bin) == len(prob_true) == len(prob_pred), (
        "calibration_curve bin count mismatch -- _calibration_bin_counts has "
        "drifted out of sync with sklearn.calibration.calibration_curve's "
        "internal binning algorithm."
    )
    calibration_curve_result = {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "n_per_bin": n_per_bin,
    }

    perm_result = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring=PERMUTATION_SCORING,
        n_repeats=PERMUTATION_N_REPEATS,
        random_state=PERMUTATION_RANDOM_STATE,
    )
    feature_names = (
        list(X_eval.columns)
        if hasattr(X_eval, "columns")
        else [f"feature_{i}" for i in range(np.asarray(X_eval).shape[1])]
    )
    permutation_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": perm_result.importances_mean,
                "importance_std": perm_result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    error_by_corruption_type = (
        _error_by_corruption_type(y_eval, p_hat, corruption_type, default_threshold)
        if corruption_type is not None
        else None
    )

    return {
        "threshold_sweep": threshold_sweep,
        "roc_auc": roc_auc,
        "pr_auc_positive_class_1": pr_auc_positive_class_1,
        "pr_auc_positive_class_0": pr_auc_positive_class_0,
        "confusion_matrix_at_default": confusion_matrix_at_default,
        "calibration_curve": calibration_curve_result,
        "permutation_importance": permutation_importance_df,
        "error_by_corruption_type": error_by_corruption_type,
    }
