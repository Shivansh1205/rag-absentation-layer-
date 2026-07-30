"""Tests for `abstention_model.evaluate`.

The tradeoff-curve computation (`_threshold_sweep`) is the one piece of
real logic in Phase 3 worth testing hard -- hand-computed expected values
at a few thresholds on a tiny, fully-known synthetic set, not just "runs
without crashing". `evaluate()` itself is exercised end-to-end with a
real (tiny) fitted classifier to confirm the wiring -- ROC-AUC, PR-AUC,
confusion matrix, calibration curve, permutation importance, and
error-by-corruption_type all present with the right shapes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from abstention_model.evaluate import (
    CALIBRATION_N_BINS,
    DEFAULT_THRESHOLD,
    N_THRESHOLDS,
    THRESHOLD_GRID,
    _calibration_bin_counts,
    _error_by_corruption_type,
    _threshold_sweep,
    evaluate,
)
from abstention_model.features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# _threshold_sweep -- hand-computed expected values
# ---------------------------------------------------------------------------

# p_hat / y chosen so every quantity below can be hand-counted:
#   index:   0    1    2    3    4    5
#   p_hat: 0.1  0.3  0.4  0.6  0.8  0.9
#   y:       0    1    0    1    0    1
# 6 rows total, 3 with y==1 (indices 1, 3, 5).
P_HAT = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9])
Y = np.array([0, 1, 0, 1, 0, 1])


def _row_at(df: pd.DataFrame, threshold: float) -> pd.Series:
    match = df.loc[np.isclose(df["threshold"], threshold)]
    assert len(match) == 1, f"expected exactly one grid row near {threshold}, got {len(match)}"
    return match.iloc[0]


def test_threshold_sweep_has_101_rows_matching_the_fixed_grid():
    df = _threshold_sweep(P_HAT, Y)
    assert len(df) == N_THRESHOLDS == 101
    assert np.allclose(df["threshold"].to_numpy(), THRESHOLD_GRID)


def test_threshold_sweep_hand_computed_at_t_0_5():
    # t=0.5: answered = {idx3 (0.6), idx4 (0.8), idx5 (0.9)} -> 3 rows.
    # Of those, y = [1, 0, 1] -> 1 hallucinated (idx4), 2 correct (idx3, idx5).
    # abstained = {idx0, idx1, idx2}; of those y=[0,1,0] -> 1 abstained-answerable (idx1).
    row = _row_at(_threshold_sweep(P_HAT, Y), 0.5)
    assert row["abstention_rate"] == pytest.approx(3 / 6)
    assert row["n_answered"] == 3
    assert row["n_answered_and_hallucinated"] == 1
    assert row["n_answered_and_correct"] == 2
    assert row["n_abstained_from_answerable"] == 1
    assert row["hallucination_rate_of_answered"] == pytest.approx(1 / 3)
    # Fixed denominator: 2 correctly-answered out of 3 TOTAL positives.
    assert row["coverage_of_answerable"] == pytest.approx(2 / 3)


def test_threshold_sweep_hand_computed_at_t_0_0_everything_answered():
    # t=0.0: every row's p_hat >= 0.0 -> all 6 answered.
    row = _row_at(_threshold_sweep(P_HAT, Y), 0.0)
    assert row["abstention_rate"] == pytest.approx(0.0)
    assert row["n_answered"] == 6
    assert row["n_answered_and_hallucinated"] == 3  # the 3 y==0 rows
    assert row["n_answered_and_correct"] == 3  # the 3 y==1 rows
    assert row["n_abstained_from_answerable"] == 0
    assert row["hallucination_rate_of_answered"] == pytest.approx(3 / 6)
    assert row["coverage_of_answerable"] == pytest.approx(3 / 3)


def test_threshold_sweep_hand_computed_at_t_1_0_nothing_answered_gives_nan_not_zero():
    # t=1.0: max(p_hat)=0.9 < 1.0 -> zero rows answered. This is the
    # exact "denominator is zero" case: hallucination_rate_of_answered
    # MUST be NaN (undefined), not 0.0 and not a crash.
    row = _row_at(_threshold_sweep(P_HAT, Y), 1.0)
    assert row["n_answered"] == 0
    assert row["abstention_rate"] == pytest.approx(1.0)
    assert np.isnan(row["hallucination_rate_of_answered"])
    # coverage_of_answerable's denominator (n_positive=3) is still
    # nonzero here -- 0 correctly-answered out of 3 positives is a
    # real, well-defined 0.0, NOT NaN. This is the sharpest possible
    # check that the two metrics don't share a denominator.
    assert row["coverage_of_answerable"] == pytest.approx(0.0)
    assert not np.isnan(row["coverage_of_answerable"])
    assert row["n_abstained_from_answerable"] == 3


def test_threshold_sweep_coverage_of_answerable_denominator_never_shrinks():
    # The total label==1 count (3) must appear as the implicit
    # denominator at every threshold -- i.e. n_answered_and_correct /
    # coverage_of_answerable == 3 (the fixed total) at every row where
    # coverage is defined, not some smaller answered-subset count.
    df = _threshold_sweep(P_HAT, Y)
    defined = df["coverage_of_answerable"].notna() & (df["coverage_of_answerable"] > 0)
    implied_denominator = (
        df.loc[defined, "n_answered_and_correct"] / df.loc[defined, "coverage_of_answerable"]
    )
    assert np.allclose(implied_denominator, 3.0)


def test_threshold_sweep_degenerate_all_same_label_no_positives():
    # Zero label==1 rows in the whole set -> coverage_of_answerable is
    # undefined (nothing to take coverage of) at every threshold, but
    # hallucination_rate_of_answered is still well-defined wherever
    # something is answered.
    p_hat = np.array([0.2, 0.5, 0.9])
    y = np.array([0, 0, 0])
    df = _threshold_sweep(p_hat, y)
    assert df["coverage_of_answerable"].isna().all()
    row = _row_at(df, 0.0)
    assert row["hallucination_rate_of_answered"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _calibration_bin_counts -- must align index-for-index with
# sklearn.calibration.calibration_curve's own (undocumented-count) output
# ---------------------------------------------------------------------------


def test_calibration_bin_counts_aligns_with_calibration_curve_length():
    from sklearn.calibration import calibration_curve

    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    p = rng.rand(200)

    prob_true, prob_pred = calibration_curve(y, p, n_bins=CALIBRATION_N_BINS, strategy="quantile")
    counts = _calibration_bin_counts(y, p, CALIBRATION_N_BINS)

    assert len(counts) == len(prob_true) == len(prob_pred)
    assert counts.sum() == len(y)  # every row lands in exactly one bin
    assert (counts > 0).all()  # empty bins are dropped, same as calibration_curve


# ---------------------------------------------------------------------------
# _error_by_corruption_type
# ---------------------------------------------------------------------------


def test_error_by_corruption_type_one_column_is_nan_when_group_is_single_label():
    # Mirrors this project's actual data: each corruption_type maps to
    # exactly one label value, so exactly one of the two rate columns
    # must be NaN per group -- not a bug, a documented consequence.
    y = np.array([1, 1, 0, 0, 0, 0])
    p_hat = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
    corruption_type = pd.Series(
        ["gold_included", "gold_included", "distractor_only", "distractor_only",
         "truncated_span_removed", "truncated_span_removed"]
    )

    df = _error_by_corruption_type(y, p_hat, corruption_type, threshold=0.5)
    df = df.set_index("corruption_type")

    # gold_included: both rows y==1 -> false_answer_rate undefined (no
    # y==0 rows in this group at all).
    assert np.isnan(df.loc["gold_included", "false_answer_rate"])
    # One of the two gold_included rows (p_hat=0.2 < 0.5) is a false
    # abstain; the other (p_hat=0.9) is correctly answered.
    assert df.loc["gold_included", "false_abstain_rate"] == pytest.approx(0.5)

    # distractor_only: both rows y==0 -> false_abstain_rate undefined.
    assert np.isnan(df.loc["distractor_only", "false_abstain_rate"])
    # p_hat=0.8 >= 0.5 is a false answer; p_hat=0.1 is correctly abstained.
    assert df.loc["distractor_only", "false_answer_rate"] == pytest.approx(0.5)

    # truncated_span_removed: both rows y==0 -> false_abstain_rate undefined.
    assert np.isnan(df.loc["truncated_span_removed", "false_abstain_rate"])
    assert df.loc["truncated_span_removed", "false_answer_rate"] == pytest.approx(0.5)

    assert df.loc["gold_included", "count"] == 2


# ---------------------------------------------------------------------------
# evaluate() -- end-to-end wiring, real (tiny) fitted classifier
# ---------------------------------------------------------------------------


def _synthetic_eval_set(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({name: rng.rand(n) for name in FEATURE_COLUMNS[:-1]})
    X[FEATURE_COLUMNS[-1]] = rng.randint(0, 2, n)
    X = X[FEATURE_COLUMNS]
    y = (X["centroid_question_relevance_cosine"] > 0.5).astype(int).to_numpy()
    corruption_type = pd.Series(
        np.where(y == 1, "gold_included", rng.choice(["distractor_only", "truncated_span_removed"], n))
    )
    return X, y, corruption_type


@pytest.fixture(scope="module")
def fitted_model():
    X, y, _ = _synthetic_eval_set(n=300, seed=1)
    model = LogisticRegression().fit(X, y)
    return model


def test_evaluate_returns_all_expected_keys(fitted_model):
    X, y, corruption_type = _synthetic_eval_set(n=200, seed=2)
    result = evaluate(fitted_model, X, y, corruption_type=corruption_type)

    assert set(result.keys()) == {
        "threshold_sweep",
        "roc_auc",
        "pr_auc_positive_class_1",
        "pr_auc_positive_class_0",
        "confusion_matrix_at_default",
        "calibration_curve",
        "permutation_importance",
        "error_by_corruption_type",
    }


def test_evaluate_threshold_sweep_shape_and_columns(fitted_model):
    X, y, _ = _synthetic_eval_set(n=200, seed=3)
    result = evaluate(fitted_model, X, y)
    sweep = result["threshold_sweep"]
    assert len(sweep) == 101
    assert list(sweep.columns) == [
        "threshold",
        "abstention_rate",
        "hallucination_rate_of_answered",
        "coverage_of_answerable",
        "n_answered",
        "n_answered_and_hallucinated",
        "n_answered_and_correct",
        "n_abstained_from_answerable",
    ]


def test_evaluate_roc_and_pr_auc_are_valid_floats(fitted_model):
    X, y, _ = _synthetic_eval_set(n=200, seed=4)
    result = evaluate(fitted_model, X, y)
    for key in ("roc_auc", "pr_auc_positive_class_1", "pr_auc_positive_class_0"):
        assert isinstance(result[key], float)
        assert 0.0 <= result[key] <= 1.0


def test_evaluate_confusion_matrix_has_expected_shape_and_note(fitted_model):
    X, y, _ = _synthetic_eval_set(n=200, seed=5)
    result = evaluate(fitted_model, X, y, default_threshold=0.5)
    cm = result["confusion_matrix_at_default"]
    assert cm["matrix"].shape == (2, 2)
    assert cm["labels"] == [0, 1]
    assert cm["threshold"] == 0.5
    assert "not a claimed optimal" in cm["note"]
    assert cm["matrix"].sum() == 200


def test_evaluate_calibration_curve_arrays_are_aligned(fitted_model):
    X, y, _ = _synthetic_eval_set(n=200, seed=6)
    result = evaluate(fitted_model, X, y)
    cal = result["calibration_curve"]
    assert len(cal["prob_true"]) == len(cal["prob_pred"]) == len(cal["n_per_bin"])
    assert cal["n_per_bin"].sum() == 200


def test_evaluate_permutation_importance_sorted_descending_with_correct_columns(fitted_model):
    X, y, _ = _synthetic_eval_set(n=200, seed=7)
    result = evaluate(fitted_model, X, y)
    importances = result["permutation_importance"]
    assert list(importances.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(importances["feature"]) == set(FEATURE_COLUMNS)
    means = importances["importance_mean"].to_numpy()
    assert np.all(means[:-1] >= means[1:])  # sorted descending


def test_evaluate_error_by_corruption_type_none_when_not_provided(fitted_model):
    X, y, _ = _synthetic_eval_set(n=100, seed=8)
    result = evaluate(fitted_model, X, y, corruption_type=None)
    assert result["error_by_corruption_type"] is None


def test_evaluate_error_by_corruption_type_present_when_provided(fitted_model):
    X, y, corruption_type = _synthetic_eval_set(n=100, seed=9)
    result = evaluate(fitted_model, X, y, corruption_type=corruption_type)
    df = result["error_by_corruption_type"]
    assert list(df.columns) == ["corruption_type", "count", "false_abstain_rate", "false_answer_rate"]
    assert df["count"].sum() == 100
