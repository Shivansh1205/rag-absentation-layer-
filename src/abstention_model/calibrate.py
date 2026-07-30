"""Probability calibration: the layer the abstention threshold actually
sits on top of.

Calibration strategy -- documented, since this is a real design choice
--------------------------------------------------------------------------
`calibrate_classifier` does **not** reuse `train.py`'s already-fitted
`HistGradientBoostingClassifier` instance. Calibrating a model's
`predict_proba` output against rows it was already fit on measures how
well it fits data it has memorized, not real-world calibration -- that
would be leakage, and would make the reliability diagram and every
downstream abstention-threshold decision look better than it actually
is.

Two ways to avoid that: hand-carve a separate calibration split out of
train, or let `sklearn.calibration.CalibratedClassifierCV` do its own
internal k-fold split (fit a clone of the base estimator on k-1 folds,
calibrate it on the held-out fold, repeat per fold, then average the k
calibrated sub-models' predictions at inference time). This module uses
the latter: an *unfitted* base estimator wrapped in
`CalibratedClassifierCV(cv=5)`. That's the sklearn-recommended, leak-free
default -- no extra bookkeeping of a hand-carved holdout, and it uses all
of the training rows for both fitting and calibration (just never the
same row for both, within any given fold).

`method="isotonic"`: with ~5000 train rows, isotonic regression's
nonparametric fit is the safer choice here over Platt/sigmoid scaling.
Sigmoid calibration assumes the miscalibration has a specific S-shape,
which is a bad structural assumption given the feature distributions
behind this classifier (entailment features sparse/right-skewed, entity
coverage bimodal) -- isotonic makes no shape assumption and has enough
data here to avoid overfitting the calibration map itself.

Relationship to `train.py`: `train.py`'s separately-fitted classifier
remains useful on its own (a single fast fit, used for feature
importance and any diagnostic that doesn't need calibrated
probabilities). The `CalibratedClassifierCV` object this module produces
is the one whose `predict_proba` output the abstention threshold and
tradeoff curve (`evaluate.py`) are actually computed from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier

from abstention_model.train import CLASS_WEIGHT, RANDOM_STATE

CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV_FOLDS = 5


def calibrate_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
    method: str = CALIBRATION_METHOD,
    cv: int = CALIBRATION_CV_FOLDS,
) -> CalibratedClassifierCV:
    """Fit a calibrated classifier from scratch on `(X, y)`.

    Builds a fresh, unfitted `HistGradientBoostingClassifier` (same
    hyperparameters as `train.py`'s) and wraps it in
    `CalibratedClassifierCV` -- see module docstring for why this doesn't
    reuse `train.py`'s already-fitted model. Returns the fitted
    `CalibratedClassifierCV`; its `.predict_proba(X)[:, 1]` is the
    calibrated P(label=1) an abstention threshold should be applied to.
    """
    base = HistGradientBoostingClassifier(random_state=random_state, class_weight=CLASS_WEIGHT)
    calibrated = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: str,
    *,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram (mean predicted probability vs.
    observed label=1 frequency per bin, plus the perfect-calibration
    diagonal) to `path` as a PNG.

    `strategy="quantile"` bins by equal *count* rather than equal probability
    -width -- with this classifier's skewed predicted-probability
    distribution, equal-width bins would leave several bins empty and
    a few overcrowded; quantile binning keeps every bin meaningfully
    populated.

    Uses matplotlib's non-interactive `Agg` backend explicitly (set right
    before importing `pyplot`) rather than relying on whatever backend
    would otherwise be auto-selected, so this works headlessly in CLI
    scripts and tests without a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.set_xlabel("Mean predicted probability (calibrated)")
    ax.set_ylabel("Observed frequency of label=1")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
