"""Baseline gradient-boosted classifier for the RAG abstention task.

Why `HistGradientBoostingClassifier`, not logistic regression
------------------------------------------------------------------
The 6 underlying features (see `abstention_features.pipeline`) have very
different shapes: the three entailment features are sparse/right-skewed
(most mass near 0, a long tail), `entity_coverage_fraction` is
effectively bimodal even before Phase 3's sentinel-splitting (clustered
near 0 and near 1), and the two diversity features are well-behaved
continuous cosine similarities. A linear model would need hand-built
interaction/transform terms to handle that mix reasonably; a
tree-based gradient booster handles skew, non-linear thresholds, and
feature interactions natively, which is the whole reason this project
uses one instead of logistic regression.

Class imbalance: weighting, not resampling
---------------------------------------------
The train split is ~59/41 (label=0 majority, unanswerable). That's mild
enough that `class_weight="balanced"` (inverse-frequency reweighting
inside the loss, no synthetic/duplicated rows) is sufficient --
resampling would add its own artifacts (SMOTE-style synthetic rows don't
make sense on these particular feature semantics; naive
duplication/undersampling would waste or distort the real training
signal) for a class split this close to even.

Deliberately not tuned yet
------------------------------
Hyperparameters are otherwise left at sklearn's defaults, with only
`random_state` fixed for reproducibility. This is meant to be a working,
calibratable baseline first; hyperparameter search is an explicit later
step, not folded in here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"


def train_classifier(
    X: pd.DataFrame, y: np.ndarray, *, random_state: int = RANDOM_STATE
) -> HistGradientBoostingClassifier:
    """Fit a `HistGradientBoostingClassifier` on `(X, y)`.

    `class_weight="balanced"` and `random_state` are the only
    non-default knobs set -- see module docstring. Returns the fitted
    estimator; the caller (`scripts/train_and_evaluate.py`) is
    responsible for persisting it if needed.
    """
    clf = HistGradientBoostingClassifier(random_state=random_state, class_weight=CLASS_WEIGHT)
    clf.fit(X, y)
    return clf
