"""Tests for `abstention_model.train`.

Fast: fits on small synthetic feature matrices shaped like
`abstention_model.features.FEATURE_COLUMNS` output, no real Phase 2 data
or models needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abstention_model.features import FEATURE_COLUMNS
from abstention_model.train import RANDOM_STATE, train_classifier


def _synthetic_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {name: rng.rand(n) for name in FEATURE_COLUMNS[:-1]}
    )
    X[FEATURE_COLUMNS[-1]] = rng.randint(0, 2, n)  # entity_coverage_undefined
    # Correlate the label with one real feature so there's signal to
    # learn, rather than the model fitting pure noise.
    y = (X["centroid_question_relevance_cosine"] > 0.5).astype(int).to_numpy()
    return X[FEATURE_COLUMNS], y


def test_train_classifier_fits_and_predicts_on_held_out_rows():
    X, y = _synthetic_data(n=300, seed=1)
    X_train, y_train = X.iloc[:250], y[:250]
    X_test = X.iloc[250:]

    clf = train_classifier(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    assert preds.shape == (50,)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(clf.classes_.tolist()) == {0, 1}


def test_train_classifier_learns_the_correlated_feature_better_than_chance():
    # Not a tight accuracy bound (this is a baseline, not a tuned
    # model) -- just confirms the fitted model actually uses the signal
    # in the data rather than e.g. silently predicting one class always.
    X, y = _synthetic_data(n=400, seed=2)
    X_train, y_train = X.iloc[:300], y[:300]
    X_test, y_test = X.iloc[300:], y[300:]

    clf = train_classifier(X_train, y_train)
    accuracy = (clf.predict(X_test) == y_test).mean()
    assert accuracy > 0.7


def test_train_classifier_is_deterministic_given_same_random_state():
    X, y = _synthetic_data(n=200, seed=3)
    clf_a = train_classifier(X, y, random_state=RANDOM_STATE)
    clf_b = train_classifier(X, y, random_state=RANDOM_STATE)
    assert np.allclose(clf_a.predict_proba(X), clf_b.predict_proba(X))


def test_train_classifier_respects_feature_column_order_and_names():
    X, y = _synthetic_data(n=200, seed=4)
    clf = train_classifier(X, y)
    assert list(clf.feature_names_in_) == FEATURE_COLUMNS
