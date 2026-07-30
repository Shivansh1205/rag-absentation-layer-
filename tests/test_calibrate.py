"""Tests for `abstention_model.calibrate`.

Fast: small synthetic feature matrices, small `cv`, no real Phase 2 data
or downloaded models needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abstention_model.calibrate import calibrate_classifier, plot_reliability_diagram
from abstention_model.features import FEATURE_COLUMNS


def _synthetic_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({name: rng.rand(n) for name in FEATURE_COLUMNS[:-1]})
    X[FEATURE_COLUMNS[-1]] = rng.randint(0, 2, n)
    y = (X["centroid_question_relevance_cosine"] > 0.5).astype(int).to_numpy()
    return X[FEATURE_COLUMNS], y


def test_calibrate_classifier_predict_proba_is_valid():
    X, y = _synthetic_data(n=300, seed=1)
    calibrated = calibrate_classifier(X, y, cv=3)

    proba = calibrated.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    assert set(calibrated.classes_.tolist()) == {0, 1}


def test_calibrate_classifier_does_not_mutate_caller_arrays():
    X, y = _synthetic_data(n=200, seed=2)
    X_before = X.copy()
    y_before = y.copy()
    calibrate_classifier(X, y, cv=3)
    pd.testing.assert_frame_equal(X, X_before)
    assert np.array_equal(y, y_before)


def test_calibrate_classifier_is_deterministic_given_same_random_state():
    X, y = _synthetic_data(n=250, seed=3)
    calibrated_a = calibrate_classifier(X, y, cv=3, random_state=7)
    calibrated_b = calibrate_classifier(X, y, cv=3, random_state=7)
    assert np.allclose(calibrated_a.predict_proba(X), calibrated_b.predict_proba(X))


def test_plot_reliability_diagram_writes_a_nonempty_png(tmp_path):
    X, y = _synthetic_data(n=300, seed=4)
    calibrated = calibrate_classifier(X, y, cv=3)
    y_prob = calibrated.predict_proba(X)[:, 1]

    out_path = tmp_path / "reliability.png"
    plot_reliability_diagram(y, y_prob, str(out_path), n_bins=5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # PNG magic bytes -- confirms this is a real image file, not an
    # empty/corrupt stub.
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
