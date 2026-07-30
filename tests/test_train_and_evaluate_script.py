"""Tests for `scripts/train_and_evaluate.py`'s plumbing: file I/O wiring
end to end against tiny synthetic parquet fixtures. No real Phase 2 data
needed -- this only checks that every promised artifact gets written
with the right shape/content, not model quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import train_and_evaluate  # noqa: E402

from abstention_features.pipeline import feature_names as RAW_FEATURE_NAMES  # noqa: E402

ALL_OUTPUT_FILENAMES = (
    train_and_evaluate.MODEL_FILENAME,
    train_and_evaluate.CALIBRATION_CURVE_PNG_FILENAME,
    train_and_evaluate.CALIBRATION_CURVE_CSV_FILENAME,
    train_and_evaluate.TRADEOFF_CURVE_PNG_FILENAME,
    train_and_evaluate.THRESHOLD_SWEEP_CSV_FILENAME,
    train_and_evaluate.FEATURE_IMPORTANCE_FILENAME,
    train_and_evaluate.ERROR_BY_CORRUPTION_TYPE_FILENAME,
    train_and_evaluate.SUMMARY_FILENAME,
)


def _make_features_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({name: rng.rand(n) for name in RAW_FEATURE_NAMES})
    # Sprinkle in a few of the entity_coverage sentinel rows, same as
    # real Phase 2 output, so load_features' transform is exercised here
    # too, not just in test_features.py.
    sentinel_idx = rng.choice(n, size=max(1, n // 20), replace=False)
    df.loc[sentinel_idx, "entity_coverage_fraction"] = -1.0

    df["label"] = (df["centroid_question_relevance_cosine"] > 0.5).astype(int)
    df["corruption_type"] = np.where(
        df["label"] == 1,
        "gold_included",
        rng.choice(["distractor_only", "truncated_span_removed"], n),
    )
    return df


@pytest.fixture()
def data_dir(tmp_path):
    _make_features_df(300, seed=1).to_parquet(tmp_path / "train_features.parquet")
    _make_features_df(120, seed=2).to_parquet(tmp_path / "eval_features.parquet")
    return tmp_path


def _run_main(data_dir, output_dir):
    return train_and_evaluate.main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--calibration-cv",
            "3",
        ]
    )


def test_main_defaults_to_artifacts_eval_under_cwd(data_dir, tmp_path, monkeypatch):
    # No --output-dir given at all -- must fall back to the documented
    # default, artifacts/eval/ relative to cwd.
    monkeypatch.chdir(tmp_path)
    exit_code = train_and_evaluate.main(["--data-dir", str(data_dir)])
    assert exit_code == 0
    default_dir = tmp_path / "artifacts" / "eval"
    assert default_dir.is_dir()
    for filename in ALL_OUTPUT_FILENAMES:
        assert (default_dir / filename).exists(), f"missing artifact: {filename}"


def test_main_writes_every_promised_artifact(data_dir, tmp_path):
    output_dir = tmp_path / "artifacts" / "eval"
    exit_code = _run_main(data_dir, output_dir)
    assert exit_code == 0

    for filename in ALL_OUTPUT_FILENAMES:
        path = output_dir / filename
        assert path.exists(), f"missing artifact: {filename}"
        assert path.stat().st_size > 0, f"empty artifact: {filename}"


def test_output_dir_is_created_if_missing(data_dir, tmp_path):
    output_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not output_dir.exists()
    exit_code = _run_main(data_dir, output_dir)
    assert exit_code == 0
    assert output_dir.is_dir()
    assert (output_dir / train_and_evaluate.MODEL_FILENAME).exists()


def test_model_joblib_round_trips_to_a_usable_classifier(data_dir, tmp_path):
    import joblib

    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)

    model = joblib.load(output_dir / train_and_evaluate.MODEL_FILENAME)
    from abstention_model.features import load_features

    X_eval, _, _ = load_features(str(data_dir / "eval_features.parquet"))
    proba = model.predict_proba(X_eval)
    assert proba.shape == (len(X_eval), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_threshold_sweep_csv_has_101_rows_and_expected_columns(data_dir, tmp_path):
    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)
    df = pd.read_csv(output_dir / train_and_evaluate.THRESHOLD_SWEEP_CSV_FILENAME)
    assert len(df) == 101
    assert list(df.columns) == [
        "threshold",
        "abstention_rate",
        "hallucination_rate_of_answered",
        "coverage_of_answerable",
        "n_answered",
        "n_answered_and_hallucinated",
        "n_answered_and_correct",
        "n_abstained_from_answerable",
    ]


def test_calibration_curve_csv_columns_and_png_are_written(data_dir, tmp_path):
    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)

    df = pd.read_csv(output_dir / train_and_evaluate.CALIBRATION_CURVE_CSV_FILENAME)
    assert list(df.columns) == ["prob_pred", "prob_true", "n_per_bin"]
    assert len(df) > 0
    assert df["n_per_bin"].sum() == 120  # the synthetic eval fixture's row count

    png_path = output_dir / train_and_evaluate.CALIBRATION_CURVE_PNG_FILENAME
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_feature_importance_csv_has_all_six_features(data_dir, tmp_path):
    # Phase 4: pipeline.py's raw feature count dropped from 6 to 5
    # (entailment.py's 3 replaced by reranker.py's 2), so the classifier's
    # final input -- 5 raw + entity_coverage_undefined -- is 6, not 7.
    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)
    df = pd.read_csv(output_dir / train_and_evaluate.FEATURE_IMPORTANCE_FILENAME)
    assert list(df.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(df) == 6


def test_error_by_corruption_type_csv_has_three_corruption_types(data_dir, tmp_path):
    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)
    df = pd.read_csv(output_dir / train_and_evaluate.ERROR_BY_CORRUPTION_TYPE_FILENAME)
    assert set(df["corruption_type"]) == {"gold_included", "distractor_only", "truncated_span_removed"}


def test_summary_json_has_expected_keys_and_matches_confusion_matrix(data_dir, tmp_path):
    output_dir = tmp_path / "artifacts" / "eval"
    _run_main(data_dir, output_dir)

    with open(output_dir / train_and_evaluate.SUMMARY_FILENAME) as f:
        summary = json.load(f)

    assert set(summary.keys()) == {
        "default_threshold",
        "roc_auc",
        "pr_auc_positive_class_1",
        "pr_auc_positive_class_0",
        "confusion_matrix",
        "calibration_signed_mean_deviation",
        "calibration_abs_mean_deviation",
        "top_permutation_importance_features",
    }
    assert summary["default_threshold"] == 0.5
    assert 0.0 <= summary["roc_auc"] <= 1.0
    assert summary["confusion_matrix"]["labels"] == [0, 1]
    assert len(summary["confusion_matrix"]["matrix"]) == 2
    assert len(summary["confusion_matrix"]["matrix"][0]) == 2
    assert len(summary["top_permutation_importance_features"]) == 4
    assert isinstance(summary["calibration_signed_mean_deviation"], float)
    assert isinstance(summary["calibration_abs_mean_deviation"], float)


def test_build_arg_parser_defaults():
    args = train_and_evaluate.build_arg_parser().parse_args([])
    assert args.data_dir == Path("data")
    assert args.output_dir == Path("artifacts/eval")
    assert args.output_dir == train_and_evaluate.DEFAULT_OUTPUT_DIR
    assert args.calibration_cv == train_and_evaluate.DEFAULT_CALIBRATION_CV
