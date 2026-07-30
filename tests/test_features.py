"""Tests for `abstention_model.features`.

Fast, no models -- pure pandas/numpy over a hand-built tiny parquet
fixture, asserting exact transformed values (not just "runs without
crashing"), since a subtle bug here (e.g. flagging a genuine 0.0 coverage
as "undefined", or vice versa) would silently corrupt every downstream
Phase 3 module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from abstention_features.entity_coverage import NO_QUESTION_ENTITIES_SENTINEL
from abstention_features.pipeline import feature_names as RAW_FEATURE_NAMES
from abstention_model.features import (
    COVERAGE_UNDEFINED_FILL_VALUE,
    FEATURE_COLUMNS,
    UNDEFINED_INDICATOR_COLUMN,
    load_features,
)

ROWS = [
    {
        "reranker_max_score": 7.2,
        "reranker_mean_score": 2.1,
        "chunk_redundancy_mean_cosine": 0.3,
        "centroid_question_relevance_cosine": 0.6,
        "entity_coverage_fraction": 0.75,  # normal, non-sentinel value
        "label": 1,
        "corruption_type": "gold_included",
    },
    {
        "reranker_max_score": -5.4,
        "reranker_mean_score": -6.1,
        "chunk_redundancy_mean_cosine": 0.9,
        "centroid_question_relevance_cosine": 0.85,
        "entity_coverage_fraction": NO_QUESTION_ENTITIES_SENTINEL,  # -1.0 sentinel
        "label": 0,
        "corruption_type": "distractor_only",
    },
    {
        "reranker_max_score": 0.3,
        "reranker_mean_score": -0.1,
        "chunk_redundancy_mean_cosine": 0.0,
        "centroid_question_relevance_cosine": 0.1,
        # Genuine zero coverage (entities existed in the question, none
        # matched in the chunks) -- must NOT be treated as the sentinel.
        "entity_coverage_fraction": 0.0,
        "label": 0,
        "corruption_type": "truncated_span_removed",
    },
]


def test_feature_columns_is_raw_names_plus_indicator():
    assert FEATURE_COLUMNS == [*RAW_FEATURE_NAMES, UNDEFINED_INDICATOR_COLUMN]
    assert len(FEATURE_COLUMNS) == 6
    assert FEATURE_COLUMNS[-1] == "entity_coverage_undefined"


def test_load_features_sentinel_transform_is_exact(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame(ROWS).to_parquet(path)

    X, y, meta = load_features(str(path))

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 3

    # Row 0: normal coverage -> indicator 0, value unchanged exactly.
    assert X.loc[0, "entity_coverage_fraction"] == pytest.approx(0.75)
    assert X.loc[0, UNDEFINED_INDICATOR_COLUMN] == 0

    # Row 1: sentinel -> indicator 1, value replaced with the neutral fill
    # (not left at -1.0).
    assert X.loc[1, "entity_coverage_fraction"] == pytest.approx(COVERAGE_UNDEFINED_FILL_VALUE)
    assert X.loc[1, "entity_coverage_fraction"] != NO_QUESTION_ENTITIES_SENTINEL
    assert X.loc[1, UNDEFINED_INDICATOR_COLUMN] == 1

    # Row 2: genuine 0.0 coverage is a real measurement, not the
    # sentinel -- must not be flagged as undefined.
    assert X.loc[2, "entity_coverage_fraction"] == pytest.approx(0.0)
    assert X.loc[2, UNDEFINED_INDICATOR_COLUMN] == 0

    # Other 4 feature columns pass through completely unchanged -- including
    # reranker scores' negative values (raw logits, not probabilities;
    # there's nothing here that should clip or floor them at 0).
    assert X.loc[0, "reranker_max_score"] == pytest.approx(7.2)
    assert X.loc[1, "reranker_mean_score"] == pytest.approx(-6.1)
    assert X.loc[1, "chunk_redundancy_mean_cosine"] == pytest.approx(0.9)
    assert X.loc[2, "centroid_question_relevance_cosine"] == pytest.approx(0.1)

    assert list(y) == [1, 0, 0]
    assert y.dtype.kind == "i"
    assert list(meta) == ["gold_included", "distractor_only", "truncated_span_removed"]


def test_load_features_raises_on_missing_columns(tmp_path):
    path = tmp_path / "broken.parquet"
    pd.DataFrame([{"entailment_max_prob": 0.1, "label": 1}]).to_parquet(path)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_features(str(path))


def test_load_features_x_y_meta_row_counts_match(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame(ROWS).to_parquet(path)
    X, y, meta = load_features(str(path))
    assert len(X) == len(y) == len(meta) == 3
