"""Tests for `abstention_data.io`: parquet round-tripping of the output
schema, including the nested list[str] and dict (meta) columns.
"""

from __future__ import annotations

import pytest

from abstention_data.io import read_parquet, rows_to_dataframe, write_parquet

SAMPLE_ROWS = [
    {
        "question": "What nationality is the director?",
        "retrieved_chunks": ["Chunk one.", "Chunk two.", "Chunk three."],
        "label": 1,
        "meta": {"corruption_type": "gold_included", "source_id": "ex_1"},
    },
    {
        "question": "Which country is the musician from?",
        "retrieved_chunks": ["Distractor A.", "Distractor B."],
        "label": 0,
        "meta": {"corruption_type": "distractor_only", "source_id": "ex_2"},
    },
]


def test_rows_to_dataframe_has_expected_columns_and_order():
    df = rows_to_dataframe(SAMPLE_ROWS)
    assert list(df.columns) == ["question", "retrieved_chunks", "label", "meta"]
    assert len(df) == 2


def test_rows_to_dataframe_raises_on_missing_column():
    broken_rows = [{"question": "Q", "label": 1, "meta": {}}]
    with pytest.raises(ValueError, match="missing required columns"):
        rows_to_dataframe(broken_rows)


def test_write_and_read_parquet_roundtrip(tmp_path):
    path = str(tmp_path / "dataset.parquet")
    write_parquet(SAMPLE_ROWS, path)

    df = read_parquet(path)
    assert len(df) == 2
    assert list(df.columns) == ["question", "retrieved_chunks", "label", "meta"]

    row0 = df.iloc[0]
    assert row0["question"] == SAMPLE_ROWS[0]["question"]
    assert list(row0["retrieved_chunks"]) == SAMPLE_ROWS[0]["retrieved_chunks"]
    assert int(row0["label"]) == 1
    assert dict(row0["meta"]) == SAMPLE_ROWS[0]["meta"]

    row1 = df.iloc[1]
    assert list(row1["retrieved_chunks"]) == SAMPLE_ROWS[1]["retrieved_chunks"]
    assert dict(row1["meta"]) == SAMPLE_ROWS[1]["meta"]


def test_roundtrip_preserves_row_count_for_larger_batch(tmp_path):
    rows = [
        {
            "question": f"Question {i}?",
            "retrieved_chunks": [f"Chunk {i}-{j}" for j in range(3)],
            "label": i % 2,
            "meta": {"corruption_type": "gold_included", "source_id": f"ex_{i}"},
        }
        for i in range(50)
    ]
    path = str(tmp_path / "bigger.parquet")
    write_parquet(rows, path)
    df = read_parquet(path)
    assert len(df) == 50
    assert df["label"].sum() == sum(r["label"] for r in rows)
