"""Tests for `scripts/extract_features.py`'s plumbing: model loading
wiring, row processing (including the parquet list-column ->
plain-Python-list conversion), output schema, and timing/limit handling.
Exercises the real `extract_split` / `main` logic against fake models
(via the same `load_models_fn=` injection pattern
`generate_dataset.py` uses for `load_fn=`), so nothing here downloads a
real model or touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# scripts/ isn't a package -- imported by adding its directory to
# sys.path, same as test_generate_dataset_script.py does for
# generate_dataset.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import extract_features  # noqa: E402

from abstention_data.io import read_parquet, write_parquet  # noqa: E402
from abstention_features.pipeline import feature_names  # noqa: E402

SAMPLE_ROWS = [
    {
        "question": "What nationality is the director?",
        "retrieved_chunks": ["Chunk one mentions Beta.", "Chunk two is unrelated."],
        "label": 1,
        "meta": {"corruption_type": "gold_included", "source_id": "ex_1"},
    },
    {
        "question": "Which country is the musician from?",
        "retrieved_chunks": ["Distractor A.", "Distractor B."],
        "label": 0,
        "meta": {"corruption_type": "distractor_only", "source_id": "ex_2"},
    },
    {
        "question": "Empty-chunks row (Phase 1 label=0 edge case).",
        "retrieved_chunks": [],
        "label": 0,
        "meta": {"corruption_type": "distractor_only", "source_id": "ex_3"},
    },
]


class FakeReranker:
    def predict(self, pairs):
        return [0.5 for _ in pairs]


class FakeEmbedder:
    def encode(self, texts):
        # Deterministic, distinct-enough vectors per text so redundancy/
        # relevance are well-defined real numbers rather than degenerate
        # zero vectors.
        rng = np.random.RandomState(0)
        return np.array([rng.rand(4) + [hash(t) % 5, 0, 0, 0] for t in texts], dtype=float)


class FakeNLP:
    def __call__(self, text):
        # No entities ever detected -- exercises the
        # NO_QUESTION_ENTITIES_SENTINEL path for every non-empty row,
        # which is a perfectly valid (if pessimistic) fake NER stub.
        return SimpleNamespace(ents=[])


def _fake_load_models():
    return FakeReranker(), FakeEmbedder(), FakeNLP()


@pytest.fixture()
def data_dir(tmp_path):
    write_parquet(SAMPLE_ROWS, str(tmp_path / "train.parquet"))
    write_parquet(SAMPLE_ROWS[:2], str(tmp_path / "eval.parquet"))
    return tmp_path


def test_extract_split_converts_ndarray_chunks_and_handles_empty_chunks(data_dir):
    df = read_parquet(str(data_dir / "train.parquet"))
    out_df, elapsed = extract_features.extract_split(
        df,
        reranker_model=FakeReranker(),
        diversity_model=FakeEmbedder(),
        entity_coverage_nlp=FakeNLP(),
    )
    assert elapsed >= 0.0
    assert len(out_df) == 3
    assert list(out_df.columns) == list(feature_names) + ["label", "corruption_type"]

    # Row 2 (index 2) has empty retrieved_chunks -> the pipeline's fully
    # -specified all-zero sentinel dict, not a crash from treating a
    # numpy array's truthiness ambiguously.
    empty_row = out_df.iloc[2]
    for name in feature_names:
        assert empty_row[name] == 0.0
    assert int(empty_row["label"]) == 0
    assert empty_row["corruption_type"] == "distractor_only"


def test_extract_split_respects_limit(data_dir):
    df = read_parquet(str(data_dir / "train.parquet"))
    out_df, _ = extract_features.extract_split(
        df,
        reranker_model=FakeReranker(),
        diversity_model=FakeEmbedder(),
        entity_coverage_nlp=FakeNLP(),
        limit=1,
    )
    assert len(out_df) == 1


def test_main_end_to_end_without_network(data_dir):
    exit_code = extract_features.main(
        ["--data-dir", str(data_dir)],
        load_models_fn=_fake_load_models,
    )
    assert exit_code == 0

    train_out = data_dir / "train_features.parquet"
    eval_out = data_dir / "eval_features.parquet"
    assert train_out.exists()
    assert eval_out.exists()

    train_df = read_parquet(str(train_out))
    eval_df = read_parquet(str(eval_out))
    assert len(train_df) == 3
    assert len(eval_df) == 2
    assert set(train_df.columns) == set(feature_names) | {"label", "corruption_type"}


def test_main_respects_limit_flag(data_dir):
    exit_code = extract_features.main(
        ["--data-dir", str(data_dir), "--limit", "1"],
        load_models_fn=_fake_load_models,
    )
    assert exit_code == 0
    train_df = read_parquet(str(data_dir / "train_features.parquet"))
    eval_df = read_parquet(str(data_dir / "eval_features.parquet"))
    assert len(train_df) == 1
    assert len(eval_df) == 1


def test_main_writes_to_separate_out_dir_when_given(data_dir, tmp_path):
    out_dir = tmp_path / "out"
    exit_code = extract_features.main(
        ["--data-dir", str(data_dir), "--out-dir", str(out_dir)],
        load_models_fn=_fake_load_models,
    )
    assert exit_code == 0
    assert (out_dir / "train_features.parquet").exists()
    assert (out_dir / "eval_features.parquet").exists()
    assert not (data_dir / "train_features.parquet").exists()


def test_main_returns_nonzero_when_no_input_files_found(tmp_path):
    exit_code = extract_features.main(
        ["--data-dir", str(tmp_path)],
        load_models_fn=_fake_load_models,
    )
    assert exit_code == 1


def test_build_arg_parser_defaults():
    args = extract_features.build_arg_parser().parse_args([])
    assert args.data_dir == Path("data")
    assert args.out_dir is None
    assert args.limit is None
