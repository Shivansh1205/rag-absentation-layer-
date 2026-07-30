"""Tests for `scripts/generate_dataset.py`'s plumbing: argument parsing,
config wiring, and file output. Exercises the real `generate_split` /
`main` logic against a fake `load_hotpotqa` so nothing here touches the
network -- consistent with keeping the whole suite network-free.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package (it's a plain CLI entry point, not part of the
# installed `abstention_data` distribution), so it's imported by adding
# its directory to sys.path rather than via a package-relative import.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import generate_dataset  # noqa: E402

from abstention_data.config import GenerationConfig  # noqa: E402
from abstention_data.io import read_parquet  # noqa: E402
from abstention_data.loader import _examples_from_raw  # noqa: E402
from tests.fixtures import SIMPLE_RAW_EXAMPLE  # noqa: E402

_BASE_EXAMPLE = next(iter(_examples_from_raw([SIMPLE_RAW_EXAMPLE])))


def _fake_load_hotpotqa(split: str, limit=None):
    """Stand-in for the real HF-Hub-backed loader: returns `limit` distinct
    usable examples regardless of `split`, with no network access."""
    n = limit or 10
    return [
        dataclasses.replace(_BASE_EXAMPLE, source_id=f"{_BASE_EXAMPLE.source_id}_{split}_{i}")
        for i in range(n)
    ]


def test_generate_split_produces_requested_rows_without_network():
    config = GenerationConfig(k=4, distractor_ratio=0.5, seed=1)
    rows, stats = generate_dataset.generate_split(
        hf_split="train",
        n_rows=5,
        config=config,
        pool_multiplier=3,
        min_pool=10,
        load_fn=_fake_load_hotpotqa,
    )
    assert len(rows) == 5
    assert stats.n_produced == 5


def test_build_arg_parser_defaults_match_spec():
    args = generate_dataset.build_arg_parser().parse_args([])
    assert args.n_train == 5000
    assert args.n_eval == 1000
    assert args.seed == 42


def test_main_end_to_end_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_dataset, "load_hotpotqa", _fake_load_hotpotqa)

    exit_code = generate_dataset.main(
        [
            "--n-train",
            "5",
            "--n-eval",
            "3",
            "--seed",
            "1",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "eval.parquet").exists()

    train_df = read_parquet(str(tmp_path / "train.parquet"))
    eval_df = read_parquet(str(tmp_path / "eval.parquet"))
    assert len(train_df) == 5
    assert len(eval_df) == 3
    assert set(train_df.columns) == {"question", "retrieved_chunks", "label", "meta"}


def test_main_is_reproducible_given_same_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_dataset, "load_hotpotqa", _fake_load_hotpotqa)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    for out_dir in (out_a, out_b):
        generate_dataset.main(
            ["--n-train", "5", "--n-eval", "3", "--seed", "99", "--out-dir", str(out_dir)]
        )

    df_a = read_parquet(str(out_a / "train.parquet"))
    df_b = read_parquet(str(out_b / "train.parquet"))
    assert df_a["question"].tolist() == df_b["question"].tolist()
    assert df_a["label"].tolist() == df_b["label"].tolist()
