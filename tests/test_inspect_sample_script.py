"""Tests for `scripts/inspect_sample.py`'s leak-detection logic, against a
fake `load_hotpotqa` -- no network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import inspect_sample  # noqa: E402

from abstention_data.io import write_parquet  # noqa: E402
from abstention_data.loader import HotpotExample  # noqa: E402

EXAMPLE = HotpotExample(
    source_id="ex_1",
    question="Where is the answer entity from?",
    gold_passage="The gold passage clearly states the answer is Mali.",
    distractor_passages=["Some unrelated distractor passage entirely."],
    answer="Mali",
)


def _fake_load_hotpotqa(split: str, limit=None):
    return [EXAMPLE]


def test_inspect_split_passes_on_clean_data(tmp_path, capsys):
    clean_rows = [
        {
            "question": EXAMPLE.question,
            "retrieved_chunks": ["A safe distractor chunk with no leak."],
            "label": 0,
            "meta": {"corruption_type": "distractor_only", "source_id": "ex_1"},
        }
    ]
    data_dir = tmp_path
    write_parquet(clean_rows, str(data_dir / "train.parquet"))

    ok = inspect_sample.inspect_split(
        data_dir, "train", n_per_type=1, seed=0, load_fn=_fake_load_hotpotqa
    )
    assert ok is True
    out = capsys.readouterr().out
    assert "CONTAINS ANSWER" not in out


def test_inspect_split_flags_a_leaking_label_0_row(tmp_path, capsys):
    leaking_rows = [
        {
            "question": EXAMPLE.question,
            "retrieved_chunks": ["This chunk carelessly repeats Mali directly."],
            "label": 0,
            "meta": {"corruption_type": "distractor_only", "source_id": "ex_1"},
        }
    ]
    data_dir = tmp_path
    write_parquet(leaking_rows, str(data_dir / "train.parquet"))

    ok = inspect_sample.inspect_split(
        data_dir, "train", n_per_type=1, seed=0, load_fn=_fake_load_hotpotqa
    )
    assert ok is False
    out = capsys.readouterr().out
    assert "CONTAINS ANSWER" in out


def test_inspect_split_ignores_leak_in_label_1_row(tmp_path):
    # A label=1 row containing the answer is expected (that's the point
    # of gold_included) and must not be flagged as a failure.
    rows = [
        {
            "question": EXAMPLE.question,
            "retrieved_chunks": ["Gamma was born in Mali in 1978."],
            "label": 1,
            "meta": {"corruption_type": "gold_included", "source_id": "ex_1"},
        }
    ]
    data_dir = tmp_path
    write_parquet(rows, str(data_dir / "train.parquet"))

    ok = inspect_sample.inspect_split(
        data_dir, "train", n_per_type=1, seed=0, load_fn=_fake_load_hotpotqa
    )
    assert ok is True


def test_inspect_split_handles_missing_file(tmp_path):
    ok = inspect_sample.inspect_split(
        tmp_path, "eval", n_per_type=1, seed=0, load_fn=_fake_load_hotpotqa
    )
    assert ok is True
