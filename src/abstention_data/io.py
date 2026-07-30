"""Write generated dataset rows to a Hugging Face `datasets`-compatible
parquet file, and read them back.

Rows are plain dicts shaped like::

    {
        "question": str,
        "retrieved_chunks": list[str],
        "label": int,
        "meta": {"corruption_type": str, "source_id": str},
    }

pandas + pyarrow round-trip nested list and struct (dict) columns
natively, and `datasets.Dataset.from_parquet` / `load_dataset("parquet", ...)`
can read the resulting file directly with those columns intact.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("question", "retrieved_chunks", "label", "meta")


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of row dicts into a DataFrame with a fixed column
    order matching the output schema."""
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Rows are missing required columns: {missing}")
    return df[list(REQUIRED_COLUMNS)]


def write_parquet(rows: list[dict], path: str) -> None:
    """Write `rows` to a parquet file at `path`."""
    df = rows_to_dataframe(rows)
    df.to_parquet(path, engine="pyarrow", index=False)


def read_parquet(path: str) -> pd.DataFrame:
    """Read a dataset parquet file back into a DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")
