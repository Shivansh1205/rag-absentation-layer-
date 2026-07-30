"""Load Phase 2 feature parquets into a model-ready matrix.

The one transform this module owns: `entity_coverage_fraction`'s sentinel
--------------------------------------------------------------------------
`abstention_features.entity_coverage` uses `-1.0` (its
`NO_QUESTION_ENTITIES_SENTINEL`) as a *categorical* flag meaning "the
question had zero detected named entities, so coverage is undefined" --
not a continuous "worse than zero coverage" value. Feeding `-1.0` to a
gradient-boosted tree directly would let it learn a spurious ordering
(e.g. treating -1.0 as "even less covered than 0.0", when the two mean
completely different things: one is a real measurement, the other is "no
measurement was possible"). `load_features` splits that one sentinel
-carrying column into two well-behaved ones:

- `entity_coverage_fraction`: the sentinel rows get `COVERAGE_UNDEFINED_FILL_VALUE`
  (`0.0`) instead of `-1.0` -- a real, in-range value that doesn't imply
  "worse than measured zero".
- `entity_coverage_undefined`: `1` exactly where the sentinel was found,
  `0` everywhere else -- so the model can still condition on "was this
  even measurable" as its own signal, rather than that information being
  destroyed by the fill.

`FEATURE_COLUMNS` is the three feature modules' own `feature_names` (via
`abstention_features.pipeline`, not hand-typed -- same
don't-let-this-drift pattern `pipeline.py` itself uses) with
`entity_coverage_fraction` cleaned in place and `entity_coverage_undefined`
appended: 6 columns total, fixed order.

Phase 4: raw feature count changed from 6 to 5 (pipeline.py replaced
entailment.py's 3 dead features with reranker.py's 2 -- see pipeline.py's
module docstring), so `FEATURE_COLUMNS`'s total dropped from 7 to 6
accordingly. Nothing in this module's logic is hardcoded to either
number -- `RAW_FEATURE_NAMES` is read from `pipeline.feature_names`, so
this file needed no code changes for that swap, only this comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abstention_features.entity_coverage import NO_QUESTION_ENTITIES_SENTINEL
from abstention_features.pipeline import feature_names as RAW_FEATURE_NAMES

COVERAGE_COLUMN = "entity_coverage_fraction"
UNDEFINED_INDICATOR_COLUMN = "entity_coverage_undefined"

# See module docstring: a real, in-range value standing in for "no
# entities were detected in the question", paired with the indicator
# column above rather than left for the model to have to infer from -1.0.
COVERAGE_UNDEFINED_FILL_VALUE = 0.0

# Fixed model-input column order: the 5 raw feature names (imported from
# pipeline.py, not hand-typed) plus the new indicator column.
FEATURE_COLUMNS: list[str] = [*RAW_FEATURE_NAMES, UNDEFINED_INDICATOR_COLUMN]

LABEL_COLUMN = "label"
META_COLUMN = "corruption_type"

_REQUIRED_COLUMNS = (*RAW_FEATURE_NAMES, LABEL_COLUMN, META_COLUMN)


def load_features(path: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Read a Phase 2 features parquet (`train_features.parquet` /
    `eval_features.parquet`) and return `(X, y, meta)`:

    - `X`: a `pandas.DataFrame` with columns `FEATURE_COLUMNS`, in that
      fixed order, with the `entity_coverage_fraction` sentinel split out
      into `entity_coverage_undefined` -- see module docstring.
    - `y`: an `int` `numpy.ndarray` of `label` (1=answerable,
      0=unanswerable).
    - `meta`: a `pandas.Series` of `corruption_type`, same row order as
      `X`/`y` -- not a model input feature, kept for error analysis.

    Raises `ValueError` if any expected column is missing, rather than
    letting a schema drift surface later as a confusing KeyError deep in
    training.
    """
    df = pd.read_parquet(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected columns: {missing}")

    df = df.reset_index(drop=True)

    coverage = df[COVERAGE_COLUMN].to_numpy(dtype=float)
    is_undefined = coverage == NO_QUESTION_ENTITIES_SENTINEL

    X = df[list(RAW_FEATURE_NAMES)].copy()
    X.loc[is_undefined, COVERAGE_COLUMN] = COVERAGE_UNDEFINED_FILL_VALUE
    X[UNDEFINED_INDICATOR_COLUMN] = is_undefined.astype(int)
    X = X[FEATURE_COLUMNS]

    y = df[LABEL_COLUMN].to_numpy(dtype=int)
    meta = df[META_COLUMN]

    return X, y, meta
