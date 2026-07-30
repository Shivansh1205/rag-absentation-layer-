"""Single entry point: merge all three feature groups into one flat,
fixed-order feature vector per `(question, retrieved_chunks)` row.

`extract_features` calls `reranker.extract_reranker_features`,
`diversity.extract_diversity_features`, and
`entity_coverage.extract_entity_coverage_features`, and merges their
three dicts into one. This module owns none of the actual feature logic
-- it is orchestration over the three already-tested modules -- but it
does own two things those modules deliberately don't:

Phase 4: entailment.py replaced by reranker.py here
-----------------------------------------------------
Phase 4's clean-subset AUC diagnostic found the prior feature set
(entailment.py's NLI proxy + diversity.py's embedding cosine
similarities) could not distinguish `gold_included` from
`truncated_span_removed` rows (AUC 0.5495, ~chance) -- both are
topically-identical passages, one with the answer sentence intact and
one with it removed, and every surviving feature at the time measured
topical relatedness, not answer presence. `reranker.py`
(`cross-encoder/ms-marco-MiniLM-L6-v2`, trained on query-passage
relevance rather than NLI) is the direct fix. `entailment.py` itself is
**not deleted** -- it's still a real, tested module in the codebase --
this module simply stopped calling it. See `reranker.py`'s module
docstring for the full diagnosis writeup.

`feature_names`: the fixed 5-key order
--------------------------------------
Built by concatenating each module's own `FEATURE_NAMES`, not
hand-typed, so this list can never silently drift from what the three
modules actually produce:

    reranker.FEATURE_NAMES        -> reranker_max_score,
                                      reranker_mean_score
    diversity.FEATURE_NAMES       -> chunk_redundancy_mean_cosine,
                                      centroid_question_relevance_cosine
    entity_coverage.FEATURE_NAMES -> entity_coverage_fraction

This is the order training/eval matrix columns must be built in.
(`abstention_model.features.load_features` appends one more derived
column, `entity_coverage_undefined`, downstream of this module -- so the
classifier's final input is 6 columns, not 5; see that module's
docstring.)

The empty-`retrieved_chunks` case is this module's own, explicit
contract -- not delegated composition
------------------------------------------------------------------------
As it happens, all three modules already agree on what an empty-chunks
row should produce: reranker and diversity each return their features
as `0.0`, and entity_coverage's empty-chunks short-circuit (checked
*before* it ever looks at the question) also returns `0.0` rather than
its own `NO_QUESTION_ENTITIES_SENTINEL` -- so calling all three and
merging would, today, happen to produce the same 5-key all-zero dict as
`EMPTY_CHUNKS_FEATURES` below. That agreement is *not* what this module
relies on, though: `extract_features` short-circuits on empty
`retrieved_chunks` and returns `EMPTY_CHUNKS_FEATURES` directly, without
calling any of the three modules at all. Three independently-maintained
modules happening to agree on a convention today is not a guarantee they
keep agreeing after an unrelated change to one of them -- pipeline.py's
contract for this case is the explicit dict below, checked by
`test_pipeline.py`, not an emergent property of composition.

Sentinels from non-empty rows are passed through untouched
------------------------------------------------------------------------
This short-circuit only covers *zero retrieved_chunks*. For any
non-empty `retrieved_chunks`, `extract_features` merges the three
modules' raw output dicts verbatim -- no clamping, no rounding, no
"clean up the weird values" pass. That matters because two of the three
modules define their own *internal* sentinels for specific undefined
cases even when chunks are non-empty:

- `entity_coverage.NO_QUESTION_ENTITIES_SENTINEL` (`-1.0`): the question
  itself has zero detected entities.
- `diversity`'s `chunk_redundancy_mean_cosine` sentinel (`0.0`): exactly
  one chunk, so no pair exists.

Both are informative "this feature is undefined for this row" signals
the downstream classifier should see as-is, not values to be silently
zeroed or imputed away at this layer.

Dependency injection for batch use
------------------------------------
`extract_features` accepts an optional pre-loaded model/loader object
for each of the three modules (`reranker_model`, `diversity_model`,
`entity_coverage_nlp` -- named distinctly, and keyword-only, because two
of the three modules both call their own parameter `model` and this
function needs to route to the right one unambiguously). Each module
already lazily caches its own real-model singleton on first use within a
process, so injection here isn't strictly required for correctness --
but it lets a batch script (`scripts/extract_features.py`) load each
model exactly once, hand the same three objects to every row's call
explicitly, and lets tests inject fakes for all three without touching
any module's internal singleton state (which would otherwise leak
between tests).
"""

from __future__ import annotations

from abstention_features.diversity import (
    FEATURE_NAMES as _DIVERSITY_FEATURE_NAMES,
    extract_diversity_features,
)
from abstention_features.entity_coverage import (
    FEATURE_NAMES as _ENTITY_COVERAGE_FEATURE_NAMES,
    extract_entity_coverage_features,
)
from abstention_features.reranker import (
    FEATURE_NAMES as _RERANKER_FEATURE_NAMES,
    extract_reranker_features,
)

# Fixed order: reranker's 2, then diversity's 2, then entity_coverage's 1
# -- concatenated from each module's own FEATURE_NAMES rather than
# hand-typed, so this can never silently drift out of sync with what the
# three modules actually return.
feature_names: list[str] = [
    *_RERANKER_FEATURE_NAMES,
    *_DIVERSITY_FEATURE_NAMES,
    *_ENTITY_COVERAGE_FEATURE_NAMES,
]

# This module's own, explicit contract for empty `retrieved_chunks` --
# see module docstring for why this is a fixed dict `extract_features`
# returns directly, not an emergent result of calling all three modules.
EMPTY_CHUNKS_FEATURES: dict[str, float] = {name: 0.0 for name in feature_names}


def extract_features(
    question: str,
    retrieved_chunks: list[str],
    *,
    reranker_model=None,
    diversity_model=None,
    entity_coverage_nlp=None,
) -> dict[str, float]:
    """Compute all 5 Phase 2/4 features for one `(question,
    retrieved_chunks)` row and return them as a single flat dict keyed by
    `feature_names`.

    Empty `retrieved_chunks` returns `EMPTY_CHUNKS_FEATURES` (a fresh
    copy) directly, without calling any of the three feature modules --
    see module docstring. For non-empty `retrieved_chunks`, this is a
    thin merge of the three modules' own outputs; any sentinel one of
    them produces internally (e.g. entity_coverage's "question has no
    entities" `-1.0`, or diversity's "only one chunk" redundancy `0.0`)
    passes through unchanged.

    `reranker_model` / `diversity_model` / `entity_coverage_nlp` are
    optional pre-loaded model objects, forwarded to the matching
    module's `extract_*_features` call (as `model=` for the first two,
    `nlp=` for the third) -- see module docstring for why batch callers
    should pass these in rather than relying on each module's default
    lazy singleton.
    """
    if not retrieved_chunks:
        return dict(EMPTY_CHUNKS_FEATURES)

    features: dict[str, float] = {}
    features.update(
        extract_reranker_features(question, retrieved_chunks, model=reranker_model)
    )
    features.update(
        extract_diversity_features(question, retrieved_chunks, model=diversity_model)
    )
    features.update(
        extract_entity_coverage_features(question, retrieved_chunks, nlp=entity_coverage_nlp)
    )
    return features
