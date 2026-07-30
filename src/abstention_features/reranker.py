"""Cross-encoder reranker scoring of (question, chunk) pairs.

Phase 4 diagnosis found the ceiling on the abstention classifier's ROC-AUC
(~0.6735) was a **feature** problem, not a label problem: clean-subset AUC
on `gold_included` vs `distractor_only` (topically different passages) was
0.7854, but on `gold_included` vs `truncated_span_removed` (the *same*
passage with only the answer sentence removed) it was 0.5495 -- barely
above chance. The existing feature set (`entailment.py`'s NLI proxy,
`diversity.py`'s embedding cosine similarities) all measure topical
relatedness between the question and the retrieved chunks. Two passages
about the same film, one with the answer sentence intact and one with it
cut, are *topically identical* -- so every one of those features scores
them almost the same, and the classifier has no way to tell them apart.

This module replaces that framing with `cross-encoder/ms-marco-MiniLM-L6-v2`,
a reranker trained directly on "does this passage answer this query"
(MS MARCO passage ranking) -- not semantic similarity, and not NLI. It is
the direct fix for the failure mode above: a reranker scores answer
*presence*, which is exactly the thing topical-relatedness features can't
see.

Not NLI, no softmax
--------------------
`entailment.py` calls `model.predict(pairs, apply_softmax=True)` because
its NLI checkpoint has a 3-class head; the raw model output there is
uninterpretable without normalizing across classes. MS MARCO cross-encoder
rerankers have a **single output neuron** -- there is nothing to
softmax against, and no label-index resolution needed (unlike
`entailment.py`'s `_resolve_label_indices`, which exists specifically to
handle a checkpoint-dependent multi-class ordering). `model.predict(pairs)`
is called with no `apply_softmax` argument, returning one raw score per
pair.

That raw score is **not a 0-1 probability** -- `cross-encoder/ms-marco-MiniLM-L6-v2`
outputs an unbounded real-valued relevance logit (empirically, for this
checkpoint, roughly in the [-11, 11] range: strongly negative for clearly
irrelevant pairs, strongly positive for clearly relevant ones, with no
enforced upper/lower bound). `test_reranker.py`'s slow real-model test
checks and documents the actual observed range rather than assuming one.
Downstream (the classifier), this is fine -- tree-based models don't need
inputs pre-scaled to any particular range.

Query-passage order matters here, unlike entailment.py's (chunk, question)
-------------------------------------------------------------------------
`entailment.py` pairs `(chunk, question)` -- chunk-as-premise,
question-as-hypothesis, because it's (mis)using an NLI model. MS MARCO
reranker checkpoints are trained on `(query, passage)` pairs specifically,
matching how MS MARCO's own training data is structured -- so this module
pairs `(question, chunk)`, question first. Swapping the order would still
run without error but would silently feed the model an off-distribution
input.

Aggregation
-----------
For a row's `retrieved_chunks`, two features are computed:

- ``reranker_max_score``: the highest per-chunk relevance score.
- ``reranker_mean_score``: the mean per-chunk relevance score.

There is no `best_margin_*`-style third feature here the way
`entailment.py` has one: that feature exists there specifically to
distinguish "highest entailment" from "highest (entailment -
contradiction) margin" in a 3-class output, and a single-scalar output has
no such second axis to disambiguate -- `reranker_max_score` already *is*
the best-chunk score, so adding a same-valued third feature under a
different name would be redundant, not additive.

Empty ``retrieved_chunks`` returns a well-defined all-zero feature dict
(same sentinel convention as `diversity.py`) rather than raising.
"""

from __future__ import annotations

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

FEATURE_NAMES = (
    "reranker_max_score",
    "reranker_mean_score",
)

_model_singleton = None


def _load_model():
    """Load the real CrossEncoder model. Not called by any fast test --
    only reached via `_get_model()` when no `model` override is supplied.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME)


def _get_model():
    """Lazy process-wide singleton: load once, reuse for every call.

    Same pattern as `entailment.py` / `diversity.py` -- this is what keeps
    the CLI (`scripts/extract_features.py`) from reloading the model on
    every row.
    """
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _load_model()
    return _model_singleton


def score_chunks(question: str, chunks: list[str], model=None) -> list[float]:
    """Score every chunk against `question` as `(question, chunk)` pairs,
    in a single batched model call.

    Returns one raw relevance score per chunk (same order as `chunks`,
    see module docstring for why this is an unbounded logit, not a
    probability). Returns an empty list for empty `chunks` without
    invoking the model at all.

    `model` is injectable (an object exposing `.predict(pairs) ->
    array-like of scalars`, matching `sentence_transformers.CrossEncoder`'s
    interface for a single-output-neuron checkpoint) so tests can supply a
    fake instead of downloading the real model; defaults to the lazy
    real-model singleton.
    """
    if not chunks:
        return []

    model = model or _get_model()
    pairs = [(question, chunk) for chunk in chunks]
    raw = model.predict(pairs)

    return [float(score) for score in raw]


def aggregate_reranker_features(per_chunk_scores: list[float]) -> dict[str, float]:
    """Reduce per-chunk reranker scores to the two row-level features
    described in the module docstring. Returns an all-zero dict for an
    empty input rather than raising (e.g. `max([])` / division by zero) --
    same sentinel convention as `diversity.py`.
    """
    if not per_chunk_scores:
        return {name: 0.0 for name in FEATURE_NAMES}

    return {
        "reranker_max_score": max(per_chunk_scores),
        "reranker_mean_score": sum(per_chunk_scores) / len(per_chunk_scores),
    }


def extract_reranker_features(
    question: str, retrieved_chunks: list[str], model=None
) -> dict[str, float]:
    """Score `retrieved_chunks` against `question` and aggregate to the
    two reranker features. The single entry point `pipeline.py` calls
    into from this module.
    """
    per_chunk = score_chunks(question, retrieved_chunks, model=model)
    return aggregate_reranker_features(per_chunk)
