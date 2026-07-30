"""Cross-encoder NLI scoring of (chunk, question) pairs.

**These are not logical entailment scores, and the ``entailment_prob``
naming should not be read as one.** `cross-encoder/nli-deberta-v3-xsmall`
is trained on declarative premise/hypothesis pairs (SNLI/MultiNLI); a
question is not a proposition, so there is no logically coherent sense in
which a passage "entails" an interrogative sentence. What we're actually
computing here is a semantic-overlap proxy: does this chunk's content
look, to an NLI model, like it's talking about the same thing the
question is asking about. It correlates with relevance, but it is a
heuristic borrowed from an off-distribution use of the model, not a
verified entailment judgment. Treat every ``*_entailment_prob`` feature
below with that caveat in mind -- this is the one design choice in the
whole feature set most likely to need revisiting if the classifier
underperforms.

Why this framing anyway: the textbook-correct alternative -- convert the
question into a declarative claim using the answer, then check whether
the passage entails *that* -- requires the answer, which the abstention
classifier must work without at inference time (the whole point is
predicting answerability before generation happens). Scoring
(chunk, question) directly is the tradeoff that fits that constraint.

Label order is read from the model at load time, not hardcoded
------------------------------------------------------------------
NLI checkpoints do not agree on whether index 0 is entailment,
contradiction, or neutral -- it's checkpoint-specific. Hardcoding an
index order would risk silently inverting the signal (reporting
contradiction probability as entailment) with nothing to throw an error.
`_resolve_label_indices` reads `model.config.id2label` at load time
instead.

Aggregation
-----------
For a row's `retrieved_chunks`, three features are computed:

- ``entailment_max_prob``: the highest per-chunk entailment probability.
- ``entailment_mean_prob``: the mean per-chunk entailment probability.
- ``best_margin_chunk_entailment_prob``: the entailment probability of
  the chunk with the highest (entailment - contradiction) margin -- NOT
  simply the argmax-entailment chunk (that would make this feature a
  near-duplicate of ``entailment_max_prob``). A chunk with slightly lower
  entailment but negligible contradiction is a cleaner, more confident
  signal than one with higher entailment but real ambiguity; this feature
  captures that distinction.

Empty ``retrieved_chunks`` (which can happen for some Phase 1 label=0
rows) returns a well-defined all-zero feature dict rather than raising.
"""

from __future__ import annotations

MODEL_NAME = "cross-encoder/nli-deberta-v3-xsmall"

FEATURE_NAMES = (
    "entailment_max_prob",
    "entailment_mean_prob",
    "best_margin_chunk_entailment_prob",
)

_REQUIRED_LABELS = ("entailment", "neutral", "contradiction")

_model_singleton = None


def _load_model():
    """Load the real CrossEncoder model. Not called by any fast test --
    only reached via `_get_model()` when no `model` override is supplied.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME)


def _get_model():
    """Lazy process-wide singleton: load once, reuse for every call.

    This is what makes the CLI (`scripts/extract_features.py`) not
    reload the ~70MB model on every row.
    """
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _load_model()
    return _model_singleton


def _resolve_label_indices(model) -> dict[str, int]:
    """Map {"entailment", "neutral", "contradiction"} -> output column
    index, read from `model.config.id2label` rather than assumed.

    Raises `ValueError` if any of the three required labels can't be
    found (rather than silently proceeding with a partial/wrong mapping).
    """
    id2label = model.config.id2label
    indices: dict[str, int] = {}
    for idx, label in id2label.items():
        normalized = str(label).strip().lower()
        if normalized in _REQUIRED_LABELS:
            indices[normalized] = int(idx)

    missing = set(_REQUIRED_LABELS) - set(indices)
    if missing:
        raise ValueError(
            f"Could not resolve NLI label indices from model.config.id2label="
            f"{id2label!r}; missing labels: {sorted(missing)}"
        )
    return indices


def score_chunks(
    question: str, chunks: list[str], model=None
) -> list[dict[str, float]]:
    """Score every chunk against `question` as (premise=chunk,
    hypothesis=question) pairs, in a single batched model call.

    Returns one dict per chunk (same order as `chunks`), each with keys
    "entailment", "neutral", "contradiction" summing to ~1.0. Returns an
    empty list for empty `chunks` without invoking the model at all.

    `model` is injectable (an object exposing `.predict(pairs,
    apply_softmax=True)` and `.config.id2label`, matching
    `sentence_transformers.CrossEncoder`'s interface) so tests can supply
    a fake instead of downloading the real model; defaults to the lazy
    real-model singleton.
    """
    if not chunks:
        return []

    model = model or _get_model()
    label_indices = _resolve_label_indices(model)

    pairs = [(chunk, question) for chunk in chunks]
    raw = model.predict(pairs, apply_softmax=True)

    return [
        {
            "entailment": float(row[label_indices["entailment"]]),
            "neutral": float(row[label_indices["neutral"]]),
            "contradiction": float(row[label_indices["contradiction"]]),
        }
        for row in raw
    ]


def aggregate_entailment_features(
    per_chunk_probs: list[dict[str, float]],
) -> dict[str, float]:
    """Reduce per-chunk NLI probabilities to the three row-level features
    described in the module docstring. Returns an all-zero dict for an
    empty input rather than raising (e.g. `max([])` / division by zero).
    """
    if not per_chunk_probs:
        return {name: 0.0 for name in FEATURE_NAMES}

    entailment_probs = [p["entailment"] for p in per_chunk_probs]
    margins = [p["entailment"] - p["contradiction"] for p in per_chunk_probs]
    best_margin_idx = max(range(len(margins)), key=lambda i: margins[i])

    return {
        "entailment_max_prob": max(entailment_probs),
        "entailment_mean_prob": sum(entailment_probs) / len(entailment_probs),
        "best_margin_chunk_entailment_prob": entailment_probs[best_margin_idx],
    }


def extract_entailment_features(
    question: str, retrieved_chunks: list[str], model=None
) -> dict[str, float]:
    """Score `retrieved_chunks` against `question` and aggregate to the
    three entailment features. The single entry point `pipeline.py` calls
    into from this module.
    """
    per_chunk = score_chunks(question, retrieved_chunks, model=model)
    return aggregate_entailment_features(per_chunk)
