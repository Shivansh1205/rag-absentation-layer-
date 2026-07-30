"""Embedding-based redundancy and relevance features.

Uses `sentence-transformers/all-MiniLM-L6-v2` (CPU) to embed the question
and every retrieved chunk, then computes two features:

- ``chunk_redundancy_mean_cosine``: mean pairwise cosine similarity
  across all chunks. High = retrieval wasted slots on near-duplicate
  content instead of diverse coverage.
- ``centroid_question_relevance_cosine``: cosine similarity between the
  chunk-embedding centroid and the question embedding. Does the
  *overall* retrieved set relate to the question topically, independent
  of whether any single chunk logically entails an answer -- this is
  meant as a broader, coarser-grained relevance signal than
  `entailment.py`'s per-chunk NLI scores, and in particular may catch
  topical/associative links (e.g. a place name implying a nationality)
  that an NLI model trained on declarative entailment won't reliably
  pick up.

Sentinel handling (read before using these features downstream)
-----------------------------------------------------------------
Pairwise similarity is undefined for fewer than 2 chunks, and there's
nothing to embed at all for zero chunks. Rather than one blanket
"empty -> all zero" rule, the two features are handled independently
based on what's actually well-defined:

- **Zero chunks**: both features return the sentinel ``0.0``. Nothing
  can be embedded, so nothing can be computed.
- **Exactly one chunk**: ``chunk_redundancy_mean_cosine`` returns the
  sentinel ``0.0`` (no pair exists), but ``centroid_question_relevance_cosine``
  is still computed normally -- the centroid of one embedding is just
  that embedding, so the single chunk's similarity to the question is a
  perfectly well-defined, real (non-sentinel) value. Collapsing this
  case to all-zero would silently throw away real signal for exactly
  the single-chunk retrievals where relevance matters most.
- **Two or more chunks**: both features are computed normally.

Same lazy-singleton + swappable-loader pattern as `entailment.py`
-------------------------------------------------------------------
`model` is always an optional, injectable parameter (an object exposing
`.encode(texts) -> array-like of shape (len(texts), dim)`, matching
`sentence_transformers.SentenceTransformer`'s interface), defaulting to a
lazy process-wide singleton. Fast tests inject a fake embedder; only
`@pytest.mark.slow` tests touch the real model.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FEATURE_NAMES = (
    "chunk_redundancy_mean_cosine",
    "centroid_question_relevance_cosine",
)

_model_singleton = None


def _load_model():
    """Load the real SentenceTransformer model. Not called by any fast
    test -- only reached via `_get_model()` when no `model` override is
    supplied.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def _get_model():
    """Lazy process-wide singleton: load once, reuse for every call."""
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _load_model()
    return _model_singleton


def embed_texts(texts: list[str], model=None) -> np.ndarray:
    """Embed a list of texts in a single batched call.

    Returns an empty list for empty `texts` without invoking the model at
    all -- mirrors `entailment.score_chunks`'s empty-input short-circuit.
    """
    if not texts:
        return []
    model = model or _get_model()
    return np.asarray(model.encode(texts), dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 (rather
    than raising or returning NaN) if either vector has zero norm."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean cosine similarity across every unordered pair of embeddings.

    Returns the sentinel 0.0 for fewer than 2 embeddings (no pair exists
    -- this is the "undefined with <2 chunks" case from the module
    docstring), rather than dividing by zero.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard zero-norm rows: dividing by 1 instead of 0 leaves that row
    # all-zero after "normalization", which naturally yields similarity
    # 0.0 against everything else rather than raising or producing NaN.
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms

    sim_matrix = normalized @ normalized.T
    i_upper, j_upper = np.triu_indices(n, k=1)
    return float(sim_matrix[i_upper, j_upper].mean())


def extract_diversity_features(
    question: str, retrieved_chunks: list[str], model=None
) -> dict[str, float]:
    """Compute both diversity features for one (question, retrieved_chunks)
    row. See module docstring for the zero/one/many-chunk sentinel rules.
    """
    if not retrieved_chunks:
        return {name: 0.0 for name in FEATURE_NAMES}

    model = model or _get_model()
    # Batch the question and every chunk into a single encode() call.
    all_embeddings = embed_texts([question] + list(retrieved_chunks), model=model)
    question_embedding = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]

    redundancy = mean_pairwise_cosine(chunk_embeddings)
    centroid = chunk_embeddings.mean(axis=0)
    relevance = _cosine_similarity(centroid, question_embedding)

    return {
        "chunk_redundancy_mean_cosine": redundancy,
        "centroid_question_relevance_cosine": relevance,
    }
