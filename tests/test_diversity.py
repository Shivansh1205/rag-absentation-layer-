"""Tests for `abstention_features.diversity`.

Fast tests use a fake embedder (just needs `.encode(texts) -> array-like`,
matching `sentence_transformers.SentenceTransformer`'s interface) with
fixed, hand-picked vectors, so every assertion checks an actual expected
number, not just "runs without crashing". One @pytest.mark.slow test at
the bottom loads the real all-MiniLM-L6-v2 model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from abstention_features.diversity import (
    FEATURE_NAMES,
    _cosine_similarity,
    embed_texts,
    extract_diversity_features,
    mean_pairwise_cosine,
)

SQRT2 = math.sqrt(2)


class FakeEmbedder:
    """Stand-in for `sentence_transformers.SentenceTransformer`: returns a
    fixed vector per text (looked up by exact string) and records every
    call it received."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self._vectors[t] for t in texts], dtype=float)


class ExplodingModel:
    """Fails the test if ever invoked -- proves the empty-chunks path
    never reaches the model."""

    def encode(self, *args, **kwargs):
        raise AssertionError("encode() should not be called for empty chunks")


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_sentinel_not_nan():
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 0.0])) == 0.0


# ---------------------------------------------------------------------------
# mean_pairwise_cosine
# ---------------------------------------------------------------------------


def test_mean_pairwise_cosine_fewer_than_two_returns_sentinel():
    assert mean_pairwise_cosine(np.array([[1.0, 0.0]])) == 0.0
    assert mean_pairwise_cosine(np.array([])) == 0.0


def test_mean_pairwise_cosine_identical_chunks_is_near_one():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert mean_pairwise_cosine(embeddings) == pytest.approx(1.0)


def test_mean_pairwise_cosine_opposite_chunks_is_negative_one():
    embeddings = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert mean_pairwise_cosine(embeddings) == pytest.approx(-1.0)


def test_mean_pairwise_cosine_matches_hand_computed_three_chunk_average():
    # pairs: (A,B)=0, (A,C)=1, (B,C)=0 -> mean = 1/3
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # A, B, C=dup of A
    assert mean_pairwise_cosine(embeddings) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# extract_diversity_features
# ---------------------------------------------------------------------------


def test_extract_diversity_features_empty_chunks_never_touches_model():
    result = extract_diversity_features("Any question?", [])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_extract_diversity_features_empty_chunks_with_exploding_model():
    result = extract_diversity_features("Any question?", [], model=ExplodingModel())
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_extract_diversity_features_single_chunk_redundancy_is_sentinel_but_relevance_is_real():
    model = FakeEmbedder(
        {
            "Q?": [1.0, 1.0],
            "only chunk": [1.0, 0.0],
        }
    )
    result = extract_diversity_features("Q?", ["only chunk"], model=model)

    # Redundancy sentinel: no pair exists with a single chunk.
    assert result["chunk_redundancy_mean_cosine"] == 0.0
    # Relevance is a real, non-sentinel computed value: cos([1,0], [1,1])
    # = 1/sqrt(2).
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1 / SQRT2)


def test_extract_diversity_features_identical_chunks_high_redundancy():
    model = FakeEmbedder(
        {
            "Q?": [1.0, 0.0],
            "chunk": [1.0, 0.0],
        }
    )
    result = extract_diversity_features("Q?", ["chunk", "chunk"], model=model)
    assert result["chunk_redundancy_mean_cosine"] == pytest.approx(1.0)
    # centroid of two identical [1,0] vectors is [1,0]; relevance to a
    # [1,0] question embedding is also 1.0.
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1.0)


def test_extract_diversity_features_maximally_different_chunks_low_redundancy():
    model = FakeEmbedder(
        {
            "Q?": [1.0, 0.0],
            "chunk a": [1.0, 0.0],
            "chunk b": [-1.0, 0.0],
        }
    )
    result = extract_diversity_features("Q?", ["chunk a", "chunk b"], model=model)
    assert result["chunk_redundancy_mean_cosine"] == pytest.approx(-1.0)


def test_extract_diversity_features_relevance_matches_manual_centroid_calc():
    # centroid of [1,0] and [0,1] is [0.5, 0.5]; cosine to question [1,0]
    # is 0.5 / (sqrt(0.5) * 1) = 1/sqrt(2).
    model = FakeEmbedder(
        {
            "Q?": [1.0, 0.0],
            "chunk a": [1.0, 0.0],
            "chunk b": [0.0, 1.0],
        }
    )
    result = extract_diversity_features("Q?", ["chunk a", "chunk b"], model=model)
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1 / SQRT2)


def test_extract_diversity_features_batches_question_and_chunks_in_one_encode_call():
    model = FakeEmbedder(
        {
            "Q?": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [1.0, 1.0],
        }
    )
    extract_diversity_features("Q?", ["a", "b", "c"], model=model)
    assert len(model.calls) == 1
    assert model.calls[0] == ["Q?", "a", "b", "c"]


def test_embed_texts_empty_list_returns_empty_without_touching_model():
    assert embed_texts([], model=ExplodingModel()) == []


# ---------------------------------------------------------------------------
# Real-model integration test (slow, requires network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_diversity_features_are_in_valid_ranges():
    """Loads the real sentence-transformers/all-MiniLM-L6-v2. Requires
    network access to huggingface.co on first run (cached after)."""
    question = "Where is the Eiffel Tower located?"
    near_duplicate_chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "The Eiffel Tower is a wrought iron lattice tower located in Paris.",
    ]

    features = extract_diversity_features(question, near_duplicate_chunks)
    assert set(features.keys()) == set(FEATURE_NAMES)
    for value in features.values():
        assert -1.0 <= value <= 1.0

    # Two near-duplicate chunks should show high redundancy.
    assert features["chunk_redundancy_mean_cosine"] > 0.8

    diverse_chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Bananas are a good source of potassium and dietary fiber.",
    ]
    diverse_features = extract_diversity_features(question, diverse_chunks)
    assert (
        diverse_features["chunk_redundancy_mean_cosine"]
        < features["chunk_redundancy_mean_cosine"]
    )
