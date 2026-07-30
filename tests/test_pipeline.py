"""Tests for `abstention_features.pipeline`.

Fast tests stub all three underlying models (fake CrossEncoder reranker,
fake SentenceTransformer, fake spaCy NER callable -- same fake interfaces
used in test_reranker.py / test_diversity.py / test_entity_coverage.py,
redefined locally so this file doesn't depend on those test modules'
internals). One @pytest.mark.slow test at the bottom loads all three real
models together.

Phase 4 note: this pipeline uses `reranker.py`
(`cross-encoder/ms-marco-MiniLM-L6-v2`), not `entailment.py`. See
pipeline.py's module docstring for why -- entailment.py is still a real,
tested module in the codebase, this pipeline just stopped calling it.

Note on feature count: the original task spec that requested this module
said "Final feature set (7 total)" but the 6 raw names at the time were
exactly entailment.FEATURE_NAMES (3) + diversity.FEATURE_NAMES (2) +
entity_coverage.FEATURE_NAMES (1), and `entity_coverage_undefined` (the
implied 7th) is `abstention_model.features.load_features`'s own derived
column, added downstream of this module -- not something pipeline.py
itself produces. That discrepancy carries forward here at a new count:
Phase 4 swapped entailment's 3 features for reranker's 2, so
`pipeline.feature_names` is 5 keys (reranker's 2 + diversity's 2 +
entity_coverage's 1), and the classifier's actual final input is still
one more than that (6) once `load_features` appends
`entity_coverage_undefined`. Tests below assert against the real, derived
list rather than a hardcoded magic number.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from abstention_features import diversity, entity_coverage, reranker
from abstention_features.pipeline import (
    EMPTY_CHUNKS_FEATURES,
    extract_features,
    feature_names,
)


# ---------------------------------------------------------------------------
# Fakes -- mirror the interfaces the three modules expect from `model=`/`nlp=`
# ---------------------------------------------------------------------------


class FakeReranker:
    def __init__(self, responses: list[float]):
        self._responses = responses

    def predict(self, pairs):
        return self._responses


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def encode(self, texts):
        return np.array([self._vectors[t] for t in texts], dtype=float)


class FakeNLP:
    def __init__(self, responses: dict[str, list[str]]):
        self._responses = responses

    def __call__(self, text: str):
        entities = self._responses.get(text, [])
        return SimpleNamespace(ents=[SimpleNamespace(text=e) for e in entities])


class ExplodingReranker:
    def predict(self, *args, **kwargs):
        raise AssertionError("CrossEncoder.predict should not be called for empty chunks")


class ExplodingEmbedder:
    def encode(self, *args, **kwargs):
        raise AssertionError("encode() should not be called for empty chunks")


class ExplodingNLP:
    def __call__(self, *args, **kwargs):
        raise AssertionError("nlp() should not be called for empty chunks")


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------


def test_feature_names_is_the_concatenation_of_the_three_modules_own_names():
    assert feature_names == [
        *reranker.FEATURE_NAMES,
        *diversity.FEATURE_NAMES,
        *entity_coverage.FEATURE_NAMES,
    ]
    assert feature_names == [
        "reranker_max_score",
        "reranker_mean_score",
        "chunk_redundancy_mean_cosine",
        "centroid_question_relevance_cosine",
        "entity_coverage_fraction",
    ]


def test_empty_chunks_features_has_exactly_feature_names_keys_all_zero():
    assert set(EMPTY_CHUNKS_FEATURES.keys()) == set(feature_names)
    assert EMPTY_CHUNKS_FEATURES == {name: 0.0 for name in feature_names}


# ---------------------------------------------------------------------------
# extract_features -- empty retrieved_chunks: single well-defined sentinel dict
# ---------------------------------------------------------------------------


def test_extract_features_empty_chunks_returns_full_sentinel_dict_touching_no_model():
    result = extract_features(
        "Any question?",
        [],
        reranker_model=ExplodingReranker(),
        diversity_model=ExplodingEmbedder(),
        entity_coverage_nlp=ExplodingNLP(),
    )
    # Exact values for all keys, not just "no crash" / "no missing keys".
    assert result == {
        "reranker_max_score": 0.0,
        "reranker_mean_score": 0.0,
        "chunk_redundancy_mean_cosine": 0.0,
        "centroid_question_relevance_cosine": 0.0,
        "entity_coverage_fraction": 0.0,
    }
    assert set(result.keys()) == set(feature_names)


def test_extract_features_empty_chunks_no_model_overrides_still_short_circuits():
    # No injected models at all -- if this ever tried to reach any
    # module's default lazy singleton it would attempt real model loads.
    result = extract_features("Any question?", [])
    assert result == EMPTY_CHUNKS_FEATURES


def test_extract_features_empty_chunks_returns_a_fresh_copy_not_the_shared_constant():
    # Callers must not be able to mutate the module-level sentinel dict
    # via the return value of a single extract_features() call.
    result = extract_features("Q?", [])
    result["reranker_max_score"] = 999.0
    assert EMPTY_CHUNKS_FEATURES["reranker_max_score"] == 0.0


# ---------------------------------------------------------------------------
# extract_features -- merged dict from three stubbed models (non-empty chunks)
# ---------------------------------------------------------------------------


def test_extract_features_merges_all_three_modules_with_correct_values():
    question = "What nationality is the director of the film that won the award?"
    chunks = ["Chunk A mentions Beta.", "Chunk B mentions nothing relevant."]
    joined_chunks = " ".join(chunks)

    # Raw reranker logits (not probabilities): chunk A clearly relevant,
    # chunk B clearly not.
    reranker_model = FakeReranker(responses=[7.5, -3.2])
    diversity_model = FakeEmbedder(
        {
            question: [1.0, 0.0],
            "Chunk A mentions Beta.": [1.0, 0.0],
            "Chunk B mentions nothing relevant.": [0.0, 1.0],
        }
    )
    entity_coverage_nlp = FakeNLP(
        {
            question: ["Beta"],
            joined_chunks: ["Beta"],
        }
    )

    result = extract_features(
        question,
        chunks,
        reranker_model=reranker_model,
        diversity_model=diversity_model,
        entity_coverage_nlp=entity_coverage_nlp,
    )

    assert set(result.keys()) == set(feature_names)
    # reranker: max=7.5 (chunk A), mean=(7.5-3.2)/2.
    assert result["reranker_max_score"] == pytest.approx(7.5)
    assert result["reranker_mean_score"] == pytest.approx((7.5 - 3.2) / 2)
    # diversity: 2 orthogonal chunk vectors -> redundancy 0.0; centroid
    # [0.5,0.5] vs question [1,0] -> cos = 1/sqrt(2).
    assert result["chunk_redundancy_mean_cosine"] == pytest.approx(0.0)
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1 / (2**0.5))
    # entity_coverage: question entity "Beta" exact-matches chunk entity
    # "Beta" -> full coverage.
    assert result["entity_coverage_fraction"] == pytest.approx(1.0)


def test_extract_features_routes_each_model_to_the_correct_module():
    # Uses distinct, easily-attributable fake outputs per module so a
    # mis-wired injection (e.g. diversity_model handed to reranker) would
    # fail loudly rather than coincidentally passing.
    question = "Q?"
    chunks = ["only chunk"]

    reranker_model = FakeReranker(responses=[0.42])
    diversity_model = FakeEmbedder({"Q?": [1.0, 1.0], "only chunk": [1.0, 0.0]})
    entity_coverage_nlp = FakeNLP({"Q?": [], "only chunk": []})

    result = extract_features(
        question,
        chunks,
        reranker_model=reranker_model,
        diversity_model=diversity_model,
        entity_coverage_nlp=entity_coverage_nlp,
    )
    assert result["reranker_max_score"] == pytest.approx(0.42)
    assert result["reranker_mean_score"] == pytest.approx(0.42)
    # Single chunk: redundancy sentinel 0.0, relevance is real.
    assert result["chunk_redundancy_mean_cosine"] == 0.0
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1 / (2**0.5))
    # Zero question entities -> entity_coverage's own sentinel.
    assert result["entity_coverage_fraction"] == entity_coverage.NO_QUESTION_ENTITIES_SENTINEL


# ---------------------------------------------------------------------------
# Sentinels from non-empty rows must pass through untouched
# ---------------------------------------------------------------------------


def test_zero_question_entities_sentinel_passes_through_while_other_features_are_real():
    # Non-empty, answerable-looking chunks; question has no detectable
    # entities. entity_coverage_fraction must be exactly the sentinel
    # (-1.0), and it must NOT contaminate/zero-out the other 4 features,
    # which should be real, distinctly non-sentinel computed values.
    question = "How does photosynthesis work?"
    chunks = [
        "Plants use sunlight to convert carbon dioxide and water into energy.",
        "Chlorophyll absorbs light in the chloroplasts of plant cells.",
    ]
    joined_chunks = " ".join(chunks)

    reranker_model = FakeReranker(responses=[5.2, 3.8])
    diversity_model = FakeEmbedder(
        {
            question: [1.0, 0.0],
            chunks[0]: [1.0, 0.0],
            chunks[1]: [0.9, 0.1],
        }
    )
    entity_coverage_nlp = FakeNLP(
        {
            question: [],  # zero entities in the question
            joined_chunks: ["Chlorophyll"],  # chunks do have entities
        }
    )

    result = extract_features(
        question,
        chunks,
        reranker_model=reranker_model,
        diversity_model=diversity_model,
        entity_coverage_nlp=entity_coverage_nlp,
    )

    assert result["entity_coverage_fraction"] == -1.0
    assert result["entity_coverage_fraction"] == entity_coverage.NO_QUESTION_ENTITIES_SENTINEL

    # The other 4 features are real, non-sentinel, non-zero values --
    # proving the -1.0 sentinel from one module didn't leak into or
    # zero-out the others.
    assert result["reranker_max_score"] == pytest.approx(5.2)
    assert result["reranker_mean_score"] == pytest.approx((5.2 + 3.8) / 2)
    assert result["chunk_redundancy_mean_cosine"] > 0.9  # near-identical vectors
    assert result["centroid_question_relevance_cosine"] > 0.9
    for name in (
        "reranker_max_score",
        "reranker_mean_score",
        "chunk_redundancy_mean_cosine",
        "centroid_question_relevance_cosine",
    ):
        assert result[name] != -1.0


def test_single_chunk_diversity_redundancy_sentinel_passes_through():
    # diversity.py's own "fewer than 2 chunks" redundancy sentinel (0.0)
    # must show up unchanged in the merged dict, alongside a real
    # (non-sentinel) relevance value from the same single-chunk call.
    question = "Q?"
    chunks = ["only chunk"]

    reranker_model = FakeReranker(responses=[2.0])
    diversity_model = FakeEmbedder({"Q?": [1.0, 1.0], "only chunk": [1.0, 0.0]})
    entity_coverage_nlp = FakeNLP({"Q?": ["Q"], "only chunk": ["Q"]})

    result = extract_features(
        question,
        chunks,
        reranker_model=reranker_model,
        diversity_model=diversity_model,
        entity_coverage_nlp=entity_coverage_nlp,
    )
    assert result["chunk_redundancy_mean_cosine"] == 0.0
    assert result["centroid_question_relevance_cosine"] == pytest.approx(1 / (2**0.5))
    assert result["centroid_question_relevance_cosine"] != 0.0


# ---------------------------------------------------------------------------
# Real-model integration test (slow, requires network + en_core_web_sm)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_models_end_to_end_on_two_examples():
    """Loads all three real models (cross-encoder/ms-marco-MiniLM-L6-v2,
    sentence-transformers/all-MiniLM-L6-v2, en_core_web_sm) and runs the
    full pipeline on two examples reused from the other modules'
    real-model tests / scripts/demo_entailment.py's fixtures."""
    # Example 1: clean case with an unambiguous named entity in the
    # question itself ("Eiffel Tower"), reused from
    # test_reranker.py / test_diversity.py's own slow tests.
    question_1 = "Where is the Eiffel Tower located?"
    chunks_1 = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Bananas are a good source of potassium and dietary fiber.",
    ]
    result_1 = extract_features(question_1, chunks_1)
    assert set(result_1.keys()) == set(feature_names)
    # reranker scores are unbounded raw logits, not probabilities -- see
    # reranker.py's module docstring for the documented empirical range.
    assert -20.0 <= result_1["reranker_max_score"] <= 20.0
    assert -1.0 <= result_1["chunk_redundancy_mean_cosine"] <= 1.0
    assert -1.0 <= result_1["centroid_question_relevance_cosine"] <= 1.0
    # "Eiffel Tower" appears verbatim in chunk 1 -> real, positive coverage.
    assert result_1["entity_coverage_fraction"] > 0.0

    # Example 2: demo_entailment.py's known-hard demo_1 row -- the
    # question names no one ("the director"), so entity_coverage_fraction
    # is expected to hit its own sentinel here rather than a computed
    # fraction; asserted explicitly rather than just range-checked, since
    # that sentinel-vs-real distinction is the entire point of this
    # module passing sentinels through untouched.
    question_2 = "What nationality is the director of the film that won the award?"
    chunks_2 = [
        "Film Alpha is a 2004 movie that won the Golden Prize.",
        "Director Beta directed Film Alpha.",
        "Beta is a French filmmaker born in Marseille.",
    ]
    result_2 = extract_features(question_2, chunks_2)
    assert set(result_2.keys()) == set(feature_names)
    assert result_2["entity_coverage_fraction"] == entity_coverage.NO_QUESTION_ENTITIES_SENTINEL

    # Empty chunks still short-circuits correctly even with real models
    # already loaded/injected via the module singletons.
    assert extract_features(question_1, []) == EMPTY_CHUNKS_FEATURES
