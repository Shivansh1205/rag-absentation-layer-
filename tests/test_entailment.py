"""Tests for `abstention_features.entailment`.

Fast tests use a fake CrossEncoder (just needs `.predict(pairs,
apply_softmax=True)` and `.config.id2label`, matching
`sentence_transformers.CrossEncoder`'s interface) so nothing here
downloads a model. One @pytest.mark.slow test at the bottom loads the
real `cross-encoder/nli-deberta-v3-xsmall` and checks its outputs are
sane -- run explicitly with `pytest -m slow` (requires network access to
huggingface.co, which this suite otherwise never needs).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstention_features.entailment import (
    FEATURE_NAMES,
    _resolve_label_indices,
    aggregate_entailment_features,
    extract_entailment_features,
    score_chunks,
)


class FakeCrossEncoder:
    """Stand-in for `sentence_transformers.CrossEncoder`: records the
    pairs it was called with and returns a fixed set of rows."""

    def __init__(self, id2label: dict, responses: list[list[float]]):
        self.config = SimpleNamespace(id2label=id2label)
        self._responses = responses
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, apply_softmax=True):
        self.calls.append(list(pairs))
        assert apply_softmax is True
        return self._responses


class ExplodingModel:
    """A model that fails the test if it's ever actually invoked -- used
    to prove the empty-chunks path never reaches the model at all."""

    def predict(self, *args, **kwargs):
        raise AssertionError("predict() should not be called for empty chunks")

    @property
    def config(self):
        raise AssertionError("config should not be accessed for empty chunks")


# ---------------------------------------------------------------------------
# _resolve_label_indices -- must read id2label, never assume an order
# ---------------------------------------------------------------------------


def test_resolve_label_indices_standard_order():
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    )
    assert _resolve_label_indices(model) == {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }


def test_resolve_label_indices_different_order_is_handled_correctly():
    # Deliberately a totally different order from the "standard" test
    # above, proving indices are read, not assumed.
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    )
    assert _resolve_label_indices(model) == {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }


def test_resolve_label_indices_handles_string_keys():
    # HF configs sometimes serialize id2label with string keys.
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={"0": "neutral", "1": "contradiction", "2": "entailment"})
    )
    assert _resolve_label_indices(model) == {
        "neutral": 0,
        "contradiction": 1,
        "entailment": 2,
    }


def test_resolve_label_indices_is_case_insensitive():
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "CONTRADICTION", 1: "Entailment", 2: "NEUTRAL"})
    )
    assert _resolve_label_indices(model) == {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }


def test_resolve_label_indices_raises_on_missing_label():
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "entailment", 1: "neutral"})  # no contradiction
    )
    with pytest.raises(ValueError, match="contradiction"):
        _resolve_label_indices(model)


# ---------------------------------------------------------------------------
# score_chunks
# ---------------------------------------------------------------------------


def test_score_chunks_empty_list_returns_empty_without_touching_model():
    result = score_chunks("Any question?", [], model=ExplodingModel())
    assert result == []


def test_score_chunks_pairs_chunk_as_premise_question_as_hypothesis():
    model = FakeCrossEncoder(
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        responses=[[0.1, 0.8, 0.1]],
    )
    score_chunks("What color is the sky?", ["The sky is blue."], model=model)
    [pairs] = model.calls
    assert pairs == [("The sky is blue.", "What color is the sky?")]


def test_score_chunks_maps_probabilities_using_this_models_label_order():
    # Label order here is deliberately unusual (neutral first) to prove
    # score_chunks doesn't assume a fixed column order.
    model = FakeCrossEncoder(
        id2label={0: "neutral", 1: "contradiction", 2: "entailment"},
        responses=[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]],
    )
    result = score_chunks("Q?", ["chunk one", "chunk two"], model=model)
    assert result == [
        {"neutral": 0.2, "contradiction": 0.3, "entailment": 0.5},
        {"neutral": 0.1, "contradiction": 0.1, "entailment": 0.8},
    ]


def test_score_chunks_batches_all_chunks_in_one_predict_call():
    model = FakeCrossEncoder(
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        responses=[[0.1, 0.8, 0.1]] * 4,
    )
    score_chunks("Q?", ["a", "b", "c", "d"], model=model)
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 4


# ---------------------------------------------------------------------------
# aggregate_entailment_features
# ---------------------------------------------------------------------------


def test_aggregate_entailment_features_empty_input_returns_zero_dict():
    result = aggregate_entailment_features([])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_aggregate_entailment_features_max_and_mean():
    per_chunk = [
        {"entailment": 0.2, "neutral": 0.7, "contradiction": 0.1},
        {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
        {"entailment": 0.5, "neutral": 0.3, "contradiction": 0.2},
    ]
    result = aggregate_entailment_features(per_chunk)
    assert result["entailment_max_prob"] == pytest.approx(0.9)
    assert result["entailment_mean_prob"] == pytest.approx((0.2 + 0.9 + 0.5) / 3)


def test_best_margin_chunk_can_differ_from_argmax_entailment_chunk():
    # Chunk A has the higher raw entailment prob but also real
    # contradiction mass (ambiguous/noisy). Chunk B has slightly lower
    # entailment but is a much cleaner, more confident signal. The
    # best-margin feature should pick B, proving it's not just a
    # duplicate of entailment_max_prob.
    chunk_a = {"entailment": 0.70, "neutral": 0.10, "contradiction": 0.20}  # margin 0.50
    chunk_b = {"entailment": 0.65, "neutral": 0.34, "contradiction": 0.01}  # margin 0.64
    per_chunk = [chunk_a, chunk_b]

    result = aggregate_entailment_features(per_chunk)

    assert result["entailment_max_prob"] == pytest.approx(0.70)  # chunk A
    assert result["best_margin_chunk_entailment_prob"] == pytest.approx(0.65)  # chunk B
    assert result["entailment_max_prob"] != result["best_margin_chunk_entailment_prob"]


def test_best_margin_matches_max_when_the_top_chunk_has_no_contradiction():
    # Sanity check for the common case: when the highest-entailment chunk
    # also has ~zero contradiction, both features naturally agree -- the
    # two features aren't adversarial, just not defined to be identical.
    per_chunk = [
        {"entailment": 0.9, "neutral": 0.1, "contradiction": 0.0},
        {"entailment": 0.3, "neutral": 0.3, "contradiction": 0.4},
    ]
    result = aggregate_entailment_features(per_chunk)
    assert result["entailment_max_prob"] == pytest.approx(0.9)
    assert result["best_margin_chunk_entailment_prob"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# extract_entailment_features (score_chunks + aggregate, end to end)
# ---------------------------------------------------------------------------


def test_extract_entailment_features_end_to_end_with_fake_model():
    model = FakeCrossEncoder(
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        responses=[[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]],
    )
    result = extract_entailment_features(
        "What nationality is the director?",
        ["Director Beta is French.", "Unrelated steel production paragraph."],
        model=model,
    )
    assert set(result.keys()) == set(FEATURE_NAMES)
    assert result["entailment_max_prob"] == pytest.approx(0.8)


def test_extract_entailment_features_empty_chunks_never_touches_model():
    # No `model=` override at all -- if this ever tried to reach
    # `_get_model()` it would attempt a real network download and this
    # test would hang/fail. It shouldn't, because score_chunks returns
    # early for an empty chunk list before the model is even resolved.
    result = extract_entailment_features("Any question?", [])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Real-model integration test (slow, requires network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_produces_sane_probability_distributions():
    """Loads the real cross-encoder/nli-deberta-v3-xsmall. Requires
    network access to huggingface.co on first run (cached after)."""
    question = "Where is the Eiffel Tower located?"
    chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Bananas are a good source of potassium and dietary fiber.",
    ]

    per_chunk = score_chunks(question, chunks)
    assert len(per_chunk) == 2
    for probs in per_chunk:
        total = probs["entailment"] + probs["neutral"] + probs["contradiction"]
        assert 0.98 <= total <= 1.02
        for v in probs.values():
            assert 0.0 <= v <= 1.0

    features = extract_entailment_features(question, chunks)
    assert set(features.keys()) == set(FEATURE_NAMES)
    for v in features.values():
        assert 0.0 <= v <= 1.0

    # The Eiffel Tower chunk should score noticeably higher than the
    # banana chunk -- a coarse sanity check that the model isn't just
    # producing noise for this framing.
    assert per_chunk[0]["entailment"] > per_chunk[1]["entailment"]
