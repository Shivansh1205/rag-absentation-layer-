"""Tests for `abstention_features.reranker`.

Fast tests use a fake CrossEncoder (just needs `.predict(pairs) ->
array-like of scalars`, matching `sentence_transformers.CrossEncoder`'s
interface for a single-output-neuron checkpoint) so nothing here
downloads a model. One @pytest.mark.slow test at the bottom loads the
real `cross-encoder/ms-marco-MiniLM-L6-v2` and checks its outputs are
sane -- run explicitly with `pytest -m slow` (requires network access to
huggingface.co, which this suite otherwise never needs).
"""

from __future__ import annotations

import pytest

from abstention_features.reranker import (
    FEATURE_NAMES,
    aggregate_reranker_features,
    extract_reranker_features,
    score_chunks,
)


class FakeCrossEncoder:
    """Stand-in for `sentence_transformers.CrossEncoder`: records the
    pairs it was called with and returns a fixed set of raw scores."""

    def __init__(self, responses: list[float]):
        self._responses = responses
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return self._responses


class ExplodingModel:
    """A model that fails the test if it's ever actually invoked -- used
    to prove the empty-chunks path never reaches the model at all."""

    def predict(self, *args, **kwargs):
        raise AssertionError("predict() should not be called for empty chunks")


# ---------------------------------------------------------------------------
# score_chunks
# ---------------------------------------------------------------------------


def test_score_chunks_empty_list_returns_empty_without_touching_model():
    result = score_chunks("Any question?", [], model=ExplodingModel())
    assert result == []


def test_score_chunks_pairs_question_first_then_chunk():
    # Unlike entailment.py's (chunk, question) framing, MS MARCO rerankers
    # are trained on (query, passage) pairs -- question must come first.
    model = FakeCrossEncoder(responses=[4.2])
    score_chunks("What color is the sky?", ["The sky is blue."], model=model)
    [pairs] = model.calls
    assert pairs == [("What color is the sky?", "The sky is blue.")]


def test_score_chunks_batches_all_chunks_in_one_predict_call():
    model = FakeCrossEncoder(responses=[1.0, 2.0, 3.0, 4.0])
    score_chunks("Q?", ["a", "b", "c", "d"], model=model)
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 4


def test_score_chunks_returns_raw_scores_in_order_including_negatives():
    # Raw logits, not probabilities -- negative values are expected and
    # must pass through unmodified (no softmax, no clipping).
    model = FakeCrossEncoder(responses=[-8.3, 0.0, 6.1])
    result = score_chunks("Q?", ["irrelevant", "borderline", "relevant"], model=model)
    assert result == pytest.approx([-8.3, 0.0, 6.1])


def test_score_chunks_does_not_pass_apply_softmax():
    # A single-output-neuron checkpoint has nothing to softmax against --
    # calling predict() with apply_softmax would either error or silently
    # do something meaningless on a real CrossEncoder. Assert the fake
    # (which only accepts `pairs`) is called correctly, i.e. this doesn't
    # raise a TypeError for an unexpected keyword argument.
    model = FakeCrossEncoder(responses=[1.0])
    score_chunks("Q?", ["chunk"], model=model)  # would raise if apply_softmax were passed


# ---------------------------------------------------------------------------
# aggregate_reranker_features
# ---------------------------------------------------------------------------


def test_aggregate_reranker_features_empty_input_returns_zero_dict():
    result = aggregate_reranker_features([])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


def test_aggregate_reranker_features_max_and_mean():
    scores = [-3.0, 5.5, 1.2]
    result = aggregate_reranker_features(scores)
    assert result["reranker_max_score"] == pytest.approx(5.5)
    assert result["reranker_mean_score"] == pytest.approx((-3.0 + 5.5 + 1.2) / 3)


def test_aggregate_reranker_features_single_chunk():
    result = aggregate_reranker_features([2.7])
    assert result["reranker_max_score"] == pytest.approx(2.7)
    assert result["reranker_mean_score"] == pytest.approx(2.7)


def test_aggregate_reranker_features_handles_all_negative_scores():
    # All-irrelevant retrieval: every chunk scores negative. max/mean must
    # not silently floor at 0 -- these are raw logits, not probabilities.
    result = aggregate_reranker_features([-9.0, -4.0, -6.0])
    assert result["reranker_max_score"] == pytest.approx(-4.0)
    assert result["reranker_mean_score"] == pytest.approx((-9.0 - 4.0 - 6.0) / 3)


# ---------------------------------------------------------------------------
# extract_reranker_features (score_chunks + aggregate, end to end)
# ---------------------------------------------------------------------------


def test_extract_reranker_features_end_to_end_with_fake_model():
    model = FakeCrossEncoder(responses=[7.1, -2.4])
    result = extract_reranker_features(
        "What nationality is the director?",
        ["Director Beta is French.", "Unrelated steel production paragraph."],
        model=model,
    )
    assert set(result.keys()) == set(FEATURE_NAMES)
    assert result["reranker_max_score"] == pytest.approx(7.1)
    assert result["reranker_mean_score"] == pytest.approx((7.1 - 2.4) / 2)


def test_extract_reranker_features_empty_chunks_never_touches_model():
    # No `model=` override at all -- if this ever tried to reach
    # `_get_model()` it would attempt a real network download and this
    # test would hang/fail. It shouldn't, because score_chunks returns
    # early for an empty chunk list before the model is even resolved.
    result = extract_reranker_features("Any question?", [])
    assert result == {name: 0.0 for name in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Real-model integration test (slow, requires network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_produces_sane_scores_and_discriminates_answer_presence():
    """Loads the real cross-encoder/ms-marco-MiniLM-L6-v2. Requires
    network access to huggingface.co on first run (cached after).

    This is the module's actual reason for existing (see module
    docstring): Phase 4's clean-subset AUC diagnostic showed the prior
    feature set (entailment.py + diversity.py) could not tell
    "passage with the answer sentence" apart from "same passage, answer
    sentence removed" (AUC 0.5495, ~chance). This test checks the reranker
    can.
    """
    question = "Where is the Eiffel Tower located?"
    with_answer = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
        "Mars in Paris, France. It was designed by Gustave Eiffel."
    )
    answer_removed = (
        "It was designed by Gustave Eiffel. The tower is one of the most "
        "recognizable structures in the world."
    )
    unrelated = "Bananas are a good source of potassium and dietary fiber."

    scores = score_chunks(question, [with_answer, answer_removed, unrelated])
    assert len(scores) == 3

    # Document the actual observed range for this checkpoint (raw logits,
    # not probabilities -- see module docstring) rather than assuming one.
    # A generous bound: fail loudly if a future model swap produces
    # something wildly out of the range this module's docstring documents,
    # rather than silently accepting garbage.
    for s in scores:
        assert -20.0 <= s <= 20.0

    score_with_answer, score_answer_removed, score_unrelated = scores

    # The core claim this module exists to validate: a passage containing
    # the answer sentence should score meaningfully higher than the same
    # passage with it removed -- the exact discrimination entailment.py
    # and diversity.py's topical-relatedness features could not make.
    assert score_with_answer > score_answer_removed

    # And both France-related passages should clearly outscore an
    # unrelated one.
    assert score_with_answer > score_unrelated
    assert score_answer_removed > score_unrelated

    features = extract_reranker_features(question, [with_answer, answer_removed, unrelated])
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert features["reranker_max_score"] == pytest.approx(score_with_answer)
