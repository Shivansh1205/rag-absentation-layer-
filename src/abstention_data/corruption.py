"""Corruption logic: turn one clean HotpotQA example into labeled
(question, retrieved_chunks, label, meta) rows simulating a retriever of
varying quality.

Three corruption types
-----------------------
- ``gold_included`` (label=1): the gold passage's answer-bearing chunk is
  guaranteed to be among the retrieved chunks. Genuinely answerable.
- ``distractor_only`` (label=0): the gold passage is dropped entirely;
  all retrieved chunks come from the example's HotpotQA distractor
  passages (topically related, near-miss, but never containing the
  answer).
- ``truncated_span_removed`` (label=0, harder): the gold passage is
  chunked and every chunk containing the answer span is removed, so the
  reader sees real gold-passage context (same entities, same topic) but
  specifically not the sentence(s) that would let them answer.

Deliberate policy choices (surfaced explicitly rather than left as silent
edge-case bugs)
-----------------------------------------------------------------------
1. **Every corruption function requires a literal answer span.** HotpotQA
   has comparison-type questions with derived answers ("yes"/"no") that
   never appear verbatim in any passage. There's no meaningful "was the
   span removed" or "is this chunk safe" check for those, so
   `is_example_usable` gates all three functions and callers (the
   builder) are expected to filter with it up front.
2. **`gold_included` and `truncated_span_removed` both require at least
   one gold-derived chunk slot** (``config.n_gold_chunks >= 1``). A
   config with ``distractor_ratio == 1.0`` makes that impossible by
   construction — rather than silently reserving a slot anyway (which
   would quietly violate the difficulty setting the caller asked for) or
   silently emitting a gold-free "truncated" example that's actually
   indistinguishable from `distractor_only`, both functions raise a clear
   `ValueError` so the caller adjusts the config instead.
3. **`distractor_only` and `truncated_span_removed` filter out any
   distractor chunk that coincidentally contains the literal answer
   span** before sampling. HotpotQA distractors are real Wikipedia
   paragraphs and can coincidentally mention the same entity/number as
   the answer. If *every* available distractor chunk leaks in this way,
   `distractor_only` raises `InsufficientSafeDistractorsError` rather
   than silently including a leaking chunk in a label=0 example.
4. **If removing answer-bearing chunks from the gold passage would leave
   zero gold chunks** (e.g. a short passage where every chunk happens to
   mention the answer), `truncated_span_removed` raises
   `TruncationNotFeasibleError` rather than silently emitting a
   label=0 example with an empty gold contribution — which would make it
   content-identical to `distractor_only` while still being tagged
   ``truncated_span_removed`` in `meta`, corrupting downstream analysis.

None of these four exceptions are caught here. The builder (`builder.py`)
decides the fallback policy (skip the example, fall back to a different
corruption type, etc.) and is the right place to see that decision made
explicitly.
"""

from __future__ import annotations

import random

from abstention_data.chunking import (
    chunk_passage,
    contains_answer_span,
    remove_answer_bearing_chunks,
)
from abstention_data.config import GenerationConfig
from abstention_data.loader import HotpotExample

CORRUPTION_TYPES = ("gold_included", "distractor_only", "truncated_span_removed")


class CorruptionError(Exception):
    """Base class for all corruption-generation failures."""

    def __init__(self, message: str, *, source_id: str):
        super().__init__(message)
        self.source_id = source_id


class NoLiteralAnswerSpanError(CorruptionError):
    """The example's gold passage never contains the answer verbatim."""

    def __init__(self, source_id: str):
        super().__init__(
            f"Example {source_id!r}: answer does not appear verbatim (word-boundary "
            "safe) in the gold passage -- e.g. a yes/no or comparison-derived answer. "
            "Not usable for any corruption type in this phase.",
            source_id=source_id,
        )


class TruncationNotFeasibleError(CorruptionError):
    """Truncating the gold passage can't produce a valid near-miss example."""

    def __init__(self, source_id: str, reason: str):
        super().__init__(
            f"Example {source_id!r}: truncated corruption not feasible ({reason}).",
            source_id=source_id,
        )
        self.reason = reason


class InsufficientSafeDistractorsError(CorruptionError):
    """Every available distractor chunk coincidentally leaks the answer."""

    def __init__(self, source_id: str):
        super().__init__(
            f"Example {source_id!r}: every distractor chunk coincidentally contains "
            "the answer span; cannot build a label=0 example without risking answer "
            "leakage.",
            source_id=source_id,
        )


class AnswerLeakError(CorruptionError):
    """Defensive invariant violation: a label=0 row still contains the answer."""

    def __init__(self, source_id: str, leaking_chunks: list[str], answer: str):
        super().__init__(
            f"Example {source_id!r}: {len(leaking_chunks)} chunk(s) still contain "
            f"the answer span {answer!r} after corruption: {leaking_chunks!r}",
            source_id=source_id,
        )
        self.leaking_chunks = leaking_chunks


def is_example_usable(example: HotpotExample) -> bool:
    """Whether an example's answer appears literally in its gold passage.

    HotpotQA comparison-type questions frequently have derived yes/no
    answers that never appear verbatim in any passage. Those can't
    meaningfully support "is the answer span present" labeling, so the
    builder should filter with this before calling any corruption
    function below.
    """
    return contains_answer_span(example.gold_passage, example.answer)


def assert_no_answer_leak(chunks: list[str], answer: str, *, source_id: str) -> None:
    """Defense-in-depth invariant check: raise `AnswerLeakError` if any
    chunk in `chunks` contains the answer span. Used as a final assertion
    before returning any label=0 row.
    """
    leaking = [c for c in chunks if contains_answer_span(c, answer)]
    if leaking:
        raise AnswerLeakError(source_id, leaking, answer)


def _sample_up_to(rng: random.Random, population: list[str], n: int) -> list[str]:
    """Sample `n` items from `population` without replacement, or return
    all of `population` if it has fewer than `n` items. Never raises for
    `n` larger than the population (unlike `random.Random.sample`).
    """
    if n <= 0:
        return []
    if len(population) <= n:
        return list(population)
    return rng.sample(population, n)


def _flatten_distractor_chunks(
    example: HotpotExample, config: GenerationConfig
) -> list[str]:
    """Chunk every distractor passage the same way the gold passage is
    chunked, and pool all the resulting chunks together."""
    chunks: list[str] = []
    for passage in example.distractor_passages:
        chunks.extend(chunk_passage(passage, config.chunking.max_sentences_per_chunk))
    return chunks


def _build_row(
    question: str,
    retrieved_chunks: list[str],
    *,
    label: int,
    corruption_type: str,
    source_id: str,
) -> dict:
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "label": label,
        "meta": {"corruption_type": corruption_type, "source_id": source_id},
    }


def make_answerable(
    example: HotpotExample, config: GenerationConfig, rng: random.Random
) -> dict:
    """Build a label=1 row: the gold passage's answer-bearing chunk is
    guaranteed to be among the `config.k` retrieved chunks, mixed with
    distractor chunks per `config.distractor_ratio`.
    """
    if not is_example_usable(example):
        raise NoLiteralAnswerSpanError(example.source_id)
    if config.n_gold_chunks < 1:
        raise ValueError(
            "make_answerable requires at least one gold-derived chunk slot "
            f"(k={config.k}, distractor_ratio={config.distractor_ratio} allocates "
            f"{config.n_gold_chunks} gold slots). Lower distractor_ratio or raise k."
        )

    gold_chunks = chunk_passage(
        example.gold_passage, config.chunking.max_sentences_per_chunk
    )
    answer_chunks = [c for c in gold_chunks if contains_answer_span(c, example.answer)]
    other_gold_chunks = [c for c in gold_chunks if c not in answer_chunks]

    if not answer_chunks:
        raise TruncationNotFeasibleError(
            example.source_id,
            reason="gold passage contains the answer but no individual chunk does "
                   "(answer spans a chunk boundary after splitting)",
        )

    # Guarantee exactly one answer-bearing chunk is present (this is what
    # makes the example genuinely answerable), then fill any remaining
    # gold slots from the rest of the gold passage.
    chosen_gold = [rng.choice(answer_chunks)]
    chosen_gold += _sample_up_to(rng, other_gold_chunks, config.n_gold_chunks - 1)

    distractor_pool = _flatten_distractor_chunks(example, config)
    n_distractor_slots = config.k - len(chosen_gold)
    chosen_distractors = _sample_up_to(rng, distractor_pool, n_distractor_slots)

    retrieved = chosen_gold + chosen_distractors
    rng.shuffle(retrieved)

    return _build_row(
        example.question,
        retrieved,
        label=1,
        corruption_type="gold_included",
        source_id=example.source_id,
    )


def make_unanswerable_distractor(
    example: HotpotExample, config: GenerationConfig, rng: random.Random
) -> dict:
    """Build a label=0 row: the gold passage is dropped entirely and all
    `config.k` retrieved chunks come from the example's distractor
    passages, after filtering out any distractor chunk that coincidentally
    contains the literal answer span.
    """
    distractor_pool = _flatten_distractor_chunks(example, config)
    safe_pool = [c for c in distractor_pool if not contains_answer_span(c, example.answer)]

    if not safe_pool:
        raise InsufficientSafeDistractorsError(example.source_id)

    chosen = _sample_up_to(rng, safe_pool, config.k)
    rng.shuffle(chosen)

    assert_no_answer_leak(chosen, example.answer, source_id=example.source_id)

    return _build_row(
        example.question,
        chosen,
        label=0,
        corruption_type="distractor_only",
        source_id=example.source_id,
    )


def make_unanswerable_truncated(
    example: HotpotExample, config: GenerationConfig, rng: random.Random
) -> dict:
    """Build a label=0 (harder) row: the gold passage is chunked and every
    chunk containing the answer span is removed, then the remaining
    gold-passage chunks are mixed with safe distractor chunks up to
    `config.k` total.
    """
    if not is_example_usable(example):
        raise NoLiteralAnswerSpanError(example.source_id)
    if config.n_gold_chunks < 1:
        raise ValueError(
            "make_unanswerable_truncated requires at least one gold-derived chunk "
            f"slot (k={config.k}, distractor_ratio={config.distractor_ratio} allocates "
            f"{config.n_gold_chunks} gold slots). Lower distractor_ratio or raise k."
        )

    gold_chunks = chunk_passage(
        example.gold_passage, config.chunking.max_sentences_per_chunk
    )
    remaining_gold, did_remove_any = remove_answer_bearing_chunks(
        gold_chunks, example.answer
    )
    if not did_remove_any:
        # Shouldn't happen given the is_example_usable precondition, but
        # kept as an explicit, descriptive failure rather than a silent
        # no-op if chunking/removal logic ever drifts out of sync.
        raise TruncationNotFeasibleError(
            example.source_id, reason="no gold chunk contained the answer span"
        )
    if not remaining_gold:
        raise TruncationNotFeasibleError(
            example.source_id,
            reason=(
                "every gold chunk contained the answer span; truncating would leave "
                "zero gold-derived context, making this indistinguishable from "
                "distractor_only"
            ),
        )

    distractor_pool = _flatten_distractor_chunks(example, config)
    safe_distractor_pool = [
        c for c in distractor_pool if not contains_answer_span(c, example.answer)
    ]

    chosen_gold = _sample_up_to(rng, remaining_gold, config.n_gold_chunks)
    n_distractor_slots = config.k - len(chosen_gold)
    chosen_distractors = _sample_up_to(rng, safe_distractor_pool, n_distractor_slots)

    retrieved = chosen_gold + chosen_distractors
    rng.shuffle(retrieved)

    assert_no_answer_leak(retrieved, example.answer, source_id=example.source_id)

    return _build_row(
        example.question,
        retrieved,
        label=0,
        corruption_type="truncated_span_removed",
        source_id=example.source_id,
    )
