"""HotpotQA loading and normalization into a common example schema.

HotpotQA's "distractor" config gives each question 10 context paragraphs:
2 "gold" paragraphs that contain the supporting facts needed to answer, and
8 "distractor" paragraphs that are topically related (often via entity or
category overlap) but insufficient on their own. That's exactly the shape
we want for building near-miss unanswerable examples in `corruption.py`.

Raw record shape (as returned by
``datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")``, or any
hand-built dict with the same nested structure)::

    {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "...",
        "answer": "...",
        "supporting_facts": {"title": ["Title A", "Title B"], "sent_id": [0, 2]},
        "context": {
            "title": ["Title A", "Title B", "Title C", ...],
            "sentences": [["Sent 1.", "Sent 2."], [...], [...], ...],
        },
    }

This module only depends on that shape, not on the `datasets` library
itself, which is why `_examples_from_raw` can be unit tested against a
local fixture with no network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class HotpotExample:
    """A normalized HotpotQA example ready for corruption.

    Attributes:
        source_id: Original HotpotQA example id, kept for traceability
            (e.g. tracing a misclassified generated example back to its
            source question during error analysis).
        question: The question text.
        gold_passage: Concatenation of all supporting ("gold") paragraphs,
            in the order they appear in ``context``. For most examples this
            is 2 paragraphs (HotpotQA is multi-hop), so the necessary
            information may span the join.
        distractor_passages: The remaining (non-gold) context paragraphs,
            each kept as a separate passage string.
        answer: The gold answer string.
    """

    source_id: str
    question: str
    gold_passage: str
    distractor_passages: list[str]
    answer: str


def _passage_from_sentences(sentences: list[str]) -> str:
    """Join a paragraph's per-sentence strings into one passage string."""
    return " ".join(s.strip() for s in sentences if s.strip())


def _parse_raw_record(raw: dict) -> Optional[HotpotExample]:
    """Parse a single raw HotpotQA-shaped dict into a `HotpotExample`, or
    `None` if it's malformed and should be skipped.

    This is the *only* place in the codebase that reads the raw HotpotQA
    field layout (``raw["context"]["title"]``, ``raw["supporting_facts"]``,
    etc.). Everything downstream — chunking, corruption, the builder —
    only ever sees a `HotpotExample`. If the live Hub schema turns out to
    differ from the documented shape assumed here (see module docstring),
    this is the one function that needs to change; nothing else does.

    Returns `None` when ``supporting_facts`` references a title that isn't
    present in ``context`` (malformed data, which does occur rarely in
    HotpotQA) — better to skip than yield a bogus empty gold passage.
    """
    titles: list[str] = raw["context"]["title"]
    sentences_per_title: list[list[str]] = raw["context"]["sentences"]
    gold_titles = set(raw["supporting_facts"]["title"])

    gold_parts: list[str] = []
    distractor_passages: list[str] = []
    for title, sentences in zip(titles, sentences_per_title):
        passage = _passage_from_sentences(sentences)
        if not passage:
            continue
        if title in gold_titles:
            gold_parts.append(passage)
        else:
            distractor_passages.append(passage)

    if not gold_parts:
        return None

    return HotpotExample(
        source_id=raw["id"],
        question=raw["question"],
        gold_passage=" ".join(gold_parts),
        distractor_passages=distractor_passages,
        answer=raw["answer"],
    )


def _examples_from_raw(raw_examples: Iterable[dict]) -> Iterator[HotpotExample]:
    """Convert an iterable of raw HotpotQA-shaped dict records into
    `HotpotExample`s, dropping any that `_parse_raw_record` reports as
    malformed.
    """
    for raw in raw_examples:
        example = _parse_raw_record(raw)
        if example is not None:
            yield example


def load_hotpotqa(split: str = "train", limit: Optional[int] = None) -> list[HotpotExample]:
    """Load HotpotQA (distractor setting) from the Hugging Face Hub and
    normalize it into `HotpotExample`s.

    Requires network access on first call (cached locally by `datasets`
    afterwards). This function is a thin wrapper around
    `_examples_from_raw` specifically so that normalization logic can be
    unit tested against a local fixture without touching the network —
    see `tests/test_loader.py`.

    Uses the canonical Hub repo id ``hotpotqa/hotpot_qa`` (not a bare
    ``"hotpot_qa"`` short name, which the Hub's namespace unification has
    made unreliable, and not `rajpurkar/*`, which is the SQuAD author's
    namespace and an unrelated dataset). No `trust_remote_code` flag is
    passed: recent `datasets` releases have dropped loading-script support
    entirely in favor of the Hub's auto-converted parquet, and passing
    that flag now just prints a stale deprecation notice for no benefit.

    Args:
        split: HotpotQA split to load ("train" or "validation"; HotpotQA
            has no public test split with answers).
        limit: If set, only load the first `limit` raw examples before
            normalizing (useful for smoke tests and fast local runs).
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return list(_examples_from_raw(ds))
