"""Passage chunking and answer-span-aware truncation.

Chunking splits a passage into sentence windows so we can build "N
retrieved chunks" of realistic size, and -- critically -- so the
"truncated" unanswerable corruption (see `corruption.py`) can drop exactly
the chunk(s) that would let a reader answer the question, while keeping
the rest of the gold passage in place as a near-miss.

Word-boundary safety
---------------------
Every answer-presence check in this module treats the answer as a whole
phrase, never a substring. A naive ``answer in text`` check is unsound in
both directions:

- It produces false positives when the answer happens to be a substring
  of an unrelated word (e.g. answer "Mali" inside "Malibu" or
  "formalize"). Treating that as "answer present" would make a properly
  scrubbed chunk look contaminated, and would make the truncation
  corruption think it can never successfully remove the span.
- Conversely, it *should* flag (and correctly does, via the same
  word-boundary check) a genuine standalone mention of the answer
  elsewhere in the passage or in a distractor -- that's a real
  informational leak, not a false alarm, and must not be suppressed.

`contains_answer_span` uses regex word-boundary lookaround to get both of
these right.
"""

from __future__ import annotations

import re

# Split on sentence-ending punctuation followed by whitespace and what
# looks like the start of a new sentence (capital letter, digit, or
# quote). This is a lightweight heuristic, not a full sentence tokenizer:
# Phase 1 is pure text-level corruption with no NLP/ML dependencies, and
# HotpotQA passages are short, clean Wikipedia-style prose where this
# works well in practice. It will occasionally mis-split on abbreviations
# (e.g. "U.S. Steel"); that's an acceptable Phase 1 tradeoff.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_into_sentences(passage: str) -> list[str]:
    """Split a passage into sentences using lightweight regex rules."""
    passage = passage.strip()
    if not passage:
        return []
    sentences = _SENTENCE_BOUNDARY_RE.split(passage)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list[str], max_sentences_per_chunk: int) -> list[str]:
    """Group consecutive sentences into windows of up to
    ``max_sentences_per_chunk`` sentences each, joined back into chunk
    strings.
    """
    if max_sentences_per_chunk < 1:
        raise ValueError("max_sentences_per_chunk must be >= 1")
    chunks: list[str] = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        window = sentences[i : i + max_sentences_per_chunk]
        chunks.append(" ".join(window))
    return chunks


def chunk_passage(passage: str, max_sentences_per_chunk: int) -> list[str]:
    """Convenience wrapper: split a passage directly into chunk strings."""
    return chunk_sentences(split_into_sentences(passage), max_sentences_per_chunk)


def contains_answer_span(text: str, answer: str) -> bool:
    r"""Word-boundary-aware, case-insensitive check for whether `answer`
    appears in `text` as a standalone phrase.

    Returns False for empty/whitespace-only answers (nothing meaningful to
    match -- e.g. some HotpotQA answers are empty strings for malformed
    examples).

    Uses lookaround assertions (``(?<!\w)`` / ``(?!\w)``) rather than the
    ``\b`` metacharacter directly: ``\b`` gets confused when the answer
    itself starts or ends with a non-word character (e.g. an answer like
    "St. Louis"), whereas the lookaround form just checks that the
    characters immediately outside the match aren't word characters.
    """
    answer = answer.strip()
    if not answer:
        return False
    pattern = r"(?<!\w)" + re.escape(answer) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def remove_answer_bearing_chunks(
    chunks: list[str], answer: str
) -> tuple[list[str], bool]:
    """Drop every chunk that contains the answer span (word-boundary safe).

    Returns ``(remaining_chunks, did_remove_any)``. Used by the
    "truncated" corruption to guarantee the specific answer span is cut
    out while keeping the surrounding gold-passage chunks in place as a
    near-miss.
    """
    remaining = [c for c in chunks if not contains_answer_span(c, answer)]
    did_remove_any = len(remaining) < len(chunks)
    return remaining, did_remove_any
