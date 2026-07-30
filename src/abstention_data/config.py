"""Configuration objects for the synthetic data generation pipeline.

Everything that controls "difficulty" of the generated dataset lives here:
how many chunks a simulated retriever returns (``k``), what fraction of
those chunks are distractors vs. gold-derived (``distractor_ratio``), how
passages get split into chunks, and how corruption types are mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Order matters for corruption_type_weights validation errors and for any
# code that wants a stable default ordering.
DEFAULT_CORRUPTION_WEIGHTS: dict[str, float] = {
    "gold_included": 0.4,
    "distractor_only": 0.3,
    "truncated_span_removed": 0.3,
}


@dataclass(frozen=True)
class ChunkingConfig:
    """Controls how a passage's sentences are grouped into retrieval chunks."""

    # Phase 1 only supports sentence-window chunking. Kept as a field (rather
    # than hardcoding) so Phase 2 can add e.g. fixed-token-window chunking
    # without changing callers.
    strategy: str = "sentence"

    # How many sentences make up one chunk before we check whether the
    # answer span survived inside it.
    max_sentences_per_chunk: int = 2

    def __post_init__(self) -> None:
        if self.strategy != "sentence":
            raise ValueError(f"Unsupported chunking strategy: {self.strategy!r}")
        if self.max_sentences_per_chunk < 1:
            raise ValueError("max_sentences_per_chunk must be >= 1")


@dataclass(frozen=True)
class GenerationConfig:
    """Top-level knobs for one dataset generation run.

    Attributes:
        k: Total number of retrieved chunks presented to the model per
            example (across all corruption types).
        distractor_ratio: Fraction of the ``k`` chunks that should come from
            distractor passages rather than the gold passage. For
            ``distractor_only`` examples this is effectively overridden to
            1.0 (no gold chunks exist to include). Higher values make the
            task harder by diluting gold signal with more near-miss
            passages, and increase how many "hard negative" chunks appear
            alongside a truncated gold passage.
        seed: Random seed for reproducible sampling/shuffling.
        chunking: Passage-to-chunk splitting configuration.
        corruption_type_weights: Relative frequency of each label/corruption
            type when building a mixed dataset. Keys must be exactly
            {"gold_included", "distractor_only", "truncated_span_removed"}
            and values must sum to 1.0 (within floating point tolerance).
    """

    k: int = 5
    distractor_ratio: float = 0.6
    seed: int = 42
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    corruption_type_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CORRUPTION_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not (0.0 <= self.distractor_ratio <= 1.0):
            raise ValueError("distractor_ratio must be in [0.0, 1.0]")

        expected_keys = set(DEFAULT_CORRUPTION_WEIGHTS)
        actual_keys = set(self.corruption_type_weights)
        if actual_keys != expected_keys:
            raise ValueError(
                "corruption_type_weights must have exactly keys "
                f"{sorted(expected_keys)}, got {sorted(actual_keys)}"
            )
        total = sum(self.corruption_type_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"corruption_type_weights must sum to 1.0, got {total}"
            )
        if any(w < 0 for w in self.corruption_type_weights.values()):
            raise ValueError("corruption_type_weights must be non-negative")

    @property
    def n_distractor_chunks(self) -> int:
        """Number of the ``k`` retrieved chunks that should be distractors
        for gold-bearing corruption types (``gold_included`` and
        ``truncated_span_removed``)."""
        return round(self.k * self.distractor_ratio)

    @property
    def n_gold_chunks(self) -> int:
        """Number of the ``k`` retrieved chunks that should be gold-derived
        chunks for gold-bearing corruption types."""
        return self.k - self.n_distractor_chunks
