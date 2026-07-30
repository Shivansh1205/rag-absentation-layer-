"""Synthetic training data pipeline for a RAG abstention classifier.

Phase 1: pure text-level corruption of HotpotQA into labeled
(question, retrieved_chunks, label, meta) rows. See README.md for the
full pipeline overview and dataset schema.
"""

from abstention_data.builder import BuildStats, build_dataset
from abstention_data.config import ChunkingConfig, GenerationConfig
from abstention_data.loader import HotpotExample, load_hotpotqa

__all__ = [
    "BuildStats",
    "build_dataset",
    "ChunkingConfig",
    "GenerationConfig",
    "HotpotExample",
    "load_hotpotqa",
]
