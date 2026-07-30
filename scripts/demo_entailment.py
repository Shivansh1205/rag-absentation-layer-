#!/usr/bin/env python
"""Print entailment + diversity + entity-coverage features for a handful
of example rows, so the signals can be eyeballed together -- this is the
actual test of whether the three-feature-group design covers each
other's blind spots, not just whether each module runs in isolation.

Builds 2 `gold_included` (label=1), 2 `distractor_only` (label=0), and 1
`truncated_span_removed` (label=0) row from hand-built HotpotQA-shaped
fixtures using Phase 1's own corruption pipeline (`abstention_data`,
unmodified), then runs Phase 2's real entailment, diversity, and
entity-coverage models on each.

Requires network access to download `cross-encoder/nli-deberta-v3-xsmall`,
`sentence-transformers/all-MiniLM-L6-v2`, and (via
`python -m spacy download en_core_web_sm`, a one-time separate step --
see README.md) spaCy's `en_core_web_sm` pipeline. No HotpotQA/dataset
download needed -- the fixtures here are self-contained.

Usage:
    uv run python scripts/demo_entailment.py
"""

from __future__ import annotations

import random

from abstention_data.corruption import (
    make_answerable,
    make_unanswerable_distractor,
    make_unanswerable_truncated,
)
from abstention_data.config import ChunkingConfig, GenerationConfig
from abstention_data.loader import HotpotExample
from abstention_features.diversity import extract_diversity_features
from abstention_features.entailment import extract_entailment_features
from abstention_features.entity_coverage import extract_entity_coverage_features

EXAMPLES = [
    HotpotExample(
        source_id="demo_1",
        question="What nationality is the director of the film that won the award?",
        gold_passage=(
            "Film Alpha is a 2004 movie that won the Golden Prize. "
            "Director Beta directed Film Alpha. "
            "Beta is a French filmmaker born in Marseille."
        ),
        distractor_passages=[
            "Unrelated Topic One is about the history of steel production in the region.",
            "Unrelated Topic Two describes a species of freshwater fish found in Southeast Asia.",
            "Unrelated Topic Three covers 19th century rail infrastructure and construction.",
        ],
        answer="French",
    ),
    HotpotExample(
        source_id="demo_2",
        question="Which country is the musician originally from?",
        gold_passage=(
            "Musician Gamma is known for blending traditional and modern styles. "
            "Gamma was born in Mali in 1978."
        ),
        distractor_passages=[
            "Malibu Beach is a well known stretch of coastline in California that attracts tourists.",
            "Lawyers sometimes formalize an agreement before signing important documents.",
        ],
        answer="Mali",
    ),
]

# demo_1's gold_included row is a known hard case for entailment.py: the
# gold passage never says "French" near a sentence that mentions the
# question's topic in NLI-friendly declarative form -- "Beta is a French
# filmmaker born in Marseille" requires inferring nationality from a place
# name ("Marseille" -> France -> French), which an NLI model trained on
# direct declarative entailment tends to miss (previously observed:
# entailment_max_prob ~0.0006 on this exact row). This flag exists so
# that number gets read every time this script runs, not just once.
KNOWN_LIMITATION_SOURCE_ID = "demo_1"


def _print_row(label_name: str, row: dict, *, flag_note: str | None = None) -> None:
    header = f"\n=== {label_name} | label={row['label']} | corruption_type={row['meta']['corruption_type']} | source_id={row['meta']['source_id']} ==="
    print(header)
    if flag_note:
        print(f"[KNOWN LIMITATION CASE] {flag_note}")
    print(f"question: {row['question']}")
    for i, chunk in enumerate(row["retrieved_chunks"]):
        print(f"  [{i}] {chunk}")

    entailment_features = extract_entailment_features(row["question"], row["retrieved_chunks"])
    diversity_features = extract_diversity_features(row["question"], row["retrieved_chunks"])
    entity_coverage_features = extract_entity_coverage_features(
        row["question"], row["retrieved_chunks"]
    )

    print("  -- entailment.py --")
    for name, value in entailment_features.items():
        print(f"  {name}: {value:.4f}")
    print("  -- diversity.py --")
    for name, value in diversity_features.items():
        print(f"  {name}: {value:.4f}")
    print("  -- entity_coverage.py --")
    for name, value in entity_coverage_features.items():
        print(f"  {name}: {value:.4f}")

    if flag_note:
        relevance = diversity_features["centroid_question_relevance_cosine"]
        entailment_max = entailment_features["entailment_max_prob"]
        coverage = entity_coverage_features["entity_coverage_fraction"]

        diversity_compensates = relevance > 0.3
        # NO_QUESTION_ENTITIES_SENTINEL (-1.0): the question itself
        # ("What nationality is the director of the film that won the
        # award?") names no one -- "the director" is a generic
        # description, not a proper noun -- so it's entirely possible
        # spaCy finds zero entities to check coverage of at all. That is
        # itself a real, worth-reporting answer (a distinct failure mode
        # from "found entities, none covered"), not just an edge case to
        # paper over -- checked explicitly rather than assumed either way.
        coverage_is_defined = coverage >= 0.0
        coverage_catches_it = coverage_is_defined and coverage > 0.3

        print(
            f"  [KNOWN LIMITATION CASE] entailment_max_prob={entailment_max:.4f} | "
            f"centroid_question_relevance_cosine={relevance:.4f} | "
            f"entity_coverage_fraction={coverage:.4f}"
            + (
                " (sentinel: spaCy found zero named entities in the question itself --"
                " 'the director' is a generic description, not a proper noun, so there is"
                " nothing for entity_coverage to check overlap against on this row)"
                if not coverage_is_defined
                else ""
            )
        )
        print(
            "  [KNOWN LIMITATION CASE] verdict: entailment misses this row (near-zero). "
            f"diversity {'partially compensates' if diversity_compensates else 'does NOT compensate'} "
            f"(relevance {'is' if diversity_compensates else 'is not'} meaningfully positive). "
            "entity_coverage "
            + (
                f"DOES catch the overlap (fraction={coverage:.4f} > 0.3 of the question's "
                "named entities are matched, exactly or fuzzily, among the chunk entities)."
                if coverage_catches_it
                else (
                    "is UNDEFINED on this row (question has no named entities to check -- "
                    "see sentinel note above), which is itself the finding: this question's "
                    "phrasing ('the director of the film...' with no name given) structurally "
                    "defeats entity-overlap matching regardless of what the chunks contain."
                    if not coverage_is_defined
                    else f"does NOT clearly catch it either (fraction={coverage:.4f})."
                )
                + " If entailment, diversity, and entity_coverage all leave the same gap on "
                "this row, that's a real blind spot in the three-feature-group design, not "
                "just a demo curiosity."
            )
        )


def main() -> int:
    config = GenerationConfig(
        k=4, distractor_ratio=0.5, seed=0, chunking=ChunkingConfig(max_sentences_per_chunk=1)
    )

    print(
        "Loading cross-encoder/nli-deberta-v3-xsmall, "
        "sentence-transformers/all-MiniLM-L6-v2, and en_core_web_sm "
        "(first run downloads/loads all three; en_core_web_sm must already "
        "be installed via `python -m spacy download en_core_web_sm`, see "
        "README.md)..."
    )

    for i, example in enumerate(EXAMPLES):
        rng = random.Random(100 + i)
        row = make_answerable(example, config, rng)
        flag_note = (
            "demo_1 gold_included: checking whether centroid_question_relevance_cosine "
            "(diversity) and entity_coverage_fraction (entity_coverage) pick up the "
            "'Marseille'/'filmmaker' -> 'French nationality' link that entailment.py "
            "is known to miss on this row."
            if example.source_id == KNOWN_LIMITATION_SOURCE_ID
            else None
        )
        _print_row("gold_included (answerable)", row, flag_note=flag_note)

    for i, example in enumerate(EXAMPLES):
        rng = random.Random(200 + i)
        _print_row(
            "distractor_only (unanswerable)",
            make_unanswerable_distractor(example, config, rng),
        )

    rng = random.Random(300)
    _print_row(
        "truncated_span_removed (unanswerable, harder)",
        make_unanswerable_truncated(EXAMPLES[1], config, rng),
    )

    print(
        "\nExpectation to eyeball: gold_included rows should show noticeably "
        "higher entailment_max_prob / best_margin_chunk_entailment_prob, "
        "higher centroid_question_relevance_cosine, AND higher "
        "entity_coverage_fraction than distractor_only or truncated_span_removed "
        "rows. If all three leave the same gap on the flagged demo_1 row "
        "specifically, that's a real, three-feature-group blind spot, not just "
        "a demo curiosity -- see the [KNOWN LIMITATION CASE] verdict line above "
        "for the actual numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
