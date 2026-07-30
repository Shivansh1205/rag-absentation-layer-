"""Hand-written HotpotQA-shaped fixtures shared across tests.

These mimic the exact nested structure returned by
``datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")`` so that
`loader._examples_from_raw` (and anything downstream of it) can be tested
without any network access. See `loader.py` module docstring for the raw
schema this is modeling.
"""

from __future__ import annotations

# A single clean, "nice" example: two gold paragraphs (multi-hop), three
# distractors, and an answer that appears verbatim in one of the gold
# sentences.
SIMPLE_RAW_EXAMPLE = {
    "id": "hotpot_ex_0001",
    "question": "What nationality is the director of the film that won the award?",
    "answer": "French",
    "supporting_facts": {
        "title": ["Film Alpha", "Director Beta"],
        "sent_id": [0, 1],
    },
    "context": {
        "title": [
            "Film Alpha",
            "Director Beta",
            "Unrelated Topic One",
            "Unrelated Topic Two",
            "Unrelated Topic Three",
        ],
        "sentences": [
            [
                "Film Alpha is a 2004 movie that won the Golden Prize.",
                "It was shot mostly in Lyon.",
            ],
            [
                "Director Beta directed Film Alpha.",
                "Beta is a French filmmaker born in Marseille.",
            ],
            [
                "Unrelated Topic One is about the history of steel production.",
                "It has nothing to do with cinema.",
            ],
            [
                "Unrelated Topic Two describes a species of freshwater fish.",
                "It is found mostly in Southeast Asia.",
            ],
            [
                "Unrelated Topic Three covers 19th century rail infrastructure.",
                "Construction began in 1861.",
            ],
        ],
    },
}

# An adversarial example designed to stress substring-based answer matching:
# the answer "Mali" appears as a *substring of another word* ("Malibu",
# "formalize") in decoy sentences, and the true (word-boundary) answer
# mention is in the gold passage.
SUBSTRING_ADVERSARIAL_RAW_EXAMPLE = {
    "id": "hotpot_ex_0002",
    "question": "Which country is the musician originally from?",
    "answer": "Mali",
    "supporting_facts": {
        "title": ["Musician Gamma"],
        "sent_id": [1],
    },
    "context": {
        "title": ["Musician Gamma", "Malibu Beach", "Legal Drafting"],
        "sentences": [
            [
                "Musician Gamma is known for blending traditional and modern styles.",
                "Gamma was born in Mali in 1978.",
            ],
            [
                "Malibu Beach is a well known stretch of coastline in California.",
                "It attracts many tourists every summer.",
            ],
            [
                "Lawyers sometimes formalize an agreement before signing.",
                "This distractor paragraph never mentions the real answer.",
            ],
        ],
    },
}

# An example whose supporting_facts reference a title not present in
# context (malformed data) — the loader should skip it rather than crash
# or emit an empty gold passage.
MALFORMED_RAW_EXAMPLE = {
    "id": "hotpot_ex_9999",
    "question": "Broken example with dangling supporting fact title?",
    "answer": "N/A",
    "supporting_facts": {
        "title": ["Title Not In Context"],
        "sent_id": [0],
    },
    "context": {
        "title": ["Some Other Title"],
        "sentences": [["A sentence that is not gold."]],
    },
}

# An example where the answer never appears verbatim in the gold passage
# (HotpotQA "comparison" questions frequently have yes/no or derived
# answers like this). Downstream corruption/builder logic must be able to
# detect and skip these rather than assuming the answer is always present.
NO_LITERAL_ANSWER_RAW_EXAMPLE = {
    "id": "hotpot_ex_0003",
    "question": "Were Film Alpha and Film Delta released in the same decade?",
    "answer": "yes",
    "supporting_facts": {
        "title": ["Film Alpha", "Film Delta"],
        "sent_id": [0, 0],
    },
    "context": {
        "title": ["Film Alpha", "Film Delta"],
        "sentences": [
            ["Film Alpha is a 2004 movie that won the Golden Prize."],
            ["Film Delta is a 2006 movie directed by an American director."],
        ],
    },
}
