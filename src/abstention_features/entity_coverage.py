"""Named-entity overlap between question and retrieved chunks.

Where `entailment.py` asks "does an NLI model think this chunk talks
about the same thing as the question" and `diversity.py` asks "is the
retrieved set topically/embedding-close to the question", this module
asks a narrower, more literal question: **do the concrete named entities
the question mentions actually show up in the retrieved text at all**.
It is deliberately the crudest of the three feature groups -- no model
judgment, just string overlap between two entity lists -- which is
exactly what makes it able to catch cases the other two miss. A chunk
can fail an NLI-declarative-entailment check and still sit far from the
question's embedding centroid while nonetheless containing every proper
noun the question asked about; entity overlap is meant to be that safety
net.

Single feature
--------------
``entity_coverage_fraction``: the fraction of the question's named
entities that are matched (exactly or fuzzily -- see below) by at least
one named entity extracted from the concatenation of `retrieved_chunks`.

Matching: exact-normalized OR fuzzy via `difflib.SequenceMatcher`
-------------------------------------------------------------------
Two entity strings are considered the same mention if, after
normalization (lowercased, surrounding whitespace stripped, periods
removed -- so "U.S." and "US" normalize identically), either:

- they are exactly equal, or
- `difflib.SequenceMatcher(None, a, b).ratio()` is **strictly greater
  than** `FUZZY_MATCH_THRESHOLD` (0.8), catching near-miss spelling/
  transliteration variants like "Marseille" vs "Marseilles" (ratio
  ~0.947) without also catching pairs that merely share a long common
  substring, e.g. "Mali" vs "Malibu" sits at *exactly* 0.8 and is
  therefore NOT counted as a match -- the same word-boundary caution
  Phase 1's answer-span matching (`abstention_data.chunking`) applies
  to substring containment, applied here to entity matching instead.

Sentinel handling -- two distinct sentinels for two distinct "undefined"s
---------------------------------------------------------------------------
- **Empty `retrieved_chunks`**: returns `entity_coverage_fraction=0.0`
  without ever calling the NER pipeline, mirroring
  `entailment.score_chunks` / `diversity.embed_texts`'s empty-input
  short-circuit. This is checked *first*, before the question is even
  looked at, exactly like the other two modules never touch their model
  for an empty-chunks row.
- **Zero entities detected in the question** (e.g. "How does
  photosynthesis work?" -- no proper nouns to check coverage of at all):
  coverage is undefined, not zero -- there is nothing to divide by and
  reporting 0.0 here would be indistinguishable from "the question had
  entities and none of them were covered", which is a materially
  different (and worse) signal. This case returns the documented
  sentinel `NO_QUESTION_ENTITIES_SENTINEL` (-1.0, chosen because it
  falls outside `entity_coverage_fraction`'s valid [0, 1] range and is
  therefore unambiguous downstream) instead.
- Zero entities detected in the *chunks* (chunks present, but nothing
  NER-taggable in them) is **not** a sentinel case: the question had
  entities to check, none of them were found, so `0.0` is the correct,
  well-defined answer.

Lazy singleton + swappable-loader pattern, same as `entailment.py` /
`diversity.py`
----------------------------------------------------------------------
`nlp` is always an optional, injectable parameter -- any callable
matching spaCy's `Language.__call__` interface (`nlp(text)` returning an
object exposing `.ents`, each entity exposing `.text`), defaulting to a
lazy process-wide singleton loading the real `en_core_web_sm` pipeline.
Fast tests inject a fake NER callable; only `@pytest.mark.slow` tests
touch the real spaCy model. `en_core_web_sm` is a separate model
download, not a pip-installable dependency on its own -- see README.md
for the one-time `python -m spacy download en_core_web_sm` setup step.
"""

from __future__ import annotations

from difflib import SequenceMatcher

MODEL_NAME = "en_core_web_sm"

FEATURE_NAMES = ("entity_coverage_fraction",)

# Strictly-greater-than threshold for `difflib.SequenceMatcher.ratio()`
# to count two normalized entity strings as the same mention. Not a
# magic number: named here so the "Mali"/"Malibu"-style boundary case
# (ratio exactly 0.8) is a deliberate, documented exclusion rather than
# an accident of a comparison operator buried in matching logic.
FUZZY_MATCH_THRESHOLD = 0.8

# Returned for `entity_coverage_fraction` when the question has zero
# detected entities -- coverage is undefined (nothing to check), not
# zero. Chosen outside the feature's valid [0, 1] range so it can never
# be confused with a real computed coverage value downstream.
NO_QUESTION_ENTITIES_SENTINEL = -1.0

_nlp_singleton = None


def _load_nlp():
    """Load the real spaCy pipeline. Not called by any fast test -- only
    reached via `_get_nlp()` when no `nlp` override is supplied.
    """
    import spacy

    return spacy.load(MODEL_NAME)


def _get_nlp():
    """Lazy process-wide singleton: load once, reuse for every call."""
    global _nlp_singleton
    if _nlp_singleton is None:
        _nlp_singleton = _load_nlp()
    return _nlp_singleton


def extract_entities(text: str, nlp=None) -> list[str]:
    """Run NER over `text` and return the surface text of every detected
    entity, in document order (duplicates included -- aggregation
    decides what to do with repeats).

    Returns an empty list for empty/whitespace-only `text` without
    invoking the model at all, mirroring `score_chunks` / `embed_texts`.
    `nlp` is injectable (any callable matching spaCy's
    `Language.__call__` interface) so tests can supply a fake instead of
    loading the real model; defaults to the lazy real-model singleton.
    """
    if not text or not text.strip():
        return []
    nlp = nlp or _get_nlp()
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def _normalize_entity(text: str) -> str:
    """Lowercase, strip surrounding whitespace, and drop periods so
    abbreviation variants like "U.S." and "US" normalize to the same
    string and compare as an exact match rather than needing fuzzy
    matching to bridge them."""
    return text.strip().lower().replace(".", "")


def _is_fuzzy_match(a: str, b: str, threshold: float = FUZZY_MATCH_THRESHOLD) -> bool:
    """True if the `difflib.SequenceMatcher` ratio between two
    (already-normalized) strings is strictly greater than `threshold`."""
    return SequenceMatcher(None, a, b).ratio() > threshold


def _entity_is_covered(entity: str, candidate_entities: list[str]) -> bool:
    """True if `entity` matches any entry in `candidate_entities`, either
    exactly (after normalization) or fuzzily (see module docstring)."""
    normalized_entity = _normalize_entity(entity)
    for candidate in candidate_entities:
        normalized_candidate = _normalize_entity(candidate)
        if normalized_entity == normalized_candidate:
            return True
        if _is_fuzzy_match(normalized_entity, normalized_candidate):
            return True
    return False


def compute_entity_coverage(
    question_entities: list[str], chunk_entities: list[str]
) -> float:
    """Fraction of `question_entities` matched by at least one entry in
    `chunk_entities` (exact-normalized or fuzzy match -- see module
    docstring). Returns `NO_QUESTION_ENTITIES_SENTINEL` if
    `question_entities` is empty (coverage undefined, not zero) rather
    than dividing by zero. An empty `chunk_entities` with non-empty
    `question_entities` is well-defined and returns `0.0` (nothing
    covers anything).
    """
    if not question_entities:
        return NO_QUESTION_ENTITIES_SENTINEL

    covered = sum(
        1 for entity in question_entities if _entity_is_covered(entity, chunk_entities)
    )
    return covered / len(question_entities)


def extract_entity_coverage_features(
    question: str, retrieved_chunks: list[str], nlp=None
) -> dict[str, float]:
    """Compute `entity_coverage_fraction` for one (question,
    retrieved_chunks) row. The single entry point `pipeline.py` calls
    into from this module.

    Empty `retrieved_chunks` returns `{"entity_coverage_fraction": 0.0}`
    without ever invoking `nlp` -- checked before the question is looked
    at, matching `entailment.py` / `diversity.py`'s empty-chunks
    short-circuit. See module docstring for the separate
    zero-question-entities sentinel.
    """
    if not retrieved_chunks:
        return {name: 0.0 for name in FEATURE_NAMES}

    nlp = nlp or _get_nlp()
    question_entities = extract_entities(question, nlp=nlp)
    chunks_text = " ".join(retrieved_chunks)
    chunk_entities = extract_entities(chunks_text, nlp=nlp)

    coverage = compute_entity_coverage(question_entities, chunk_entities)
    return {"entity_coverage_fraction": coverage}
