"""FastAPI live-scoring service for the RAG Abstention Layer demo site.

Single endpoint (`POST /score`) that runs the exact Phase 4 clean-subset
pipeline -- `abstention_features.pipeline.extract_features` for the 5 raw
features, then the same sentinel transform
`abstention_model.features.load_features` applies to
`entity_coverage_fraction`, then the calibrated
`artifacts/eval_clean_subset/model.joblib` classifier -- against
user-submitted (question, chunks) pairs from the site's "Try It Yourself"
section.

This module does not modify `src/abstention_features/` or
`src/abstention_model/` -- it imports their public functions/constants and
reuses them as-is. The one piece of logic duplicated here rather than
imported is `abstention_model.features.load_features`'s sentinel-splitting
transform, because that function only reads from a parquet path; the
constants it's built from (`COVERAGE_COLUMN`, `COVERAGE_UNDEFINED_FILL_VALUE`,
`UNDEFINED_INDICATOR_COLUMN`, `FEATURE_COLUMNS`) are imported directly so
`_build_model_input` below cannot silently drift from what that module
actually does.

Path resolution (local dev vs. Docker)
-----------------------------------------
This file assumes `src/abstention_features` and `src/abstention_model` are
importable as a sibling `src/` directory one level up from this file's
directory (i.e. `<repo_root>/api/app.py` next to `<repo_root>/src/`), and
that the trained model lives at `<repo_root>/artifacts/eval_clean_subset/model.joblib`
-- both true for local dev straight out of the repo. `Dockerfile` mirrors
this exact relative layout inside the image (`/app/src`, `/app/api/app.py`,
`/app/artifacts/eval_clean_subset/model.joblib`) specifically so no
environment variable is required for either path to resolve correctly in
either environment. `MODEL_PATH` can still be overridden via env var for
deployments that want the model mounted somewhere else (e.g. object
storage synced to a volume).
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Make the existing src/ package importable without an editable install ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from abstention_features import diversity, entity_coverage, reranker  # noqa: E402
from abstention_features.entity_coverage import NO_QUESTION_ENTITIES_SENTINEL  # noqa: E402
from abstention_features.pipeline import extract_features  # noqa: E402
from abstention_model.features import (  # noqa: E402
    COVERAGE_COLUMN,
    COVERAGE_UNDEFINED_FILL_VALUE,
    FEATURE_COLUMNS,
    UNDEFINED_INDICATOR_COLUMN,
)

# Checked in this order: MODEL_PATH env var override, then api/model.joblib
# (same directory as this file -- true both in Docker, where the image's
# build context is api/ and model.joblib is COPYed alongside app.py, and in
# local dev now that the model lives in the repo at api/model.joblib), then
# the original artifacts/eval_clean_subset/model.joblib location (kept as a
# fallback for anyone regenerating the model there via
# scripts/train_clean_subset.py without having copied it into api/ yet).
_MODEL_PATH_CANDIDATES = [
    Path(__file__).resolve().parent / "model.joblib",
    Path(__file__).resolve().parent.parent / "artifacts" / "eval_clean_subset" / "model.joblib",
]
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH")
    or next((str(p) for p in _MODEL_PATH_CANDIDATES if p.exists()), str(_MODEL_PATH_CANDIDATES[0]))
)

DEFAULT_THRESHOLD = 0.5


class ScoreRequest(BaseModel):
    question: str
    chunks: list[str] = Field(default_factory=list)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    should_abstain: bool
    confidence: float
    features: dict[str, float]
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# Populated once by `lifespan` at startup; every request reads from these
# instead of loading anything itself. `models` is a plain dict rather than
# five module-level globals so `/health` and `/score` can check readiness
# with one `is None` test instead of five.
models: dict[str, object] = {
    "reranker_model": None,
    "diversity_model": None,
    "entity_coverage_nlp": None,
    "classifier": None,
}


def _models_ready() -> bool:
    return all(v is not None for v in models.values())


def _load_all_models() -> None:
    """Load the three feature-extraction models plus the calibrated
    classifier once. Called from `lifespan` at process startup -- FastAPI
    (via uvicorn's default single-worker startup sequence) does not begin
    accepting connections until this completes, so a reachable `/health`
    already implies "past cold start," not just "process is running." The
    explicit `models_loaded` flag in the response exists for the same
    reason the task asked for it: to make that readiness state visible to
    the frontend rather than only implicit in connection-refused-vs-not.
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer

    import spacy

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Trained model not found at {MODEL_PATH}. Run "
            "scripts/train_clean_subset.py first, or set MODEL_PATH to point "
            "at an existing artifacts/eval_clean_subset/model.joblib."
        )

    models["reranker_model"] = CrossEncoder(reranker.MODEL_NAME)
    models["diversity_model"] = SentenceTransformer(diversity.MODEL_NAME)
    models["entity_coverage_nlp"] = spacy.load(entity_coverage.MODEL_NAME)
    models["classifier"] = joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_models()
    yield
    models.clear()


app = FastAPI(title="RAG Abstention Layer API", lifespan=lifespan)

origins = os.environ.get("ALLOWED_ORIGINS", "*")
if origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_model_input(raw_features: dict[str, float]) -> pd.DataFrame:
    """Apply the exact same `entity_coverage_fraction` sentinel transform
    `abstention_model.features.load_features` applies to a parquet column,
    to a single in-memory feature dict instead. Built from that module's
    own constants (imported above), not re-derived, so this cannot
    silently drift from what the trained model was actually fit on.
    """
    row = dict(raw_features)
    is_undefined = row[COVERAGE_COLUMN] == NO_QUESTION_ENTITIES_SENTINEL
    if is_undefined:
        row[COVERAGE_COLUMN] = COVERAGE_UNDEFINED_FILL_VALUE
    row[UNDEFINED_INDICATOR_COLUMN] = int(is_undefined)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=_models_ready())


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not _models_ready():
        # Shouldn't be reachable in practice -- lifespan blocks serving
        # until models are loaded -- but fail loudly rather than crash on
        # a None.predict_proba if that assumption is ever violated.
        raise HTTPException(status_code=503, detail="Models are still loading. Try again shortly.")

    raw_features = extract_features(
        request.question,
        list(request.chunks),
        reranker_model=models["reranker_model"],
        diversity_model=models["diversity_model"],
        entity_coverage_nlp=models["entity_coverage_nlp"],
    )
    X = _build_model_input(raw_features)

    classifier = models["classifier"]
    confidence = float(classifier.predict_proba(X)[:, 1][0])
    should_abstain = confidence < request.threshold

    return ScoreResponse(
        should_abstain=should_abstain,
        confidence=confidence,
        features=raw_features,
        threshold_used=request.threshold,
    )
