# RAG Abstention Layer -- live scoring API.
#
# Build context must be the REPO ROOT, not api/, because this needs to
# copy sibling directories (src/, artifacts/eval_clean_subset/) that live
# outside api/. From the repo root:
#
#   docker build -f api/Dockerfile -t rag-abstention-api .
#   docker run -p 8000:8000 -e ALLOWED_ORIGINS="https://your-site.vercel.app" rag-abstention-api
#
# The image mirrors the exact same relative layout app.py already assumes
# for local dev (api/app.py next to src/, and
# artifacts/eval_clean_subset/model.joblib two levels up from app.py) --
# see app.py's module docstring. That's deliberate: it means neither
# MODEL_PATH nor any src-path override needs to be set as an env var here,
# the default resolution just works.

FROM python:3.12-slim

WORKDIR /app

# System deps sentence-transformers/spacy/torch need to build/run cleanly
# on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -m spacy download en_core_web_sm

# Application code + the existing feature/model packages this service
# imports from, unmodified -- see app.py's module docstring for the exact
# import chain (abstention_features.pipeline, abstention_model.features).
COPY api/app.py ./api/app.py
COPY src/abstention_features ./src/abstention_features
COPY src/abstention_model ./src/abstention_model

# The trained clean-subset classifier this service serves. Regenerate via
# scripts/train_clean_subset.py if it's missing or stale at build time.
COPY api/model.joblib ./model.joblib

WORKDIR /app/api

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
