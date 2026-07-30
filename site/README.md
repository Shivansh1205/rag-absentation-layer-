# RAG Abstention Layer — demo site

Static portfolio site (Vite + React + Tailwind) for the RAG Abstention
Layer project. The Demo Gallery and Tradeoff Chart sections are fully
static (pre-baked data in `src/data/`) and work with zero setup. The "Try
It Yourself" section additionally calls a live scoring API in `../api/`.

## Static-only (no live scoring)

```bash
npm install
npm run dev
```

The Demo Gallery, Tradeoff Chart, Integration Snippet, and How It Works
sections all work with this alone. "Try It Yourself" will show its
offline state ("Live scoring is currently offline...") since there's no
API running — this is expected and doesn't break anything else on the
page.

## To run with live scoring locally

Two terminals, from the repo root:

**Terminal 1 — the API:**

```bash
cd api
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m spacy download en_core_web_sm
uvicorn app:app --reload
```

First startup loads three real models (the reranker, the sentence
embedder, and spaCy's NER pipeline) plus the trained classifier, so it
can take a little while before `GET http://localhost:8000/health` returns
`{"status": "ok", "models_loaded": true}`. See `../api/app.py` and
`../docs/phase4_diagnosis.md` for what those models are and why.

**Terminal 2 — the site:**

```bash
cd site
npm run dev
```

The site connects to `http://localhost:8000` for live scoring by default
— no `.env` needed for local dev. If your API runs somewhere else, copy
`.env.example` to `.env.local` and set `VITE_API_URL` there.

## Deploying

- **Site**: `npm run build` → deploy `dist/` to Vercel/Netlify/GitHub
  Pages. Set `VITE_API_URL` (as a build-time env var on whichever
  platform you use) to your deployed API's URL.
- **API**: see `../api/Dockerfile` — deploy to Render/Railway/Fly.io.
  Set `ALLOWED_ORIGINS` on the API to your deployed site's origin (e.g.
  `https://your-site.vercel.app`) once you know it; it defaults to `*`.

## Project structure

```
site/
  src/
    components/     # Hero, DemoGallery, TryItYourself, TradeoffChart, IntegrationSnippet, HowItWorks
    data/           # pre-baked examples.json + threshold_sweep.json (static sections only)
    constants.js    # GitHub URL / pip install command placeholders
```
