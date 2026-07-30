import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Loader2, Plus, Sparkles, X } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const PRESETS = [
  { label: "Balanced (0.50)", value: 0.5 },
  { label: "Conservative (0.73)", value: 0.73 },
  { label: "Zero hallucination (0.90)", value: 0.9 },
];

const FEATURE_LABELS = {
  reranker_max_score: "Reranker max score",
  reranker_mean_score: "Reranker mean score",
  chunk_redundancy_mean_cosine: "Chunk redundancy (mean cosine)",
  centroid_question_relevance_cosine: "Question relevance (centroid cosine)",
  entity_coverage_fraction: "Entity coverage fraction",
};
const DOMINANT_FEATURE = "reranker_max_score";

const MAX_CHUNKS = 6;
const HEALTH_TIMEOUT_MS = 4000;
const SCORE_TIMEOUT_MS = 20000;
const HEALTH_RETRY_DELAY_MS = 5000;
const HEALTH_MAX_ATTEMPTS = 4; // 1 initial check + up to 3 retries

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function pingHealth() {
  try {
    const res = await fetchWithTimeout(`${API_URL}/health`, {}, HEALTH_TIMEOUT_MS);
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok" && data.models_loaded === true;
  } catch {
    return false;
  }
}

function ServerStatusBanner({ status }) {
  if (status === "ready") return null;

  if (status === "offline") {
    return (
      <div className="mb-6 flex items-start gap-2.5 rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 text-sm text-slate-300">
        <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warn-400" />
        <span>
          Live scoring is currently offline. Check out the pre-built examples above to see
          how it works.
        </span>
      </div>
    );
  }

  // "checking" or "waking"
  return (
    <div className="mb-6 flex items-center gap-2.5 rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 text-sm text-slate-300">
      <Loader2 size={16} className="shrink-0 animate-spin text-brand-400" />
      <span>Waking up the scoring server &mdash; this may take 10-15 seconds on first use.</span>
    </div>
  );
}

function FeatureTable({ features }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-left text-sm">
        <tbody>
          {Object.entries(features).map(([key, value]) => {
            const dominant = key === DOMINANT_FEATURE;
            return (
              <tr
                key={key}
                className={`border-b border-slate-800 last:border-b-0 ${
                  dominant ? "bg-brand-500/10" : "bg-slate-950/40"
                }`}
              >
                <td
                  className={`px-3 py-2 ${dominant ? "font-semibold text-brand-300" : "text-slate-400"}`}
                >
                  {FEATURE_LABELS[key] || key}
                  {dominant && (
                    <span className="ml-2 rounded-full bg-brand-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-brand-300">
                      dominant
                    </span>
                  )}
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono ${
                    dominant ? "font-semibold text-brand-300" : "text-slate-300"
                  }`}
                >
                  {value.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ResultCard({ result }) {
  const { should_abstain, confidence, features, threshold_used } = result;
  const answered = !should_abstain;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Confidence
          </div>
          <div
            className={`font-mono text-5xl font-bold ${
              answered ? "text-brand-400" : "text-warn-400"
            }`}
          >
            {confidence.toFixed(2)}
          </div>
        </div>
        <div
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold ${
            answered ? "bg-brand-500/15 text-brand-400" : "bg-warn-400/15 text-warn-400"
          }`}
        >
          {answered ? `ANSWERED (confidence: ${confidence.toFixed(2)})` : `ABSTAINED (confidence: ${confidence.toFixed(2)})`}
        </div>
      </div>

      {!answered && (
        <div className="rounded-lg border border-warn-400/30 bg-slate-950/60 px-3 py-2.5 text-sm italic leading-snug text-slate-300">
          &ldquo;I don&rsquo;t have enough information to answer this.&rdquo;
        </div>
      )}

      <p className="text-sm leading-relaxed text-slate-400">
        At your threshold of{" "}
        <span className="font-mono font-semibold text-slate-200">
          {threshold_used.toFixed(2)}
        </span>
        , this query would be{" "}
        <span className={`font-semibold ${answered ? "text-brand-400" : "text-warn-400"}`}>
          {answered ? "answered" : "abstained"}
        </span>
        .
      </p>

      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
          Feature breakdown
        </div>
        <FeatureTable features={features} />
      </div>
    </div>
  );
}

export default function TryItYourself() {
  const [serverStatus, setServerStatus] = useState("checking"); // checking | waking | ready | offline
  const [question, setQuestion] = useState("");
  const [chunks, setChunks] = useState([""]);
  const [threshold, setThreshold] = useState(0.5);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState(null);
  const [result, setResult] = useState(null);

  const attemptsRef = useRef(0);
  const timeoutRef = useRef(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;

    async function runCheck() {
      const healthy = await pingHealth();
      if (cancelledRef.current) return;

      if (healthy) {
        setServerStatus("ready");
        return;
      }

      attemptsRef.current += 1;
      if (attemptsRef.current >= HEALTH_MAX_ATTEMPTS) {
        setServerStatus("offline");
        return;
      }
      setServerStatus("waking");
      timeoutRef.current = setTimeout(runCheck, HEALTH_RETRY_DELAY_MS);
    }

    runCheck();

    return () => {
      cancelledRef.current = true;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  function updateChunk(i, value) {
    setChunks((prev) => prev.map((c, idx) => (idx === i ? value : c)));
  }

  function addChunk() {
    setChunks((prev) => (prev.length >= MAX_CHUNKS ? prev : [...prev, ""]));
  }

  function removeChunk(i) {
    setChunks((prev) => (prev.length <= 1 ? prev : prev.filter((_, idx) => idx !== i)));
  }

  const canSubmit =
    serverStatus === "ready" &&
    !scoring &&
    question.trim().length > 0 &&
    chunks.some((c) => c.trim().length > 0);

  async function handleScore(e) {
    e.preventDefault();
    if (!canSubmit) return;

    setScoring(true);
    setScoreError(null);
    setResult(null);

    try {
      const res = await fetchWithTimeout(
        `${API_URL}/score`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            chunks: chunks.map((c) => c.trim()).filter(Boolean),
            threshold,
          }),
        },
        SCORE_TIMEOUT_MS
      );
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch {
      setScoreError(
        "Couldn't reach the scoring server. Live scoring may be offline — the pre-built examples above still work."
      );
    } finally {
      setScoring(false);
    }
  }

  return (
    <section id="try-it" className="px-6 py-24">
      <div className="mx-auto max-w-5xl">
        <div className="mx-auto mb-10 max-w-2xl text-center">
          <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs font-medium text-brand-400">
            <Sparkles size={12} />
            Live scoring
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Try it yourself
          </h2>
          <p className="mt-3 text-balance text-slate-400">
            Paste a question and a few retrieved chunks. The real, running classifier scores
            them live &mdash; nothing pre-baked.
          </p>
        </div>

        <ServerStatusBanner status={serverStatus} />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Form */}
          <form
            onSubmit={handleScore}
            className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6"
          >
            <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wider text-slate-500">
              Your question
            </label>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What year was the Eiffel Tower built?"
              rows={2}
              className="mb-5 w-full resize-none rounded-xl border border-slate-700 bg-slate-950/60 px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-brand-500/60 focus:outline-none"
            />

            <div className="mb-1.5 flex items-center justify-between">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
                Retrieved chunks
              </label>
              <button
                type="button"
                onClick={addChunk}
                disabled={chunks.length >= MAX_CHUNKS}
                className="inline-flex items-center gap-1 text-xs font-medium text-brand-400 transition-colors hover:text-brand-300 disabled:cursor-not-allowed disabled:text-slate-600"
              >
                <Plus size={13} />
                Add another chunk
              </button>
            </div>
            <div className="mb-5 space-y-2.5">
              {chunks.map((chunk, i) => (
                <div key={i} className="flex items-start gap-2">
                  <textarea
                    value={chunk}
                    onChange={(e) => updateChunk(i, e.target.value)}
                    placeholder={`Chunk ${i + 1}...`}
                    rows={2}
                    className="w-full resize-none rounded-xl border border-slate-700 bg-slate-950/60 px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-brand-500/60 focus:outline-none"
                  />
                  {chunks.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeChunk(i)}
                      aria-label={`Remove chunk ${i + 1}`}
                      className="mt-1 shrink-0 rounded-lg p-1.5 text-slate-600 transition-colors hover:bg-slate-800 hover:text-slate-300"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>

            <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-500">
              Threshold: <span className="font-mono text-slate-300">{threshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="mb-3 h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-brand-400"
              aria-label="Abstention threshold"
            />
            <div className="mb-6 flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => setThreshold(p.value)}
                  className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                    Math.abs(threshold - p.value) < 0.005
                      ? "border-brand-500 bg-brand-500/15 text-brand-300"
                      : "border-slate-700 bg-slate-950/60 text-slate-400 hover:border-slate-600 hover:text-slate-200"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand-500 px-5 py-3 text-sm font-semibold text-slate-950 transition-all hover:bg-brand-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              {scoring ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Scoring...
                </>
              ) : (
                "Score"
              )}
            </button>
          </form>

          {/* Results */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
            {scoring ? (
              <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-3 text-slate-400">
                <Loader2 size={22} className="animate-spin text-brand-400" />
                <span className="text-sm">Scoring...</span>
              </div>
            ) : scoreError ? (
              <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-3 px-4 text-center">
                <AlertTriangle size={22} className="text-warn-400" />
                <p className="text-sm text-slate-400">{scoreError}</p>
              </div>
            ) : result ? (
              <ResultCard result={result} />
            ) : serverStatus === "offline" ? (
              <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-3 px-4 text-center text-slate-500">
                <AlertTriangle size={22} className="text-slate-600" />
                <p className="text-sm">
                  Live scoring is currently offline. Check out the pre-built examples above to
                  see how it works.
                </p>
              </div>
            ) : (
              <div className="flex h-full min-h-[240px] flex-col items-center justify-center gap-2 px-4 text-center text-slate-500">
                <Sparkles size={22} className="text-slate-700" />
                <p className="text-sm">
                  Fill in the form and click <span className="font-medium text-slate-400">Score</span> to
                  see a live result.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
