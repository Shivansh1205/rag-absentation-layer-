import { useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, XCircle } from "lucide-react";
import examples from "../data/examples.json";

function ChunkList({ chunks }) {
  return (
    <ul className="space-y-1.5">
      {chunks.map((chunk, i) => (
        <li
          key={i}
          className="rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-xs leading-relaxed text-slate-400"
        >
          {chunk}
        </li>
      ))}
    </ul>
  );
}

function WithoutCard({ example }) {
  const isHallucinated = example.label === 0;
  return (
    <div className="flex h-full flex-col rounded-2xl border border-danger-500/25 bg-slate-900/60 p-6">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Without Abstention Layer
        </h4>
        <XCircle className="h-5 w-5 text-danger-500/70" />
      </div>

      <p className="mb-4 text-sm font-medium leading-snug text-slate-200">{example.question}</p>

      <div className="mb-4">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Retrieved chunks
        </div>
        <ChunkList chunks={example.chunks} />
      </div>

      <div className="mt-auto space-y-2.5 border-t border-slate-800 pt-4">
        <div
          className={`rounded-lg px-3 py-2.5 text-sm leading-snug ${
            isHallucinated
              ? "border border-danger-500/40 bg-danger-500/10 text-danger-300"
              : "border border-slate-700 bg-slate-950/60 text-slate-300"
          }`}
        >
          <span className="font-semibold">Answer: </span>
          {isHallucinated ? example.hallucinated_answer : example.correct_answer}
        </div>
        {isHallucinated && (
          <div className="inline-flex items-center gap-1.5 rounded-full bg-danger-500/15 px-3 py-1 text-xs font-semibold text-danger-400">
            <AlertTriangle size={12} />
            HALLUCINATED &mdash; chunks don&rsquo;t contain the answer
          </div>
        )}
      </div>
    </div>
  );
}

function WithCard({ example }) {
  const answered = example.label === 1;
  return (
    <div
      className={`flex h-full flex-col rounded-2xl border p-6 ${
        answered ? "border-brand-500/30 bg-slate-900/60" : "border-warn-400/25 bg-slate-900/60"
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          With Abstention Layer
        </h4>
        <CheckCircle2
          className={`h-5 w-5 ${answered ? "text-brand-400" : "text-warn-400/80"}`}
        />
      </div>

      <p className="mb-4 text-sm font-medium leading-snug text-slate-200">{example.question}</p>

      <div className="mb-4">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
          Retrieved chunks
        </div>
        <ChunkList chunks={example.chunks} />
      </div>

      <div className="mt-auto space-y-2.5 border-t border-slate-800 pt-4">
        {answered ? (
          <>
            <div className="rounded-lg border border-brand-500/30 bg-slate-950/60 px-3 py-2.5 text-sm leading-snug text-slate-200">
              <span className="font-semibold">Answer: </span>
              {example.correct_answer}
            </div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-brand-500/15 px-3 py-1 text-xs font-semibold text-brand-400">
              ANSWERED (confidence: {example.confidence_score.toFixed(2)})
            </div>
          </>
        ) : (
          <>
            <div className="rounded-lg border border-warn-400/30 bg-slate-950/60 px-3 py-2.5 text-sm italic leading-snug text-slate-300">
              &ldquo;I don&rsquo;t have enough information to answer this.&rdquo;
            </div>
            <div className="inline-flex items-center gap-1.5 rounded-full bg-warn-400/15 px-3 py-1 text-xs font-semibold text-warn-400">
              ABSTAINED (confidence: {example.confidence_score.toFixed(2)})
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function DemoGallery() {
  const [index, setIndex] = useState(0);
  const example = examples[index];

  function go(delta) {
    setIndex((prev) => (prev + delta + examples.length) % examples.length);
  }

  return (
    <section id="demo" className="px-6 py-24">
      <div className="mx-auto max-w-5xl">
        <div className="mx-auto mb-4 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            See it catch a hallucination
          </h2>
          <p className="mt-3 text-balance text-slate-400">
            8 real questions from the eval set. Same retrieval, two outcomes &mdash; with and
            without the abstention layer sitting in between.
          </p>
        </div>

        <div className="mb-6 flex items-center justify-center gap-2">
          <span className="rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs font-medium text-slate-400">
            {example.corruption_type === "gold_included" ? "Answerable case" : "Unanswerable case"}
          </span>
          <span className="text-xs text-slate-600">
            {index + 1} / {examples.length}
          </span>
        </div>

        <div className="relative grid grid-cols-1 gap-5 md:grid-cols-2">
          <WithoutCard example={example} />
          <WithCard example={example} />
        </div>

        {/* Navigation: arrows + dots */}
        <div className="mt-8 flex items-center justify-center gap-6">
          <button
            type="button"
            onClick={() => go(-1)}
            aria-label="Previous example"
            className="rounded-full border border-slate-800 bg-slate-900/60 p-2 text-slate-300 transition-colors hover:border-brand-500/50 hover:text-brand-400"
          >
            <ChevronLeft size={18} />
          </button>

          <div className="flex items-center gap-2">
            {examples.map((ex, i) => (
              <button
                key={ex.id}
                type="button"
                onClick={() => setIndex(i)}
                aria-label={`Go to example ${i + 1}`}
                className={`h-2 rounded-full transition-all ${
                  i === index ? "w-6 bg-brand-400" : "w-2 bg-slate-700 hover:bg-slate-500"
                }`}
              />
            ))}
          </div>

          <button
            type="button"
            onClick={() => go(1)}
            aria-label="Next example"
            className="rounded-full border border-slate-800 bg-slate-900/60 p-2 text-slate-300 transition-colors hover:border-brand-500/50 hover:text-brand-400"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
    </section>
  );
}
