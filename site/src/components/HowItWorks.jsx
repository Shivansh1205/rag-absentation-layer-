import { Search, ShieldCheck, ToggleRight } from "lucide-react";

const STEPS = [
  {
    icon: Search,
    title: "Your retriever pulls chunks",
    body: "Whatever retrieval you already run — vector search, hybrid, rerank-first.",
  },
  {
    icon: ShieldCheck,
    title: "Our scorer evaluates evidence quality",
    body: "A calibrated classifier estimates whether the retrieved context actually supports an answer.",
  },
  {
    icon: ToggleRight,
    title: "Answer or abstain based on your threshold",
    body: "Confident enough → generate. Not confident enough → say so, instead of guessing.",
  },
];

export default function HowItWorks() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-5xl">
        <div className="mx-auto mb-14 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            How it works
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          {STEPS.map(({ icon: Icon, title, body }, i) => (
            <div key={title} className="relative rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500/15 text-brand-400">
                  <Icon size={20} strokeWidth={2.25} />
                </div>
                <span className="font-mono text-xs text-slate-600">STEP {i + 1}</span>
              </div>
              <h3 className="mb-2 text-base font-semibold text-slate-100">{title}</h3>
              <p className="text-sm leading-relaxed text-slate-400">{body}</p>
            </div>
          ))}
        </div>

        <p className="mx-auto mt-12 max-w-3xl text-balance text-center text-sm leading-relaxed text-slate-400 sm:text-base">
          Under the hood, the scorer extracts features from the query-chunk relationship
          &mdash; relevance scoring via a cross-encoder reranker, embedding-based diversity
          analysis, and entity coverage &mdash; and feeds them into a calibrated
          gradient-boosted classifier. The output is a probability that the retrieved
          context actually contains the answer, not just related text.
        </p>
      </div>
    </section>
  );
}
