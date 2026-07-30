import { ArrowDownRight, Github, ShieldCheck, TrendingDown } from "lucide-react";
import CopyButton from "./CopyButton.jsx";
import { GITHUB_URL, PIP_INSTALL_COMMAND } from "../constants.js";

// Real numbers from artifacts/eval_clean_subset/threshold_sweep.csv
// (gold_included vs distractor_only clean-subset eval, 708 rows):
//   t=0.00 (answer everything): hallucination_rate_of_answered = 44.77%
//   t=0.50 (default threshold): hallucination_rate_of_answered = 17.33%,
//                                coverage_of_answerable          = 79.28%
//   t=0.90 (strictest preset):  hallucination_rate_of_answered =  0.00%
const STATS = [
  {
    value: "45% → 17%",
    label: "Hallucination rate at default threshold",
    icon: TrendingDown,
  },
  {
    value: "79%",
    label: "Answerable questions still covered",
    icon: ShieldCheck,
  },
  {
    value: "0%",
    label: "Hallucinations at strictest setting",
    icon: ShieldCheck,
  },
];

export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-glow px-6 pb-20 pt-24 sm:pt-32">
      <div className="mx-auto flex max-w-5xl flex-col items-center text-center">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/60 px-4 py-1.5 text-xs font-medium uppercase tracking-wider text-brand-400">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-400" />
          Open-source middleware
        </div>

        <h1 className="text-balance text-4xl font-extrabold tracking-tight text-white sm:text-6xl">
          RAG Abstention Layer
        </h1>
        <p className="mt-4 text-balance text-xl font-medium text-slate-300 sm:text-2xl">
          Stop your RAG system from hallucinating.
        </p>

        {/* Stat cards */}
        <div className="mt-12 grid w-full grid-cols-1 gap-4 sm:grid-cols-3">
          {STATS.map(({ value, label, icon: Icon }) => (
            <div
              key={label}
              className="group rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-left transition-colors hover:border-brand-500/50"
            >
              <Icon className="mb-3 h-5 w-5 text-brand-400" strokeWidth={2.25} />
              <div className="font-mono text-3xl font-bold text-white sm:text-4xl">{value}</div>
              <div className="mt-2 text-sm leading-snug text-slate-400">{label}</div>
            </div>
          ))}
        </div>

        <p className="mt-10 max-w-2xl text-balance text-base leading-relaxed text-slate-400 sm:text-lg">
          A pip-installable middleware that sits between your retriever and your LLM. It
          scores retrieved context and abstains when the evidence isn&rsquo;t strong enough
          &mdash; so your system says &ldquo;I don&rsquo;t know&rdquo; instead of making
          things up.
        </p>

        <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row">
          <CopyButton text={PIP_INSTALL_COMMAND} variant="primary" />
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-5 py-3 text-sm font-medium text-slate-100 transition-colors hover:border-brand-500/60 hover:bg-slate-900"
          >
            <Github size={16} />
            View on GitHub
          </a>
        </div>

        <a
          href="#demo"
          className="mt-14 inline-flex items-center gap-1 text-sm text-slate-500 transition-colors hover:text-brand-400"
        >
          See it catch a hallucination
          <ArrowDownRight size={15} />
        </a>
      </div>
    </section>
  );
}
