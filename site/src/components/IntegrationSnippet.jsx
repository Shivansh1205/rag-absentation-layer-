import { BookOpen } from "lucide-react";
import CopyButton from "./CopyButton.jsx";
import { GITHUB_URL, PIP_INSTALL_COMMAND } from "../constants.js";

const SNIPPET = `from rag_abstention import AbstentionScorer

scorer = AbstentionScorer(threshold=0.5)
result = scorer.score(question="...", retrieved_chunks=["..."])

if result.should_abstain:
    print("I don't have enough information to answer that.")
else:
    answer = your_llm.generate(question, chunks)`;

export default function IntegrationSnippet() {
  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-3xl">
        <div className="mx-auto mb-10 max-w-xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Five lines to integrate
          </h2>
          <p className="mt-3 text-balance text-slate-400">
            Drop it in front of the generation step you already have.
          </p>
        </div>

        <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900">
          <div className="flex items-center gap-1.5 border-b border-slate-800 px-4 py-3">
            <span className="h-2.5 w-2.5 rounded-full bg-danger-500/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-warn-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-brand-500/70" />
            <span className="ml-3 font-mono text-xs text-slate-500">quickstart.py</span>
          </div>
          <pre className="overflow-x-auto px-5 py-5 text-sm leading-relaxed">
            <code className="font-mono text-slate-300">{SNIPPET}</code>
          </pre>
        </div>

        <p className="mt-6 text-balance text-center text-sm text-slate-400">
          Works with any retriever and any LLM. LangChain and LlamaIndex adapters coming
          soon.
        </p>

        <div className="mt-8 flex flex-col items-center justify-center gap-4 sm:flex-row">
          <CopyButton text={PIP_INSTALL_COMMAND} variant="primary" />
          <a
            href={`${GITHUB_URL}#readme`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-5 py-3 text-sm font-medium text-slate-100 transition-colors hover:border-brand-500/60 hover:bg-slate-900"
          >
            <BookOpen size={16} />
            Read the docs
          </a>
        </div>
      </div>
    </section>
  );
}
