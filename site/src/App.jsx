import Hero from "./components/Hero.jsx";
import DemoGallery from "./components/DemoGallery.jsx";
import TryItYourself from "./components/TryItYourself.jsx";
import TradeoffChart from "./components/TradeoffChart.jsx";
import IntegrationSnippet from "./components/IntegrationSnippet.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import { GITHUB_URL } from "./constants.js";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950">
      <main>
        <Hero />
        <div className="mx-auto h-px max-w-5xl bg-slate-800/70" />
        <DemoGallery />
        <div className="mx-auto h-px max-w-5xl bg-slate-800/70" />
        <TryItYourself />
        <div className="mx-auto h-px max-w-5xl bg-slate-800/70" />
        <TradeoffChart />
        <div className="mx-auto h-px max-w-5xl bg-slate-800/70" />
        <IntegrationSnippet />
        <div className="mx-auto h-px max-w-5xl bg-slate-800/70" />
        <HowItWorks />
      </main>

      <footer className="border-t border-slate-800/70 px-6 py-10">
        <div className="mx-auto flex max-w-5xl flex-col items-center justify-between gap-4 text-sm text-slate-500 sm:flex-row">
          <span>RAG Abstention Layer &mdash; a portfolio project.</span>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-brand-400">
            github.com/your-username/rag-abstention
          </a>
        </div>
      </footer>
    </div>
  );
}
