import { useState } from "react";
import { Check, Copy } from "lucide-react";

/**
 * Button that copies `text` to the clipboard and shows a brief "Copied!"
 * toast. Used for the "pip install rag-abstention" CTA in both the Hero
 * and the Integration Snippet sections so the two stay visually/behaviorally
 * identical rather than drifting into two separate implementations.
 */
export default function CopyButton({ text, className = "", variant = "primary" }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API can be unavailable (e.g. insecure context, older
      // browsers) -- fall back to a hidden textarea + execCommand rather
      // than silently doing nothing.
      const el = document.createElement("textarea");
      el.value = text;
      el.style.position = "fixed";
      el.style.opacity = "0";
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  const base =
    "relative inline-flex items-center gap-2 rounded-xl px-5 py-3 font-mono text-sm font-medium transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950";
  const variants = {
    primary:
      "bg-brand-500 text-slate-950 hover:bg-brand-400 shadow-lg shadow-brand-500/20 active:scale-[0.98]",
    ghost:
      "border border-slate-700 bg-slate-900/60 text-slate-100 hover:border-brand-500/60 hover:bg-slate-900 active:scale-[0.98]",
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`${base} ${variants[variant]} ${className}`}
      aria-label={`Copy "${text}" to clipboard`}
    >
      {copied ? <Check size={16} /> : <Copy size={16} />}
      <span>{text}</span>
      <span
        role="status"
        aria-live="polite"
        className={`pointer-events-none absolute -top-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-lg bg-brand-500 px-2.5 py-1 text-xs font-semibold text-slate-950 shadow-lg transition-all duration-200 ${
          copied ? "opacity-100 -translate-y-0" : "opacity-0 translate-y-1"
        }`}
      >
        Copied!
      </span>
    </button>
  );
}
