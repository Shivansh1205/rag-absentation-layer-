import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import sweep from "../data/threshold_sweep.json";

const PRESETS = [
  { label: "Balanced (0.50)", value: 0.5 },
  { label: "Conservative (0.73)", value: 0.73 },
  { label: "Zero hallucination (0.90)", value: 0.9 },
];

const pct = (v) => `${(v * 100).toFixed(0)}%`;
const pct1 = (v) => `${(v * 100).toFixed(1)}%`;

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1 font-mono font-semibold text-slate-200">threshold = {label.toFixed(2)}</div>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-1.5" style={{ color: p.color }}>
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.color }} />
          {p.name}: {pct1(p.value)}
        </div>
      ))}
    </div>
  );
}

export default function TradeoffChart() {
  const [threshold, setThreshold] = useState(0.5);

  // Grid is np.linspace(0, 1, 101) -- exact 0.01 steps -- so this index
  // lookup is exact, not a nearest-neighbor approximation.
  const current = useMemo(() => {
    const idx = Math.round(threshold * 100);
    return sweep[Math.min(Math.max(idx, 0), sweep.length - 1)];
  }, [threshold]);

  return (
    <section className="px-6 py-24">
      <div className="mx-auto max-w-5xl">
        <div className="mx-auto mb-10 max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Pick your own tradeoff
          </h2>
          <p className="mt-3 text-balance text-slate-400">
            The threshold is yours to set. Higher = fewer hallucinations, but the system
            abstains more often. Real numbers from the clean-subset eval set (708 rows).
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-4 sm:p-8">
          <div className="h-72 w-full sm:h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sweep} margin={{ top: 8, right: 12, left: -12, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="threshold"
                  type="number"
                  domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  tickFormatter={pct}
                  stroke="#64748b"
                  fontSize={12}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={pct}
                  stroke="#64748b"
                  fontSize={12}
                  width={44}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
                  formatter={(value) => <span className="text-slate-400">{value}</span>}
                />
                <ReferenceLine
                  x={current.threshold}
                  stroke="#e2e8f0"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                />
                <Line
                  type="monotone"
                  dataKey="abstention_rate"
                  name="Abstention rate"
                  stroke="#94a3b8"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="hallucination_rate_of_answered"
                  name="Hallucination rate (of answered)"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="coverage_of_answerable"
                  name="Coverage of answerable"
                  stroke="#1ab173"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Slider */}
          <div className="mt-6">
            <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
              <span>0.00</span>
              <span className="font-mono text-sm font-semibold text-slate-200">
                threshold = {threshold.toFixed(2)}
              </span>
              <span>1.00</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-brand-400"
              aria-label="Abstention threshold"
            />
          </div>

          {/* Presets */}
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setThreshold(p.value)}
                className={`rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors ${
                  Math.abs(threshold - p.value) < 0.005
                    ? "border-brand-500 bg-brand-500/15 text-brand-300"
                    : "border-slate-700 bg-slate-900/60 text-slate-400 hover:border-slate-600 hover:text-slate-200"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Dynamic sentence */}
          <p className="mt-6 text-balance text-center text-sm leading-relaxed text-slate-300 sm:text-base">
            At threshold{" "}
            <span className="font-mono font-semibold text-white">{threshold.toFixed(2)}</span>,
            the system answers{" "}
            <span className="font-mono font-semibold text-brand-400">
              {pct1(current.coverage_of_answerable)}
            </span>{" "}
            of answerable questions with a{" "}
            <span className="font-mono font-semibold text-danger-400">
              {pct1(current.hallucination_rate_of_answered)}
            </span>{" "}
            hallucination rate.
          </p>
        </div>
      </div>
    </section>
  );
}
