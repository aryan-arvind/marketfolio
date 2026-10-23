import React from 'react';
import { CheckCircle2, RefreshCw, Clock, Terminal, Activity, Layers, ShieldCheck, TrendingUp } from 'lucide-react';

const STEPS = [
  {
    step: '1/5',
    title: 'Data Ingestion & Anomaly Cleaning',
    subtitle: 'NSE trading calendar alignment, dividend normalization & jump filter (|r| > 3σ)',
    icon: Layers,
  },
  {
    step: '2/5',
    title: 'Stationarity & Econometric Diagnostics',
    subtitle: 'ADF & KPSS dual unit root testing, Ljung-Box Q-test, STL Loess trend/seasonal decomposition',
    icon: Activity,
  },
  {
    step: '3/5',
    title: 'GARCH Volatility Modeling & Bootstrap Cone',
    subtitle: 'GARCH(1,1) conditional variance fitting, 500-path empirical residual bootstrap simulation',
    icon: TrendingUp,
  },
  {
    step: '4/5',
    title: 'Benchmark Sensitivity & Macro VAR System',
    subtitle: 'ARMA-error regression vs ^NSEI, Vector Autoregression (VAR) & Granger causality',
    icon: ShieldCheck,
  },
  {
    step: '5/5',
    title: 'Rolling-Origin Out-of-Sample Backtesting',
    subtitle: 'Expanding-window forecast evaluation across multiple origins & MASE metric vs naive baseline',
    icon: Terminal,
  },
];

export default function CalibrationProgress({ assetInput, elapsedSeconds, stepIndex }) {
  const currentStep = Math.min(stepIndex, STEPS.length - 1);
  const progressPct = Math.min(96, Math.max(12, (currentStep / (STEPS.length - 1)) * 90 + Math.min(10, elapsedSeconds * 0.4)));

  return (
    <div className="flex-1 flex items-center justify-center p-4 select-none">
      <div className="w-full max-w-xl bg-[#131722] border border-[#242936] rounded-lg shadow-2xl p-6 relative overflow-hidden">
        {/* Glow accent */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#06b6d4] to-transparent opacity-80" />

        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-[#1e222d] mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-[#06b6d4]/10 border border-[#06b6d4]/30 flex items-center justify-center text-[#06b6d4]">
              <RefreshCw className="animate-spin" size={16} />
            </div>
            <div>
              <div className="text-sm font-bold text-white font-mono tracking-tight flex items-center gap-2">
                <span>Calibrating Econometric Pipeline</span>
                <span className="text-xs px-1.5 py-0.2 rounded bg-[#06b6d4]/10 text-[#06b6d4] border border-[#06b6d4]/20">
                  {STEPS[currentStep].step}
                </span>
              </div>
              <div className="text-[11px] text-[#94a3b8]">
                Target Instrument: <span className="text-[#38bdf8] font-mono font-semibold">{assetInput}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1.5 text-xs font-mono text-[#64748b] bg-[#0c1017] px-2.5 py-1 rounded border border-[#1e222d]">
            <Clock size={12} className="text-[#06b6d4]" />
            <span>{elapsedSeconds}s elapsed</span>
          </div>
        </div>

        {/* Top Progress bar */}
        <div className="w-full bg-[#0c1017] h-1.5 rounded-full overflow-hidden mb-5 border border-[#1e222d]">
          <div
            className="bg-gradient-to-r from-[#089981] via-[#06b6d4] to-[#38bdf8] h-full transition-all duration-500 rounded-full"
            style={{ width: `${progressPct}%` }}
          />
        </div>

        {/* 5 Distinct Live Steps Checklist */}
        <div className="space-y-2.5">
          {STEPS.map((s, idx) => {
            const isCompleted = idx < currentStep;
            const isCurrent = idx === currentStep;
            const isPending = idx > currentStep;
            const Icon = s.icon;

            return (
              <div
                key={s.step}
                className={`p-2.5 rounded-md border transition-all flex items-start gap-3 ${
                  isCurrent
                    ? 'bg-[#182030] border-[#06b6d4]/50 shadow-sm'
                    : isCompleted
                    ? 'bg-[#0c1017]/50 border-[#1e222d]'
                    : 'bg-[#0c1017]/20 border-transparent opacity-40'
                }`}
              >
                {/* Status Indicator */}
                <div className="mt-0.5 shrink-0">
                  {isCompleted ? (
                    <CheckCircle2 size={16} className="text-[#089981]" />
                  ) : isCurrent ? (
                    <RefreshCw size={16} className="text-[#06b6d4] animate-spin" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-[#242936] text-[10px] font-mono flex items-center justify-center text-[#64748b]">
                      {idx + 1}
                    </div>
                  )}
                </div>

                {/* Step Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-semibold font-mono tracking-tight ${
                        isCurrent
                          ? 'text-[#06b6d4]'
                          : isCompleted
                          ? 'text-[#e2e8f0]'
                          : 'text-[#64748b]'
                      }`}
                    >
                      {s.step}: {s.title}
                    </span>
                    {isCurrent && (
                      <span className="text-[10px] font-mono uppercase tracking-wider text-[#06b6d4] animate-pulse">
                        Calculating...
                      </span>
                    )}
                    {isCompleted && (
                      <span className="text-[10px] font-mono text-[#089981]">
                        Done
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-[#94a3b8] mt-0.5 leading-snug line-clamp-2">
                    {s.subtitle}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer Note */}
        <div className="mt-4 pt-3 border-t border-[#1e222d] flex items-center justify-between text-[11px] text-[#64748b] font-mono">
          <span>ATSA Econometric Engine (statsmodels + arch)</span>
          <span>Zero Look-Ahead Bias Enforced</span>
        </div>
      </div>
    </div>
  );
}
