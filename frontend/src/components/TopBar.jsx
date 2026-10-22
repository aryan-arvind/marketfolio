import React, { useState } from 'react';
import { Search, SlidersHorizontal, Play, RefreshCw, Terminal, CheckCircle2 } from 'lucide-react';

export default function TopBar({
  assetMode,
  setAssetMode,
  assetInput,
  setAssetInput,
  benchmarkSymbol,
  setBenchmarkSymbol,
  horizon,
  setHorizon,
  maxOrigins,
  setMaxOrigins,
  forceRefresh,
  setForceRefresh,
  onRunAnalysis,
  isLoading,
  meta,
}) {
  const [showSettings, setShowSettings] = useState(false);

  const stockPills = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'POWERGRID.NS', '^NSEI'];
  const fundPills = ['Nifty 50 Index Fund Direct Growth', 'Parag Parikh Flexi Cap Direct Growth'];

  const pills = assetMode === 'Indian Stock / Index' ? stockPills : fundPills;

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      onRunAnalysis();
    }
  };

  return (
    <header className="bg-[#0c1017] border-b border-[#1e222d] px-4 py-2 flex flex-wrap items-center justify-between gap-3 text-sm select-none">
      {/* 1. Left: Branding & Instrument Selector */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 pr-2 border-r border-[#1e222d]">
          <div className="w-6 h-6 rounded bg-[#06b6d4]/10 border border-[#06b6d4]/30 flex items-center justify-center text-[#06b6d4]">
            <Terminal size={14} />
          </div>
          <span className="font-bold text-white text-base tracking-tight font-sans">
            MarketFolio
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold tracking-wider uppercase bg-[#1e222d] text-[#06b6d4] border border-[#242936]">
            TERMINAL
          </span>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center bg-[#131722] p-0.5 rounded border border-[#242936] text-xs">
          <button
            onClick={() => setAssetMode('Indian Stock / Index')}
            className={`px-2.5 py-1 rounded transition-colors ${
              assetMode === 'Indian Stock / Index'
                ? 'bg-[#1e293b] text-white font-medium shadow-sm'
                : 'text-[#64748b] hover:text-white'
            }`}
          >
            NSE Stock / Index
          </button>
          <button
            onClick={() => setAssetMode('Indian Mutual Fund')}
            className={`px-2.5 py-1 rounded transition-colors ${
              assetMode === 'Indian Mutual Fund'
                ? 'bg-[#1e293b] text-white font-medium shadow-sm'
                : 'text-[#64748b] hover:text-white'
            }`}
          >
            Mutual Fund (AMFI)
          </button>
        </div>
      </div>

      {/* 2. Center: Quick Search & Pills */}
      <div className="flex items-center gap-2 flex-1 max-w-xl">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#64748b]" size={14} />
          <input
            type="text"
            value={assetInput}
            onChange={(e) => setAssetInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={assetMode === 'Indian Stock / Index' ? "Enter Yahoo Ticker (e.g. RELIANCE.NS, ^NSEI)" : "Enter AMFI Scheme Name"}
            className="w-full bg-[#131722] border border-[#242936] focus:border-[#06b6d4] focus:outline-none rounded pl-8 pr-3 py-1.5 text-xs text-white placeholder-[#475569] font-mono"
          />
        </div>

        {/* Pill recommendations */}
        <div className="hidden lg:flex items-center gap-1">
          {pills.slice(0, 3).map((p) => (
            <button
              key={p}
              onClick={() => {
                setAssetInput(p);
                // Trigger auto run
                setTimeout(() => onRunAnalysis(p), 50);
              }}
              className="px-2 py-1 bg-[#131722] hover:bg-[#1e222d] border border-[#242936] rounded text-[11px] text-[#94a3b8] hover:text-[#38bdf8] font-mono transition-colors truncate max-w-[120px]"
            >
              {p.replace('.NS', '')}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Right: Settings Dropdown & Run Button */}
      <div className="flex items-center gap-2">
        {/* Settings Toggle */}
        <div className="relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-1.5 rounded border transition-colors flex items-center gap-1.5 text-xs ${
              showSettings
                ? 'bg-[#1e293b] border-[#06b6d4] text-[#06b6d4]'
                : 'bg-[#131722] border-[#242936] text-[#94a3b8] hover:text-white hover:bg-[#1e222d]'
            }`}
            title="Configure Horizon & Backtest Depth"
          >
            <SlidersHorizontal size={14} />
            <span className="hidden md:inline font-mono">{horizon}D | {maxOrigins} Origins</span>
          </button>

          {/* Settings Popover */}
          {showSettings && (
            <div className="absolute right-0 top-full mt-2 w-72 bg-[#131722] border border-[#242936] rounded-md shadow-xl p-3 z-50 text-xs">
              <div className="font-semibold text-white mb-2.5 pb-1 border-b border-[#242936] flex items-center justify-between">
                <span>Model Calibration Parameters</span>
                <button
                  onClick={() => setShowSettings(false)}
                  className="text-[#64748b] hover:text-white"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-[#94a3b8] mb-1">
                    <span>Forecast Horizon</span>
                    <span className="text-[#38bdf8] font-mono">{horizon} Days</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="60"
                    step="1"
                    value={horizon}
                    onChange={(e) => setHorizon(Number(e.target.value))}
                    className="w-full accent-[#06b6d4]"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[#94a3b8] mb-1">
                    <span>Benchmark Symbol</span>
                  </div>
                  <input
                    type="text"
                    value={benchmarkSymbol}
                    onChange={(e) => setBenchmarkSymbol(e.target.value)}
                    className="w-full bg-[#0c1017] border border-[#242936] rounded px-2 py-1 text-xs text-white font-mono"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-[#94a3b8] mb-1">
                    <span>Backtest Depth</span>
                    <span className="text-[#38bdf8] font-mono">{maxOrigins} Origins</span>
                  </div>
                  <div className="grid grid-cols-5 gap-1">
                    {[6, 12, 18, 24, 36].map((depth) => (
                      <button
                        key={depth}
                        onClick={() => setMaxOrigins(depth)}
                        className={`py-1 rounded text-center font-mono text-[11px] border ${
                          maxOrigins === depth
                            ? 'bg-[#06b6d4] text-[#0c1017] font-bold border-[#06b6d4]'
                            : 'bg-[#0c1017] text-[#94a3b8] border-[#242936] hover:border-[#475569]'
                        }`}
                      >
                        {depth}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-1 border-t border-[#242936]">
                  <input
                    type="checkbox"
                    id="forceRefresh"
                    checked={forceRefresh}
                    onChange={(e) => setForceRefresh(e.target.checked)}
                    className="accent-[#06b6d4] rounded"
                  />
                  <label htmlFor="forceRefresh" className="text-[#94a3b8] text-[11px] cursor-pointer">
                    Bypass local cache (Force live API fetch)
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Run Analysis Button */}
        <button
          onClick={() => onRunAnalysis()}
          disabled={isLoading}
          className="bg-[#06b6d4] hover:bg-[#0891b2] text-[#0c1017] font-semibold px-3.5 py-1.5 rounded text-xs flex items-center gap-1.5 shadow-sm transition-all disabled:opacity-50"
        >
          {isLoading ? (
            <>
              <RefreshCw className="animate-spin" size={13} />
              <span>Calibrating...</span>
            </>
          ) : (
            <>
              <Play size={13} fill="currentColor" />
              <span>Run Analysis</span>
            </>
          )}
        </button>

        {/* Data Status Chip */}
        {meta && (
          <div className="hidden xl:flex items-center gap-1.5 px-2 py-1 bg-[#131722] border border-[#242936] rounded text-[11px] text-[#64748b]">
            <CheckCircle2 size={12} className="text-[#089981]" />
            <span>{meta.source || 'Public API'}</span>
          </div>
        )}
      </div>
    </header>
  );
}
