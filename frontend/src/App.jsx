import React, { useState, useEffect } from 'react';
import TopBar from './components/TopBar';
import HeaderStatsStrip from './components/HeaderStatsStrip';
import TradingViewChart from './components/TradingViewChart';
import DeskTabs from './components/DeskTabs';
import { AlertCircle, RefreshCw, Terminal } from 'lucide-react';

export default function App() {
  const [assetMode, setAssetMode] = useState('Indian Stock / Index');
  const [assetInput, setAssetInput] = useState('RELIANCE.NS');
  const [benchmarkSymbol, setBenchmarkSymbol] = useState('^NSEI');
  const [horizon, setHorizon] = useState(10);
  const [maxOrigins, setMaxOrigins] = useState(3);
  const [forceRefresh, setForceRefresh] = useState(false);

  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const runAnalysis = async (customSymbol) => {
    const symbolToRun = customSymbol || assetInput;
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset_mode: assetMode,
          asset_input: symbolToRun,
          benchmark_symbol: benchmarkSymbol,
          horizon: Number(horizon),
          force_refresh: forceRefresh,
          max_origins: Number(maxOrigins),
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${res.status}`);
      }

      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err.message || 'Failed to complete quantitative analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!data) return;
    try {
      const res = await fetch('/api/report/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset_mode: assetMode,
          asset_input: data.symbol_code,
          benchmark_symbol: benchmarkSymbol,
          horizon: Number(horizon),
          force_refresh: false,
          max_origins: Number(maxOrigins),
        }),
      });

      if (!res.ok) throw new Error('Failed to generate report file.');

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `MarketFolio_Memo_${data.symbol_code}.md`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Report download error: ${err.message}`);
    }
  };

  // Initial load
  useEffect(() => {
    runAnalysis('RELIANCE.NS');
  }, []);

  return (
    <div className="min-h-screen bg-[#0c1017] text-[#e5e7eb] flex flex-col font-sans">
      {/* 1. TradingView Top Control Bar */}
      <TopBar
        assetMode={assetMode}
        setAssetMode={setAssetMode}
        assetInput={assetInput}
        setAssetInput={setAssetInput}
        benchmarkSymbol={benchmarkSymbol}
        setBenchmarkSymbol={setBenchmarkSymbol}
        horizon={horizon}
        setHorizon={setHorizon}
        maxOrigins={maxOrigins}
        setMaxOrigins={setMaxOrigins}
        forceRefresh={forceRefresh}
        setForceRefresh={setForceRefresh}
        onRunAnalysis={runAnalysis}
        isLoading={isLoading}
        meta={data?.meta}
      />

      {/* 2. Error Display (if any) */}
      {error && (
        <div className="mx-4 my-2 p-3 bg-[#f23645]/10 border border-[#f23645]/30 rounded text-xs text-[#fca5a5] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle size={15} className="text-[#f23645]" />
            <span><b>Execution Error:</b> {error}</span>
          </div>
          <button
            onClick={() => runAnalysis()}
            className="px-2 py-0.5 bg-[#f23645]/20 hover:bg-[#f23645]/30 text-white rounded text-[11px] font-medium"
          >
            Retry
          </button>
        </div>
      )}

      {/* 3. Main Workspace */}
      {data ? (
        <main className="flex-1 flex flex-col p-3 space-y-3">
          {/* Header Stats Strip */}
          <HeaderStatsStrip data={data} />

          {/* Large Central Dominant TradingView Chart */}
          <TradingViewChart data={data} horizon={horizon} />

          {/* Lower Section: 8 Analytical Secondary Tabs */}
          <DeskTabs data={data} onDownloadReport={handleDownloadReport} />
        </main>
      ) : (
        /* Standby Skeleton */
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center max-w-sm">
            <div className="w-12 h-12 rounded-full bg-[#131722] border border-[#242936] flex items-center justify-center mx-auto mb-4 text-[#06b6d4]">
              <RefreshCw className={isLoading ? "animate-spin" : ""} size={20} />
            </div>
            <h2 className="text-base font-bold text-white mb-1">
              {isLoading ? "Calibrating Risk & Macro Engine..." : "MarketFolio Terminal Ready"}
            </h2>
            <p className="text-xs text-[#64748b] mb-4">
              {isLoading
                ? `Running out-of-sample backtests, GARCH variance bootstrap, and macro lead-lag transmission for ${assetInput}...`
                : "Select an equity ticker or mutual fund from the top bar to calibrate models."}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
