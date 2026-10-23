import React, { useState, useEffect } from 'react';
import TopBar from './components/TopBar';
import HeaderStatsStrip from './components/HeaderStatsStrip';
import TradingViewChart from './components/TradingViewChart';
import DeskTabs from './components/DeskTabs';
import CalibrationProgress from './components/CalibrationProgress';
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
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    let timer;
    if (isLoading) {
      setElapsedSeconds(0);
      setStepIndex(0);
      const start = Date.now();
      timer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - start) / 1000);
        setElapsedSeconds(elapsed);
        if (elapsed < 3) {
          setStepIndex(0);
        } else if (elapsed < 7) {
          setStepIndex(1);
        } else if (elapsed < 12) {
          setStepIndex(2);
        } else if (elapsed < 17) {
          setStepIndex(3);
        } else {
          setStepIndex(4);
        }
      }, 1000);
    } else {
      setElapsedSeconds(0);
      setStepIndex(0);
    }
    return () => clearInterval(timer);
  }, [isLoading]);

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

  // Don't auto-calculate on initial mount - let the user land on the clean terminal workspace!

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
      {isLoading ? (
        <CalibrationProgress
          assetInput={assetInput}
          elapsedSeconds={elapsedSeconds}
          stepIndex={stepIndex}
        />
      ) : data ? (
        <main className="flex-1 flex flex-col p-3 space-y-3">
          {/* Header Stats Strip */}
          <HeaderStatsStrip data={data} />

          {/* Large Central Dominant TradingView Chart */}
          <TradingViewChart data={data} horizon={horizon} />

          {/* Lower Section: 8 Analytical Secondary Tabs */}
          <DeskTabs data={data} onDownloadReport={handleDownloadReport} />
        </main>
      ) : (
        /* Clean Generic Landing Page / Desk Standby Workspace */
        <div className="flex-1 flex flex-col items-center justify-center p-6 max-w-4xl mx-auto w-full">
          <div className="text-center space-y-3 mb-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#131722] border border-[#242936] text-[11px] text-[#06b6d4] font-mono">
              <Terminal size={13} />
              <span>MARKETFOLIO QUANTITATIVE DESK</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Institutional Time-Series Analytics & Risk Terminal
            </h1>
            <p className="text-sm text-[#94a3b8] max-w-xl mx-auto leading-relaxed">
              Real historical econometric research workspace for Indian equities and mutual funds. Select an instrument or click any quick-start asset below to begin.
            </p>
          </div>

          {/* Quick Start Instruments Grid */}
          <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-3 mb-8">
            {[
              {
                sym: 'RELIANCE.NS',
                name: 'Reliance Industries',
                sector: 'Energy & Conglomerate',
                mode: 'Indian Stock / Index',
              },
              {
                sym: 'TCS.NS',
                name: 'Tata Consultancy Services',
                sector: 'Information Technology',
                mode: 'Indian Stock / Index',
              },
              {
                sym: 'HDFCBANK.NS',
                name: 'HDFC Bank Ltd',
                sector: 'Banking & Financials',
                mode: 'Indian Stock / Index',
              },
              {
                sym: '^NSEI',
                name: 'NIFTY 50 Index',
                sector: 'Domestic Benchmark',
                mode: 'Indian Stock / Index',
              },
              {
                sym: 'POWERGRID.NS',
                name: 'Power Grid Corp',
                sector: 'Utilities & Power',
                mode: 'Indian Stock / Index',
              },
              {
                sym: 'Parag Parikh Flexi Cap Direct Growth',
                name: 'Parag Parikh Flexi Cap Fund',
                sector: 'Mutual Fund (AMFI)',
                mode: 'Indian Mutual Fund',
              },
            ].map((item) => (
              <button
                key={item.sym}
                onClick={() => {
                  setAssetMode(item.mode);
                  setAssetInput(item.sym);
                  runAnalysis(item.sym);
                }}
                className="bg-[#131722] hover:bg-[#181c27] border border-[#242936] hover:border-[#06b6d4]/50 p-4 rounded text-left transition-all duration-150 group cursor-pointer"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-sm font-bold text-white group-hover:text-[#06b6d4]">
                    {item.sym.replace('.NS', '')}
                  </span>
                  <span className="text-[10px] text-[#64748b] bg-[#1e222d] px-1.5 py-0.5 rounded">
                    {item.mode === 'Indian Mutual Fund' ? 'MF' : 'NSE'}
                  </span>
                </div>
                <div className="text-xs text-[#cbd5e1] font-medium">{item.name}</div>
                <div className="text-[11px] text-[#64748b] mt-1">{item.sector}</div>
              </button>
            ))}
          </div>

          {/* Quantitative Capabilities Strip */}
          <div className="w-full bg-[#131722]/60 border border-[#242936] rounded p-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-left">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] font-mono">Volatility</div>
              <div className="text-xs font-semibold text-white mt-0.5">GARCH Risk Cone</div>
              <div className="text-[11px] text-[#94a3b8] mt-0.5">Residual bootstrap 90% bounds</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] font-mono">Trend Analysis</div>
              <div className="text-xs font-semibold text-white mt-0.5">STL Loess Filter</div>
              <div className="text-[11px] text-[#94a3b8] mt-0.5">Macro trend vs calendar seasonality</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] font-mono">Macro Transmission</div>
              <div className="text-xs font-semibold text-white mt-0.5">Vector Autoregression</div>
              <div className="text-[11px] text-[#94a3b8] mt-0.5">Granger causality from S&P, VIX, Oil</div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#64748b] font-mono">Backtesting</div>
              <div className="text-xs font-semibold text-white mt-0.5">Rolling Out-of-Sample</div>
              <div className="text-[11px] text-[#94a3b8] mt-0.5">Zero look-ahead bias audit & MASE</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
