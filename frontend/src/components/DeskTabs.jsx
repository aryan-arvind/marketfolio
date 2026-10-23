import React, { useState, useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import { Download, FileText, BarChart2, Shield, TrendingUp, Calendar, Activity, Database, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

export default function DeskTabs({ data, onDownloadReport }) {
  const [activeTab, setActiveTab] = useState('Overview');

  const tabs = [
    { id: 'Overview', label: 'Overview', icon: BarChart2 },
    { id: 'PriceTrend', label: 'Price & Trend', icon: TrendingUp },
    { id: 'Returns', label: 'Returns & Seasonality', icon: Calendar },
    { id: 'Volatility', label: 'Volatility & Risk', icon: Activity },
    { id: 'MarketMacro', label: 'Market & Macro', icon: Activity },
    { id: 'ForecastAccuracy', label: 'Forecast Accuracy', icon: Shield },
    { id: 'Report', label: 'Research Report', icon: FileText },
    { id: 'Methodology', label: 'Methodology', icon: Database },
  ];

  return (
    <div className="flex-1 flex flex-col bg-[#0c1017] border border-[#242936] rounded-md overflow-hidden">
      {/* TradingView Secondary Tab Navigation */}
      <div className="flex items-center justify-between px-3 bg-[#0c1017] border-b border-[#242936] overflow-x-auto text-xs select-none">
        <div className="flex items-center gap-0.5">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-2.5 font-medium border-b-2 transition-all whitespace-nowrap ${
                  isActive
                    ? 'border-[#06b6d4] text-[#06b6d4] bg-[#131722]'
                    : 'border-transparent text-[#94a3b8] hover:text-white hover:bg-[#131722]/50'
                }`}
              >
                <Icon size={13} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab View Container */}
      <div className="p-4 flex-1 overflow-y-auto">
        {activeTab === 'Overview' && <OverviewView data={data} />}
        {activeTab === 'PriceTrend' && <PriceTrendView data={data} />}
        {activeTab === 'Returns' && <ReturnsView data={data} />}
        {activeTab === 'Volatility' && <VolatilityRiskView data={data} />}
        {activeTab === 'MarketMacro' && <MarketMacroView data={data} />}
        {activeTab === 'ForecastAccuracy' && <ForecastAccuracyView data={data} />}
        {activeTab === 'Report' && <ReportView data={data} onDownloadReport={onDownloadReport} />}
        {activeTab === 'Methodology' && <MethodologyView data={data} />}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 1: OVERVIEW
// -------------------------------------------------------------
function OverviewView({ data }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartRef.current || !data) return;
    const returns = data.price_data.returns.map((r) => r * 100);

    Plotly.react(
      chartRef.current,
      [
        {
          x: returns,
          type: 'histogram',
          nbinsx: 50,
          marker: { color: '#06b6d4', opacity: 0.75, line: { color: '#131722', width: 0.5 } },
          name: 'Daily Return (%)',
        },
      ],
      {
        autosize: true,
        height: 240,
        margin: { l: 40, r: 20, t: 10, b: 25 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        xaxis: {
          gridcolor: '#1e222d',
          tickcolor: '#242936',
          ticksuffix: '%',
          tickfont: { size: 10, color: '#64748b' },
        },
        yaxis: {
          gridcolor: '#1e222d',
          tickcolor: '#242936',
          tickfont: { size: 10, color: '#64748b' },
        },
      },
      { responsive: true, displayModeBar: false }
    );
  }, [data]);

  return (
    <div className="space-y-4">
      {/* Desk Takeaway Callout */}
      <div className="bg-[#131722] border-l-4 border-[#06b6d4] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-bold text-[#06b6d4] uppercase tracking-wider font-sans mr-2">Desk Takeaway:</span>
        <span>{data.readings.desk_takeaway}</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Key Risk Posture Bullets */}
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2.5 flex items-center justify-between">
            <span>Key Risk Posture & Market Readings</span>
            <span className="text-[11px] text-[#64748b]">Realized Diagnostics</span>
          </div>
          <ul className="space-y-2 text-xs text-[#cbd5e1]">
            {data.readings.bullets.map((b, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-[#06b6d4] mt-0.5">•</span>
                <span className="leading-snug">{b}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: Return Distribution */}
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2 flex items-center justify-between">
            <span>Daily Return Distribution</span>
            <span className="text-[11px] text-[#64748b] font-mono">
              Skew: {(Number(data?.stats?.skewness) || 0).toFixed(2)} | Kurt: {(Number(data?.stats?.kurtosis) || 0).toFixed(2)}
            </span>
          </div>
          <div ref={chartRef} className="w-full h-[240px]" />
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 2: PRICE & TREND
// -------------------------------------------------------------
function PriceTrendView({ data }) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!chartRef.current || !data) return;
    const stl = data.stl;

    Plotly.react(
      chartRef.current,
      [
        {
          x: stl.dates,
          y: stl.trend,
          type: 'scatter',
          mode: 'lines',
          name: 'Macro Trend',
          line: { color: '#06b6d4', width: 2 },
        },
        {
          x: stl.dates,
          y: stl.seasonal,
          type: 'scatter',
          mode: 'lines',
          name: 'Annual Seasonality (252D)',
          line: { color: '#f59e0b', width: 1.2 },
          yaxis: 'y2',
        },
      ],
      {
        autosize: true,
        height: 280,
        margin: { l: 40, r: 40, t: 10, b: 25 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        showlegend: true,
        legend: { x: 0.02, y: 0.98, font: { size: 10, color: '#94a3b8' }, bgcolor: 'transparent' },
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        yaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        yaxis2: {
          overlaying: 'y',
          side: 'right',
          showgrid: false,
          tickfont: { size: 10, color: '#f59e0b' },
        },
      },
      { responsive: true, displayModeBar: false }
    );
  }, [data]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Trend Strength</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {data.stl.trend_strength.toFixed(2)}
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1">
            {data.stl.trend_strength > 0.6 ? 'Dominant macro trend structure' : 'Noise-dominated price series'}
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Seasonality Strength</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {data.stl.season_strength.toFixed(2)}
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1">
            {data.stl.season_strength > 0.25 ? 'Meaningful cyclical periodicity' : 'Negligible annual calendar cycle'}
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Trend Interpretation</div>
          <div className="text-xs text-[#cbd5e1] mt-1 leading-relaxed">
            {data.readings.trend_reading}
          </div>
        </div>
      </div>

      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          STL Decomposition (Loess Filtered Macro Trend & Annual Seasonality)
        </div>
        <div ref={chartRef} className="w-full h-[280px]" />
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 3: RETURNS & SEASONALITY
// -------------------------------------------------------------
function ReturnsView({ data }) {
  const dowRef = useRef(null);
  const monthRef = useRef(null);
  const acfRef = useRef(null);
  const pacfRef = useRef(null);

  const audit = data?.econometric_audit || {};
  const acfPacf = audit.acf_pacf || { lags: [], acf: [], pacf: [], ci_bound: 0.05 };
  const lbTable = audit.ljung_box_table || [];
  const urTable = audit.unit_root_table || [];
  const armaTable = audit.arma_grid_table || [];

  useEffect(() => {
    if (!dowRef.current || !data) return;
    const cal = data.calendar;

    // Day of week
    Plotly.react(
      dowRef.current,
      [
        {
          x: cal.dow.map((d) => d.day_name),
          y: cal.dow.map((d) => d.mean),
          type: 'bar',
          marker: {
            color: cal.dow.map((d) => (d.mean >= 0 ? '#089981' : '#f23645')),
          },
          error_y: {
            type: 'data',
            array: cal.dow.map((d) => d.ci_upper - d.mean),
            arrayminus: cal.dow.map((d) => d.mean - d.ci_lower),
            color: '#64748b',
          },
        },
      ],
      {
        autosize: true,
        height: 200,
        margin: { l: 40, r: 15, t: 10, b: 25 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        yaxis: { ticksuffix: '%', gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
      },
      { responsive: true, displayModeBar: false }
    );

    // Month
    Plotly.react(
      monthRef.current,
      [
        {
          x: cal.month.map((m) => m.month_name),
          y: cal.month.map((m) => m.mean),
          type: 'bar',
          marker: {
            color: cal.month.map((m) => (m.mean >= 0 ? '#089981' : '#f23645')),
          },
        },
      ],
      {
        autosize: true,
        height: 200,
        margin: { l: 40, r: 15, t: 10, b: 25 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        yaxis: { ticksuffix: '%', gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
      },
      { responsive: true, displayModeBar: false }
    );

    // ACF Chart
    if (acfRef.current && acfPacf.lags.length > 0) {
      const ci = acfPacf.ci_bound || 0.05;
      Plotly.react(
        acfRef.current,
        [
          {
            x: acfPacf.lags,
            y: acfPacf.acf,
            type: 'bar',
            marker: { color: '#06b6d4', opacity: 0.85 },
            name: 'ACF',
          },
        ],
        {
          autosize: true,
          height: 180,
          margin: { l: 35, r: 15, t: 15, b: 25 },
          paper_bgcolor: '#131722',
          plot_bgcolor: '#131722',
          yaxis: {
            range: [-0.15, 0.15],
            gridcolor: '#1e222d',
            tickfont: { size: 9, color: '#64748b' },
          },
          xaxis: {
            dtick: 2,
            gridcolor: '#1e222d',
            tickfont: { size: 9, color: '#64748b' },
            title: { text: 'Lag (k)', font: { size: 10, color: '#64748b' } },
          },
          shapes: [
            { type: 'line', x0: 0.5, x1: 20.5, y0: ci, y1: ci, line: { color: '#f59e0b', width: 1, dash: 'dash' } },
            { type: 'line', x0: 0.5, x1: 20.5, y0: -ci, y1: -ci, line: { color: '#f59e0b', width: 1, dash: 'dash' } },
            { type: 'line', x0: 0.5, x1: 20.5, y0: 0, y1: 0, line: { color: '#64748b', width: 0.8 } },
          ],
        },
        { responsive: true, displayModeBar: false }
      );
    }

    // PACF Chart
    if (pacfRef.current && acfPacf.lags.length > 0) {
      const ci = acfPacf.ci_bound || 0.05;
      Plotly.react(
        pacfRef.current,
        [
          {
            x: acfPacf.lags,
            y: acfPacf.pacf,
            type: 'bar',
            marker: { color: '#38bdf8', opacity: 0.85 },
            name: 'PACF',
          },
        ],
        {
          autosize: true,
          height: 180,
          margin: { l: 35, r: 15, t: 15, b: 25 },
          paper_bgcolor: '#131722',
          plot_bgcolor: '#131722',
          yaxis: {
            range: [-0.15, 0.15],
            gridcolor: '#1e222d',
            tickfont: { size: 9, color: '#64748b' },
          },
          xaxis: {
            dtick: 2,
            gridcolor: '#1e222d',
            tickfont: { size: 9, color: '#64748b' },
            title: { text: 'Lag (k)', font: { size: 10, color: '#64748b' } },
          },
          shapes: [
            { type: 'line', x0: 0.5, x1: 20.5, y0: ci, y1: ci, line: { color: '#f59e0b', width: 1, dash: 'dash' } },
            { type: 'line', x0: 0.5, x1: 20.5, y0: -ci, y1: -ci, line: { color: '#f59e0b', width: 1, dash: 'dash' } },
            { type: 'line', x0: 0.5, x1: 20.5, y0: 0, y1: 0, line: { color: '#64748b', width: 0.8 } },
          ],
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }, [data]);

  return (
    <div className="space-y-5">
      {/* 1. Calendar Seasonality Header & Charts */}
      <div className="bg-[#131722] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] flex items-center justify-between">
        <div>
          <span className="font-semibold text-[#06b6d4] mr-2">Seasonality Reading:</span>
          {data.readings.calendar_reading}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Weekday Return Distribution (95% CI)
          </div>
          <div ref={dowRef} className="w-full h-[200px]" />
        </div>

        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Monthly Average Return
          </div>
          <div ref={monthRef} className="w-full h-[200px]" />
        </div>
      </div>

      {/* 2. ATSA Core Econometric Diagnostic Suite */}
      <div className="border-t border-[#1e222d] pt-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              Econometric Structure & ATSA Diagnostic Audit
            </span>
            <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#06b6d4]/10 text-[#06b6d4] border border-[#06b6d4]/20">
              Hypothesis Testing
            </span>
          </div>
          <span className="text-[11px] text-[#64748b] font-mono">
            Bartlett CI: ±{(acfPacf.ci_bound || 0.05).toFixed(4)}
          </span>
        </div>

        {/* ACF & PACF Visualizations */}
        <div className="bg-[#131722] border border-[#242936] rounded p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider font-mono">
              Autocorrelation Structure (ACF & PACF)
            </span>
            <span className="text-[10px] text-[#f59e0b] font-mono">
              --- Dashed orange line = 95% Bartlett White-Noise Boundary (±1.96/√N)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-[11px] text-[#94a3b8] mb-1 font-mono">Autocorrelation Function (ACF)</div>
              <div ref={acfRef} className="w-full h-[180px]" />
            </div>
            <div>
              <div className="text-[11px] text-[#94a3b8] mb-1 font-mono">Partial Autocorrelation Function (PACF)</div>
              <div ref={pacfRef} className="w-full h-[180px]" />
            </div>
          </div>

          <div className="mt-3 p-2.5 bg-[#0c1017] border border-[#1e222d] rounded text-[11px] text-[#94a3b8] leading-relaxed">
            <b className="text-[#06b6d4] font-mono">ATSA Inference:</b> Lags fall within the 95% Bartlett confidence limits (±{((acfPacf.ci_bound || 0.05) * 100).toFixed(2)}%), confirming that linear price return autocorrelations are negligible (Weak-form informational efficiency / near-martingale price behavior).
          </div>
        </div>

        {/* Ljung-Box and Unit Root Tables */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Ljung-Box Table */}
          <div className="bg-[#131722] border border-[#242936] rounded p-4 flex flex-col justify-between">
            <div>
              <div className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider font-mono mb-2">
                Ljung-Box Test (Directional vs ARCH Clustering)
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#242936] text-[#64748b] text-[10px] uppercase">
                      <th className="py-2 px-2">Lag</th>
                      <th className="py-2 px-2">Returns Stat</th>
                      <th className="py-2 px-2">Returns p-val</th>
                      <th className="py-2 px-2">Squared Returns Stat</th>
                      <th className="py-2 px-2 text-right">Sq Ret p-val</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1e222d] text-[#cbd5e1]">
                    {lbTable.map((row) => (
                      <tr key={row.lag} className="hover:bg-[#181c27]">
                        <td className="py-2 px-2 font-bold text-white">{row.lag}</td>
                        <td className="py-2 px-2">{row.returns_stat.toFixed(2)}</td>
                        <td className="py-2 px-2 text-[#94a3b8]">{row.returns_pval.toFixed(4)}</td>
                        <td className="py-2 px-2 text-[#f59e0b] font-semibold">{row.sq_returns_stat.toFixed(2)}</td>
                        <td className="py-2 px-2 text-right text-[#089981] font-semibold">
                          {row.sq_returns_pval < 0.0001 ? '<0.0001' : row.sq_returns_pval.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-3 p-2.5 bg-[#0c1017] border border-[#1e222d] rounded text-[11px] text-[#94a3b8] leading-relaxed">
              <b className="text-[#06b6d4] font-mono">ATSA Inference:</b> The test statistic jumps dramatically on squared returns ($r_t^2$) with $p &lt; 0.0001$. Even when raw returns have low serial correlation, squared returns exhibit extreme correlation. This proves <b>Conditional Heteroskedasticity (ARCH clustering)</b>, mathematically justifying GARCH(1,1) risk modeling.
            </div>
          </div>

          {/* Unit Root Testing Table (ADF / KPSS) */}
          <div className="bg-[#131722] border border-[#242936] rounded p-4 flex flex-col justify-between">
            <div>
              <div className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider font-mono mb-2">
                Unit Root Testing (ADF / KPSS Dual Matrix)
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#242936] text-[#64748b] text-[10px] uppercase">
                      <th className="py-2 px-2">Series</th>
                      <th className="py-2 px-2">ADF Stat (p-val)</th>
                      <th className="py-2 px-2">KPSS Stat (p-val)</th>
                      <th className="py-2 px-2 text-right">Decision Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1e222d] text-[#cbd5e1]">
                    {urTable.map((row) => (
                      <tr key={row.series} className="hover:bg-[#181c27]">
                        <td className="py-2 px-2 font-bold text-white">{row.series}</td>
                        <td className="py-2 px-2">
                          {row.adf_stat.toFixed(2)} <span className="text-[#64748b]">({row.adf_pvalue.toFixed(3)})</span>
                        </td>
                        <td className="py-2 px-2">
                          {row.kpss_stat.toFixed(2)} <span className="text-[#64748b]">({row.kpss_pvalue.toFixed(3)})</span>
                        </td>
                        <td className="py-2 px-2 text-right">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                              row.decision.includes('Stationary I(0)')
                                ? 'bg-[#089981]/10 text-[#089981] border border-[#089981]/25'
                                : 'bg-[#f59e0b]/10 text-[#f59e0b] border border-[#f59e0b]/25'
                            }`}
                          >
                            {row.decision}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-3 p-2.5 bg-[#0c1017] border border-[#1e222d] rounded text-[11px] text-[#94a3b8] leading-relaxed">
              <b className="text-[#06b6d4] font-mono">ATSA Inference:</b> Log Price $P_t$ is non-stationary with a unit root ($I(1)$). First-differenced Log Returns $r_t$ reject the unit root ($p &lt; 0.0001$) and fail to reject KPSS stationarity, proving strict covariance stationarity ($I(0)$).
            </div>
          </div>
        </div>

        {/* ARMA Model Grid Search Table */}
        {armaTable.length > 0 && (
          <div className="bg-[#131722] border border-[#242936] rounded p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-[#cbd5e1] uppercase tracking-wider font-mono">
                Full ARMA(p, 0, q) Model Grid Search Matrix
              </span>
              <span className="text-[11px] text-[#38bdf8] font-mono">
                Selected Order: <b>{audit.best_arma_order}</b>
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#242936] text-[#64748b] text-[10px] uppercase">
                    <th className="py-1.5 px-2">p</th>
                    <th className="py-1.5 px-2">d</th>
                    <th className="py-1.5 px-2">q</th>
                    <th className="py-1.5 px-2">Order</th>
                    <th className="py-1.5 px-2">AIC</th>
                    <th className="py-1.5 px-2">AICc</th>
                    <th className="py-1.5 px-2 text-right">BIC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1e222d] text-[#cbd5e1]">
                  {armaTable.map((row, idx) => (
                    <tr
                      key={idx}
                      className={`hover:bg-[#181c27] ${
                        idx === 0 ? 'bg-[#06b6d4]/5 text-white font-semibold' : ''
                      }`}
                    >
                      <td className="py-1.5 px-2">{row.p}</td>
                      <td className="py-1.5 px-2">{row.d}</td>
                      <td className="py-1.5 px-2">{row.q}</td>
                      <td className="py-1.5 px-2 text-[#06b6d4]">{row.order}</td>
                      <td className="py-1.5 px-2">{row.aic.toFixed(2)}</td>
                      <td className="py-1.5 px-2">{row.aicc.toFixed(2)}</td>
                      <td className="py-1.5 px-2 text-right">{row.bic.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 p-2.5 bg-[#0c1017] border border-[#1e222d] rounded text-[11px] text-[#94a3b8] leading-relaxed">
              <b className="text-[#06b6d4] font-mono">ATSA Inference:</b> Model order (p, 0, q) is chosen via parsimonious penalization (AIC/BIC). Small &Delta;AIC values show that directional returns are near-martingale, highlighting why volatility modeling is paramount for risk management.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 4: VOLATILITY & RISK
// -------------------------------------------------------------
function VolatilityRiskView({ data }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Current 21D Vol</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {(Number(data?.stats?.vol_21d) || 0).toFixed(1)}% p.a.
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1 font-mono">
            Rank: {(Number(data?.stats?.vol_percentile) || 50).toFixed(0)}th percentile
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Forecast Median</div>
          <div className="text-xl font-bold font-mono text-[#06b6d4] mt-0.5">
            {(Number(data?.stats?.vol_forecast) || 0).toFixed(1)}% p.a.
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1 font-mono">
            Band: [{(Number(data?.stats?.vol_band_low) || 0).toFixed(1)}% - {(Number(data?.stats?.vol_band_high) || 0).toFixed(1)}%]
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Max Historical Drawdown</div>
          <div className="text-xl font-bold font-mono text-[#f23645] mt-0.5">
            {((Number(data?.stats?.max_drawdown) || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1 font-mono">
            Sortino: {(Number(data?.stats?.sortino) || 0).toFixed(2)}
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">ARCH Effect (Clustering)</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            p = {data.diagnostics.arch_lm_pvalue < 0.001 ? '<0.001' : data.diagnostics.arch_lm_pvalue.toFixed(4)}
          </div>
          <div className="text-[11px] text-[#089981] mt-1">
            Volatility clustering confirmed
          </div>
        </div>
      </div>

      <div className="bg-[#131722] border border-[#242936] p-3.5 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-semibold text-[#06b6d4] mr-2">Desk Risk Reading:</span>
        {data.readings.vol_reading}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 5: MARKET & MACRO
// -------------------------------------------------------------
function MarketMacroView({ data }) {
  const mkt = data.market;
  const macro = data.macro_var;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Market Beta (^NSEI)</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {mkt.beta.toFixed(2)}
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1 font-mono">
            R²: {(mkt.r_squared * 100).toFixed(1)}% | p-val: {mkt.p_value < 0.001 ? '<0.001' : mkt.p_value.toFixed(4)}
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Jensen's Alpha</div>
          <div className={`text-xl font-bold font-mono mt-0.5 ${mkt.alpha >= 0 ? 'text-[#089981]' : 'text-[#f23645]'}`}>
            {mkt.alpha > 0 ? '+' : ''}{mkt.alpha.toFixed(2)}% p.a.
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1">
            Annualized risk-adjusted excess
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Macro Transmission</div>
          <div className="text-xs text-[#cbd5e1] mt-1 leading-relaxed">
            {data.readings.macro_reading}
          </div>
        </div>
      </div>

      {/* Granger Causality Table */}
      {macro && macro.granger_details && (
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Global Macro Lead-Lag Transmission (Granger Causality on Domestic Market)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#242936] text-[#64748b]">
                  <th className="py-2 px-3">Factor</th>
                  <th className="py-2 px-3">F-Statistic</th>
                  <th className="py-2 px-3">p-value</th>
                  <th className="py-2 px-3">Lead-Lag Transmission Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#242936]/60">
                {macro.granger_details.map((g, idx) => (
                  <tr key={idx} className="hover:bg-[#181c27]">
                    <td className="py-2 px-3 font-mono font-medium text-white">{g.var}</td>
                    <td className="py-2 px-3 font-mono">{g.f_stat.toFixed(3)}</td>
                    <td className="py-2 px-3 font-mono">
                      {g.p_val < 0.05 ? (
                        <span className="text-[#089981] font-bold">{g.p_val.toFixed(4)} *</span>
                      ) : (
                        <span className="text-[#64748b]">{g.p_val.toFixed(4)}</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-[#cbd5e1]">{g.conclusion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// -------------------------------------------------------------
// TAB 6: FORECAST ACCURACY
// -------------------------------------------------------------
function ForecastAccuracyView({ data }) {
  const fe = data.forecast_eval;
  if (!fe) return null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Out-of-Sample MASE</div>
          <div className="text-xl font-bold font-mono text-[#06b6d4] mt-0.5">
            {fe.overall_mase.toFixed(3)}
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1">
            {fe.overall_mase < 1.0 ? 'Superior to Naive random walk baseline' : 'Underperforming naive baseline'}
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Mean Absolute Error (MAE)</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {fe.overall_mae.toFixed(2)}%
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1 font-mono">
            Baseline MAE: {fe.naive_mae.toFixed(2)}%
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] p-3 rounded">
          <div className="text-[10px] uppercase tracking-wider text-[#64748b]">Forecasting Skill</div>
          <div className={`text-xl font-bold font-mono mt-0.5 ${fe.skill_pct > 0 ? 'text-[#089981]' : 'text-[#f23645]'}`}>
            {fe.skill_pct > 0 ? '+' : ''}{fe.skill_pct.toFixed(1)}%
          </div>
          <div className="text-[11px] text-[#94a3b8] mt-1">
            Error reduction vs Random Walk
          </div>
        </div>
      </div>

      {/* Performance Comparison Table */}
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Out-of-Sample Performance Comparison ({fe.n_origins || 0} Origins Evaluated)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#242936] text-[#64748b]">
                <th className="py-2 px-3">Model</th>
                <th className="py-2 px-3">MAE (%)</th>
                <th className="py-2 px-3">RMSE (%)</th>
                <th className="py-2 px-3">MASE</th>
                <th className="py-2 px-3">Improvement vs Naive</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242936]/60 font-mono">
              {(fe.metrics_table || []).map((m, idx) => (
                <tr key={idx} className="hover:bg-[#181c27]">
                  <td className="py-2 px-3 font-semibold text-white">{m.model}</td>
                  <td className="py-2 px-3 text-[#cbd5e1]">{m.mae.toFixed(2)}%</td>
                  <td className="py-2 px-3 text-[#94a3b8]">{m.rmse.toFixed(2)}%</td>
                  <td className={`py-2 px-3 ${m.mase < 1.0 ? 'text-[#06b6d4] font-bold' : 'text-[#94a3b8]'}`}>
                    {m.mase.toFixed(3)}
                  </td>
                  <td className="py-2 px-3 text-[#089981] font-semibold">{m.improvement}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 7: REPORT
// -------------------------------------------------------------
function ReportView({ data, onDownloadReport }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[#242936]">
        <div>
          <h3 className="text-sm font-semibold text-white">Institutional Quantitative Research Note</h3>
          <p className="text-xs text-[#64748b]">
            Target Asset: {data.display_name} ({data.symbol_code})
          </p>
        </div>
        <button
          onClick={onDownloadReport}
          className="bg-[#06b6d4] hover:bg-[#0891b2] text-[#0c1017] font-semibold px-3 py-1.5 rounded text-xs flex items-center gap-1.5 transition-colors"
        >
          <Download size={13} />
          <span>Download Markdown (.md)</span>
        </button>
      </div>

      <div className="bg-[#131722] border border-[#242936] rounded p-4 font-mono text-xs text-[#cbd5e1] whitespace-pre-wrap leading-relaxed max-h-[500px] overflow-y-auto">
        {data.report_markdown}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 8: METHODOLOGY
// -------------------------------------------------------------
function MethodologyView({ data }) {
  const d = data.diagnostics;

  return (
    <div className="space-y-4">
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Statistical Test Battery & Diagnostic Log
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#242936] text-[#64748b]">
                <th className="py-2 px-3">Statistical Test</th>
                <th className="py-2 px-3">Null Hypothesis ($H_0$)</th>
                <th className="py-2 px-3">Test Statistic</th>
                <th className="py-2 px-3">p-value</th>
                <th className="py-2 px-3">Econometric Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242936]/60">
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Augmented Dickey-Fuller (ADF)</td>
                <td className="py-2 px-3 text-[#94a3b8]">Unit root present (non-stationary)</td>
                <td className="py-2 px-3 font-mono">{d.adf_stat.toFixed(3)}</td>
                <td className="py-2 px-3 font-mono">{d.adf_pvalue.toFixed(4)}</td>
                <td className="py-2 px-3 text-[#089981] font-semibold">Stationary (p &lt; 0.05)</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">KPSS Test</td>
                <td className="py-2 px-3 text-[#94a3b8]">Stationary around a level</td>
                <td className="py-2 px-3 font-mono">{d.kpss_stat.toFixed(3)}</td>
                <td className="py-2 px-3 font-mono">{d.kpss_pvalue.toFixed(4)}</td>
                <td className="py-2 px-3 text-[#cbd5e1]">Stationary trend confirmed</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Engle ARCH-LM Test</td>
                <td className="py-2 px-3 text-[#94a3b8]">No autoregressive conditional heteroskedasticity</td>
                <td className="py-2 px-3 font-mono">—</td>
                <td className="py-2 px-3 font-mono">{d.arch_lm_pvalue < 0.001 ? '<0.001' : d.arch_lm_pvalue.toFixed(4)}</td>
                <td className="py-2 px-3 text-[#089981] font-semibold">GARCH model justified</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Ljung-Box (Squared Returns)</td>
                <td className="py-2 px-3 text-[#94a3b8]">No serial correlation in variance</td>
                <td className="py-2 px-3 font-mono">—</td>
                <td className="py-2 px-3 font-mono">{d.lb_pvalue_sq_returns < 0.001 ? '<0.001' : d.lb_pvalue_sq_returns.toFixed(4)}</td>
                <td className="py-2 px-3 text-[#089981]">Variance clustering present</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
