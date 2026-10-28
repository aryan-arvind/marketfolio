import React, { useState, useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import {
  Download,
  FileText,
  BarChart2,
  Shield,
  TrendingUp,
  Calendar,
  Activity,
  Database,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Share2,
  ExternalLink,
  Layers,
  Cpu,
} from 'lucide-react';

export default function DeskTabs({ data, onDownloadReport }) {
  const [activeTab, setActiveTab] = useState('Overview');

  const tabs = [
    { id: 'Overview', label: 'Overview', icon: BarChart2 },
    { id: 'DeepAnomaly', label: 'Deep Anomaly (VAE)', icon: Activity },
    { id: 'TransformerRegime', label: 'Volatility Regimes (Transformer)', icon: TrendingUp },
    { id: 'IntermarketGCN', label: 'Inter-Market Topology (GCN)', icon: Share2 },
    { id: 'Volatility', label: 'Volatility Cone & Bootstrap', icon: Shield },
    { id: 'Classical', label: 'Classical Econometrics', icon: Database },
    { id: 'Trello', label: 'Research Tracker', icon: CheckCircle2 },
    { id: 'Report', label: 'Research Report', icon: FileText },
    { id: 'Methodology', label: 'Methodology', icon: Layers },
  ];

  return (
    <div className="flex-1 flex flex-col bg-[#0c1017] border border-[#242936] rounded-md overflow-hidden">
      {/* Secondary Tab Navigation */}
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
        {activeTab === 'DeepAnomaly' && <DeepAnomalyView data={data} />}
        {activeTab === 'TransformerRegime' && <TransformerRegimeView data={data} />}
        {activeTab === 'IntermarketGCN' && <IntermarketGCNView data={data} />}
        {activeTab === 'Volatility' && <VolatilityRiskView data={data} />}
        {activeTab === 'Classical' && <ClassicalEconometricsView data={data} />}
        {activeTab === 'Trello' && <TrelloTrackerView data={data} />}
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
            <span className="text-[10px] text-[#64748b] font-mono">Daily Calibrated</span>
          </div>
          <ul className="space-y-2">
            {data.readings.bullets.map((b, i) => (
              <li key={i} className="text-xs text-[#cbd5e1] flex items-start gap-2">
                <span className="text-[#06b6d4] font-bold leading-none mt-1">•</span>
                <span className="leading-relaxed">{b}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: Daily Return Distribution Histogram */}
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5 flex flex-col justify-between">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>Daily Return Empirical Distribution</span>
            <span className="text-[10px] text-[#64748b] font-mono">50 Bins</span>
          </div>
          <div ref={chartRef} className="w-full" />
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 2: DEEP ANOMALY (VAE)
// -------------------------------------------------------------
function DeepAnomalyView({ data }) {
  const chartReconRef = useRef(null);
  const chartLatentRef = useRef(null);
  const vae = data.vae_anomaly || {};

  useEffect(() => {
    if (!chartReconRef.current || !vae.available || !vae.timeline) return;

    const dates = vae.timeline.map((t) => t.date);
    const errors = vae.timeline.map((t) => t.error);
    const thresh = vae.threshold;

    const anomDates = vae.timeline.filter((t) => t.is_anomaly).map((t) => t.date);
    const anomErrors = vae.timeline.filter((t) => t.is_anomaly).map((t) => t.error);

    // Timeline Chart
    Plotly.react(
      chartReconRef.current,
      [
        {
          x: dates,
          y: errors,
          type: 'scatter',
          mode: 'lines',
          name: 'Reconstruction Loss',
          line: { color: '#06b6d4', width: 1.5 },
        },
        {
          x: [dates[0], dates[dates.length - 1]],
          y: [thresh, thresh],
          type: 'scatter',
          mode: 'lines',
          name: `Train P95 Cutoff (${thresh.toFixed(4)})`,
          line: { color: '#f43f5e', width: 2, dash: 'dash' },
        },
        {
          x: anomDates,
          y: anomErrors,
          type: 'scatter',
          mode: 'markers',
          name: 'Flagged Dislocation',
          marker: { color: '#e11d48', size: 7, symbol: 'diamond' },
        },
      ],
      {
        autosize: true,
        height: 280,
        margin: { l: 45, r: 20, t: 15, b: 25 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right', font: { color: '#94a3b8', size: 10 } },
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        yaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, title: { text: 'MSE Loss', font: { size: 10, color: '#94a3b8' } } },
      },
      { responsive: true, displayModeBar: false }
    );

    // Latent Space Chart
    if (chartLatentRef.current) {
      const normalPts = vae.timeline.filter((t) => !t.is_anomaly);
      const anomPts = vae.timeline.filter((t) => t.is_anomaly);

      Plotly.react(
        chartLatentRef.current,
        [
          {
            x: normalPts.map((t) => t.z1),
            y: normalPts.map((t) => t.z2),
            type: 'scatter',
            mode: 'markers',
            name: 'Normal Manifold',
            marker: { color: '#06b6d4', size: 5, opacity: 0.4 },
          },
          {
            x: anomPts.map((t) => t.z1),
            y: anomPts.map((t) => t.z2),
            type: 'scatter',
            mode: 'markers',
            name: 'Dislocation Outlier',
            marker: { color: '#f43f5e', size: 7, symbol: 'x' },
          },
        ],
        {
          autosize: true,
          height: 260,
          margin: { l: 40, r: 20, t: 15, b: 25 },
          paper_bgcolor: '#131722',
          plot_bgcolor: '#131722',
          legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right', font: { color: '#94a3b8', size: 10 } },
          xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, title: { text: 'μ_z₁', font: { size: 10, color: '#94a3b8' } } },
          yaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, title: { text: 'μ_z₂', font: { size: 10, color: '#94a3b8' } } },
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }, [vae]);

  if (!vae.available) {
    return (
      <div className="bg-[#131722] border border-[#242936] p-4 rounded text-xs text-[#94a3b8]">
        Deep VAE Anomaly Engine not available for this run.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Interpretation Banner */}
      <div className="bg-[#131722] border-l-4 border-[#06b6d4] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-bold text-[#06b6d4] uppercase tracking-wider font-sans mr-2">VAE Market Takeaway:</span>
        <span>{vae.interpretation}</span>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Train P95 Threshold</div>
          <div className="text-lg font-bold font-mono text-[#06b6d4] mt-1">{vae.threshold.toFixed(4)}</div>
          <div className="text-[10px] text-[#64748b]">Calibrated pre-split only</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Flagged Anomalies</div>
          <div className="text-lg font-bold font-mono text-[#f43f5e] mt-1">{vae.n_anomalies}</div>
          <div className="text-[10px] text-[#64748b]">{vae.anomaly_rate_pct.toFixed(1)}% of total sample</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Latent Space</div>
          <div className="text-lg font-bold font-mono text-white mt-1">d_z = {vae.model_summary.latent_dim}</div>
          <div className="text-[10px] text-[#64748b]">Gaussian Prior N(0, I)</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Lookback Window</div>
          <div className="text-lg font-bold font-mono text-white mt-1">{vae.model_summary.lookback} Days</div>
          <div className="text-[10px] text-[#64748b]">{vae.model_summary.input_dim} total features</div>
        </div>
      </div>

      {/* Main Timeline Chart */}
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Multi-Channel Reconstruction Loss vs Calibrated Anomaly Cutoff
        </div>
        <div ref={chartReconRef} className="w-full" />
      </div>

      {/* Grid: Latent Space & Baseline Scorecard */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Latent Manifold Projection (z₁ vs z₂)
          </div>
          <div ref={chartLatentRef} className="w-full" />
        </div>

        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Out-of-Sample Scorecard vs Classical Baselines
          </div>
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#242936] text-[#64748b]">
                <th className="py-2">Model</th>
                <th className="py-2">Flagged</th>
                <th className="py-2">Rate</th>
                <th className="py-2">Mechanism</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242936]/60">
              {vae.scorecard &&
                vae.scorecard.map((s, idx) => (
                  <tr key={idx} className="hover:bg-[#181c27]">
                    <td className="py-2 font-medium text-white">{s.Model}</td>
                    <td className="py-2 font-mono text-[#06b6d4]">{s['Anomalies Flagged']}</td>
                    <td className="py-2 font-mono">{s['Anomaly Rate']}</td>
                    <td className="py-2 text-[11px] text-[#94a3b8]">{s.Mechanism}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Top Anomalies Table */}
      {vae.top_anomalies && vae.top_anomalies.length > 0 && (
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Top Flagged Historical Market Dislocations
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-[#242936] text-[#64748b]">
                  <th className="py-2 px-2">Date</th>
                  <th className="py-2 px-2">Daily Return</th>
                  <th className="py-2 px-2">Recon Loss</th>
                  <th className="py-2 px-2">Loss / Threshold</th>
                  <th className="py-2 px-2">Historical Macro Event</th>
                  <th className="py-2 px-2">3σ Univariate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#242936]/60">
                {vae.top_anomalies.map((a, i) => (
                  <tr key={i} className="hover:bg-[#181c27]">
                    <td className="py-2 px-2 font-mono text-white">{a.date}</td>
                    <td className={`py-2 px-2 font-mono ${a.return_pct < 0 ? 'text-[#f43f5e]' : 'text-[#10b981]'}`}>
                      {a.return_pct > 0 ? `+${a.return_pct.toFixed(2)}%` : `${a.return_pct.toFixed(2)}%`}
                    </td>
                    <td className="py-2 px-2 font-mono">{a.reconstruction_error.toFixed(4)}</td>
                    <td className="py-2 px-2 font-mono font-bold text-[#06b6d4]">{a.threshold_ratio}x</td>
                    <td className="py-2 px-2 text-[#cbd5e1]">{a.matched_event}</td>
                    <td className="py-2 px-2">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${a.is_zscore_confirmed ? 'bg-[#f43f5e]/20 text-[#f43f5e]' : 'bg-[#1e293b] text-[#94a3b8]'}`}>
                        {a.is_zscore_confirmed ? 'Flagged' : 'Normal'}
                      </span>
                    </td>
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
// TAB 3: VOLATILITY REGIMES (TRANSFORMER)
// -------------------------------------------------------------
function TransformerRegimeView({ data }) {
  const chartAttnRef = useRef(null);
  const chartCmRef = useRef(null);
  const trans = data.transformer_regime || {};

  useEffect(() => {
    if (!chartAttnRef.current || !trans.available || !trans.attention_weights) return;

    const attns = trans.attention_weights;
    const lags = attns.map((_, i) => (i === attns.length - 1 ? 't (Current)' : `t-${attns.length - 1 - i}`));

    // Attention Bar Chart
    Plotly.react(
      chartAttnRef.current,
      [
        {
          x: lags,
          y: attns,
          type: 'bar',
          marker: { color: '#06b6d4', line: { color: '#131722', width: 0.5 } },
          name: 'Attention Weight',
        },
      ],
      {
        autosize: true,
        height: 250,
        margin: { l: 40, r: 20, t: 15, b: 35 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 9, color: '#64748b' }, tickangle: -30 },
        yaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, title: { text: 'Weight', font: { size: 10, color: '#94a3b8' } } },
      },
      { responsive: true, displayModeBar: false }
    );

    // Confusion Matrix Heatmap
    if (chartCmRef.current && trans.confusion_matrix) {
      const labels = ['Low Vol', 'Normal Vol', 'High Vol'];
      Plotly.react(
        chartCmRef.current,
        [
          {
            z: trans.confusion_matrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: [
              [0, '#111827'],
              [0.5, '#0e7490'],
              [1, '#06b6d4'],
            ],
            showscale: false,
          },
        ],
        {
          autosize: true,
          height: 250,
          margin: { l: 70, r: 20, t: 15, b: 35 },
          paper_bgcolor: '#131722',
          plot_bgcolor: '#131722',
          xaxis: { title: { text: 'Predicted Regime', font: { size: 10, color: '#94a3b8' } }, tickfont: { size: 10, color: '#94a3b8' } },
          yaxis: { title: { text: 'True Regime', font: { size: 10, color: '#94a3b8' } }, tickfont: { size: 10, color: '#94a3b8' }, autorange: 'reversed' },
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }, [trans]);

  if (!trans.available) {
    return (
      <div className="bg-[#131722] border border-[#242936] p-4 rounded text-xs text-[#94a3b8]">
        Transformer Volatility Classifier not available for this run.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Dynamic Verdict Banner */}
      <div className="bg-[#131722] border-l-4 border-[#06b6d4] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-bold text-[#06b6d4] uppercase tracking-wider font-sans mr-2">{trans.verdict}:</span>
        <span>{trans.interpretation}</span>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Test Macro F1</div>
          <div className="text-lg font-bold font-mono text-[#06b6d4] mt-1">{trans.macro_f1.toFixed(4)}</div>
          <div className="text-[10px] text-[#64748b]">Harmonic mean across 3 regimes</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Balanced Accuracy</div>
          <div className="text-lg font-bold font-mono text-white mt-1">{trans.balanced_accuracy.toFixed(1)}%</div>
          <div className="text-[10px] text-[#64748b]">Prevalence-adjusted</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Test Accuracy</div>
          <div className="text-lg font-bold font-mono text-white mt-1">{trans.accuracy.toFixed(1)}%</div>
          <div className="text-[10px] text-[#64748b]">Out-of-sample exact matches</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Regime Cutoffs</div>
          <div className="text-sm font-bold font-mono text-[#38bdf8] mt-1">
            Low &lt;= {trans.thresholds.low.toFixed(1)}% | High &gt; {trans.thresholds.high.toFixed(1)}%
          </div>
          <div className="text-[10px] text-[#64748b]">Train quantiles only</div>
        </div>
      </div>

      {/* Attention & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Multi-Head Self-Attention Temporal Weights
          </div>
          <div ref={chartAttnRef} className="w-full" />
        </div>

        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Out-of-Sample Confusion Matrix
          </div>
          <div ref={chartCmRef} className="w-full" />
        </div>
      </div>

      {/* Comparative Scorecard Table */}
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Academic Baseline Comparison Scorecard
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#242936] text-[#64748b]">
                <th className="py-2 px-3">Model Architecture</th>
                <th className="py-2 px-3">Accuracy</th>
                <th className="py-2 px-3">Balanced Acc</th>
                <th className="py-2 px-3">Macro F1</th>
                <th className="py-2 px-3">Δ F1 vs Persistence</th>
                <th className="py-2 px-3">Classification Type</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242936]/60">
              {trans.scorecard &&
                trans.scorecard.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[#181c27]">
                    <td className="py-2 px-3 font-medium text-white">{row.Model}</td>
                    <td className="py-2 px-3 font-mono">{row.Accuracy}</td>
                    <td className="py-2 px-3 font-mono">{row['Balanced Acc']}</td>
                    <td className="py-2 px-3 font-mono font-bold text-[#06b6d4]">{row['Macro F1']}</td>
                    <td className="py-2 px-3 font-mono">{row['Delta F1 vs Persistence']}</td>
                    <td className="py-2 px-3 text-[#94a3b8] text-[11px]">{row.Status}</td>
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
// TAB 4: INTER-MARKET GCN
// -------------------------------------------------------------
function IntermarketGCNView({ data }) {
  const chartAdjRef = useRef(null);
  const chartVulnRef = useRef(null);
  const gcn = data.gcn_relational || {};

  useEffect(() => {
    if (!chartAdjRef.current || !gcn.available || !gcn.adjacency_matrix) return;

    // Adjacency Matrix Heatmap
    Plotly.react(
      chartAdjRef.current,
      [
        {
          z: gcn.adjacency_matrix,
          x: gcn.nodes,
          y: gcn.nodes,
          type: 'heatmap',
          colorscale: [
            [0, '#111827'],
            [0.5, '#0e7490'],
            [1, '#06b6d4'],
          ],
          showscale: false,
        },
      ],
      {
        autosize: true,
        height: 260,
        margin: { l: 80, r: 20, t: 15, b: 45 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        xaxis: { tickfont: { size: 9, color: '#94a3b8' }, tickangle: -25 },
        yaxis: { tickfont: { size: 9, color: '#94a3b8' }, autorange: 'reversed' },
      },
      { responsive: true, displayModeBar: false }
    );

    // Node Vulnerability Bar Chart
    if (chartVulnRef.current && gcn.node_vulnerability) {
      const sorted = Object.entries(gcn.node_vulnerability).sort((a, b) => b[1] - a[1]);
      Plotly.react(
        chartVulnRef.current,
        [
          {
            x: sorted.map((s) => s[1]),
            y: sorted.map((s) => s[0]),
            type: 'bar',
            orientation: 'h',
            marker: { color: '#f43f5e', line: { color: '#131722', width: 0.5 } },
          },
        ],
        {
          autosize: true,
          height: 260,
          margin: { l: 80, r: 20, t: 15, b: 35 },
          paper_bgcolor: '#131722',
          plot_bgcolor: '#131722',
          xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, ticksuffix: '%' },
          yaxis: { tickfont: { size: 9, color: '#94a3b8' }, autorange: 'reversed' },
        },
        { responsive: true, displayModeBar: false }
      );
    }
  }, [gcn]);

  if (!gcn.available) {
    return (
      <div className="bg-[#131722] border border-[#242936] p-4 rounded text-xs text-[#94a3b8]">
        Inter-Market GCN not available for this run ({gcn.reason || 'Insufficient aligned data'}).
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Interpretation Callout */}
      <div className="bg-[#131722] border-l-4 border-[#06b6d4] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-bold text-[#06b6d4] uppercase tracking-wider font-sans mr-2">GCN Topology Takeaway:</span>
        <span>{gcn.interpretation}</span>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">GCN Systemic F1</div>
          <div className="text-lg font-bold font-mono text-[#06b6d4] mt-1">{gcn.systemic_f1.toFixed(4)}</div>
          <div className="text-[10px] text-[#64748b]">vs Baseline {gcn.baseline_f1.toFixed(4)}</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Systemic Accuracy</div>
          <div className="text-lg font-bold font-mono text-white mt-1">{gcn.systemic_accuracy.toFixed(1)}%</div>
          <div className="text-[10px] text-[#64748b]">Out-of-sample contagion forecast</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Primary Risk Node</div>
          <div className="text-base font-bold font-mono text-[#f43f5e] mt-1 truncate">
            {Object.keys(gcn.node_vulnerability).reduce((a, b) => (gcn.node_vulnerability[a] > gcn.node_vulnerability[b] ? a : b))}
          </div>
          <div className="text-[10px] text-[#64748b]">Max transmission weight</div>
        </div>
        <div className="bg-[#131722] border border-[#242936] rounded p-3">
          <div className="text-[11px] text-[#94a3b8] uppercase font-semibold">Graph Dimension</div>
          <div className="text-lg font-bold font-mono text-white mt-1">{gcn.nodes.length} Nodes</div>
          <div className="text-[10px] text-[#64748b]">Spectral Convolution (Kipf-Welling)</div>
        </div>
      </div>

      {/* Adjacency & Vulnerability Ranking */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Normalized Adjacency Matrix (Â)
          </div>
          <div ref={chartAdjRef} className="w-full" />
        </div>

        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            Node Systemic Vulnerability & Transmission (%)
          </div>
          <div ref={chartVulnRef} className="w-full" />
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 5: VOLATILITY CONE & BOOTSTRAP (RETAINED)
// -------------------------------------------------------------
function VolatilityRiskView({ data }) {
  const chartRef = useRef(null);
  const vol = data.volatility_cone;

  useEffect(() => {
    if (!chartRef.current || !vol) return;

    Plotly.react(
      chartRef.current,
      [
        {
          x: vol.hist_dates,
          y: vol.hist_vol,
          type: 'scatter',
          mode: 'lines',
          name: 'Historical 21D Realized Vol',
          line: { color: '#06b6d4', width: 1.5 },
        },
        {
          x: vol.forecast_dates,
          y: vol.forecast_median,
          type: 'scatter',
          mode: 'lines',
          name: 'ARIMA Median Forecast',
          line: { color: '#f59e0b', width: 2, dash: 'dash' },
        },
        {
          x: vol.forecast_dates,
          y: vol.forecast_upper,
          type: 'scatter',
          mode: 'lines',
          name: '95th Percentile Bound',
          line: { color: 'rgba(244, 63, 94, 0.4)', width: 1 },
          showlegend: false,
        },
        {
          x: vol.forecast_dates,
          y: vol.forecast_lower,
          type: 'scatter',
          mode: 'lines',
          name: '5th Percentile Bound',
          fill: 'tonexty',
          fillcolor: 'rgba(244, 63, 94, 0.1)',
          line: { color: 'rgba(244, 63, 94, 0.4)', width: 1 },
        },
      ],
      {
        autosize: true,
        height: 320,
        margin: { l: 45, r: 20, t: 15, b: 30 },
        paper_bgcolor: '#131722',
        plot_bgcolor: '#131722',
        legend: { orientation: 'h', y: 1.1, x: 1, xanchor: 'right', font: { color: '#94a3b8', size: 10 } },
        xaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' } },
        yaxis: { gridcolor: '#1e222d', tickfont: { size: 10, color: '#64748b' }, ticksuffix: '%' },
      },
      { responsive: true, displayModeBar: false }
    );
  }, [vol]);

  return (
    <div className="space-y-4">
      <div className="bg-[#131722] border-l-4 border-[#06b6d4] border border-[#242936] p-3 rounded text-xs text-[#cbd5e1] leading-relaxed">
        <span className="font-bold text-[#06b6d4] uppercase tracking-wider font-sans mr-2">Volatility Takeaway:</span>
        <span>{data.readings.vol_reading}</span>
      </div>

      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Realized Volatility Term Structure & Residual Bootstrap Cone (B = 2,000)
        </div>
        <div ref={chartRef} className="w-full" />
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 6: CONSOLIDATED CLASSICAL ECONOMETRICS (RELOCATED)
// -------------------------------------------------------------
function ClassicalEconometricsView({ data }) {
  const mkt = data.market || {};
  const varData = data.macro_var || {};
  const evalData = data.forecast_eval || {};
  const cal = data.calendar || {};

  return (
    <div className="space-y-4">
      {/* 1. Stationarity & Autocorrelation Structure */}
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-3">
          1. Stationarity Suite & Autocorrelation Audit (ADF / KPSS / Ljung-Box)
        </div>
        <table className="w-full text-xs text-left">
          <thead>
            <tr className="border-b border-[#242936] text-[#64748b]">
              <th className="py-2">Series</th>
              <th className="py-2">ADF Stat</th>
              <th className="py-2">ADF p-val</th>
              <th className="py-2">KPSS Stat</th>
              <th className="py-2">KPSS p-val</th>
              <th className="py-2">Econometric Verdict</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#242936]/60">
            {data.econometric_audit &&
              data.econometric_audit.unit_root_table.map((row, idx) => (
                <tr key={idx} className="hover:bg-[#181c27]">
                  <td className="py-2 font-medium text-white">{row.series}</td>
                  <td className="py-2 font-mono">{row.adf_stat.toFixed(3)}</td>
                  <td className="py-2 font-mono">{row.adf_pvalue.toFixed(4)}</td>
                  <td className="py-2 font-mono">{row.kpss_stat.toFixed(3)}</td>
                  <td className="py-2 font-mono">{row.kpss_pvalue.toFixed(4)}</td>
                  <td className="py-2 font-semibold text-[#06b6d4]">{row.decision}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      {/* 2. Benchmark Regression & Macro VAR */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            2. Single-Index SARIMAX Beta (Autocorrelated Errors)
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">Market Beta (β):</span>
              <span className="font-mono font-bold text-white">{mkt.beta ? mkt.beta.toFixed(3) : '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">Annualized Alpha (α):</span>
              <span className="font-mono text-[#06b6d4]">{mkt.alpha ? `${mkt.alpha.toFixed(2)}%` : '—'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">R-Squared:</span>
              <span className="font-mono text-white">{mkt.r_squared ? mkt.r_squared.toFixed(3) : '—'}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#94a3b8]">p-value:</span>
              <span className="font-mono text-white">{mkt.p_value ? mkt.p_value.toFixed(4) : '—'}</span>
            </div>
          </div>
        </div>

        <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
          <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
            3. 5-Variable Macro VAR System
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">Selected Lag:</span>
              <span className="font-mono text-white">Lag {varData.selected_lag || '1'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">Granger Factors:</span>
              <span className="font-mono text-[#06b6d4]">
                {varData.granger_factors && varData.granger_factors.length > 0 ? varData.granger_factors.join(', ') : 'None at 5%'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#242936]/60">
              <span className="text-[#94a3b8]">Cholesky Ordering:</span>
              <span className="font-mono text-[#94a3b8] text-[10px]">
                {varData.cholesky_order ? varData.cholesky_order.join(' → ') : '—'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#94a3b8]">OOS 1-Step Verdict:</span>
              <span className="text-white font-semibold">{varData.oos_verdict || 'Evaluation Complete'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Rolling-Origin OOS Backtest */}
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          4. Expanding-Window Rolling-Origin Backtest
        </div>
        <table className="w-full text-xs text-left">
          <thead>
            <tr className="border-b border-[#242936] text-[#64748b]">
              <th className="py-2">Model</th>
              <th className="py-2">MAE (%)</th>
              <th className="py-2">RMSE (%)</th>
              <th className="py-2">MASE</th>
              <th className="py-2">Improvement vs Naive</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#242936]/60">
            {evalData.table &&
              evalData.table.map((row, idx) => (
                <tr key={idx} className="hover:bg-[#181c27]">
                  <td className="py-2 font-medium text-white">{row.model}</td>
                  <td className="py-2 font-mono">{row.mae.toFixed(2)}%</td>
                  <td className="py-2 font-mono">{row.rmse.toFixed(2)}%</td>
                  <td className="py-2 font-mono font-bold text-[#06b6d4]">{row.mase.toFixed(3)}</td>
                  <td className="py-2 font-mono">{row.improvement}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 7: RESEARCH TRACKER (TRELLO)
// -------------------------------------------------------------
function TrelloTrackerView({ data }) {
  const trello = data.trello || {};
  const lists = trello.lists || [];

  return (
    <div className="space-y-4">
      {/* Board Header Banner */}
      <div className="bg-[#131722] border border-[#242936] p-4 rounded flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div>
          <div className="text-xs font-semibold text-[#06b6d4] uppercase tracking-wider">Research Governance Tracker</div>
          <div className="text-sm font-bold text-white mt-0.5">Academic Experiment & Model Validation Board</div>
          <div className="text-xs text-[#94a3b8] mt-1">{trello.status_label || 'Local Audit Roadmap Active'}</div>
        </div>
        <a
          href={trello.board_url || 'https://trello.com/b/marketfolio-advanced-ml'}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 bg-[#06b6d4] hover:bg-[#0891b2] text-[#0c1017] font-semibold text-xs rounded transition-all"
        >
          <span>Open Board</span>
          <ExternalLink size={13} />
        </a>
      </div>

      {/* Workflow Lists Columns */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {lists.map((lst, idx) => (
          <div key={idx} className="bg-[#111827] border border-[#242936] rounded p-3 flex flex-col">
            <div className="text-xs font-bold text-white uppercase tracking-wider pb-2 mb-3 border-b border-[#06b6d4]">
              {lst.name}
            </div>
            <div className="space-y-2.5 flex-1">
              {lst.cards &&
                lst.cards.map((c, cIdx) => (
                  <div key={cIdx} className="bg-[#131722] border border-[#242936] rounded p-2.5 space-y-1.5">
                    <div className="text-xs font-semibold text-white">{c.name}</div>
                    <div className="text-[11px] text-[#94a3b8] leading-relaxed">{c.desc}</div>
                    <div className="flex items-center justify-between pt-1">
                      <div className="flex gap-1 flex-wrap">
                        {c.tags &&
                          c.tags.map((t, tIdx) => (
                            <span key={tIdx} className="bg-[#1e293b] text-[#38bdf8] text-[9px] px-1.5 py-0.5 rounded">
                              {t}
                            </span>
                          ))}
                      </div>
                      <span className="text-[10px] text-[#10b981] font-semibold font-mono">{c.status || 'Verified'}</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 8: RESEARCH REPORT
// -------------------------------------------------------------
function ReportView({ data, onDownloadReport }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between bg-[#131722] border border-[#242936] p-3 rounded">
        <div>
          <div className="text-xs font-bold text-white uppercase tracking-wider">Quantitative Investment & Model Memo</div>
          <div className="text-[11px] text-[#94a3b8]">Institutional statistical report with model scorecards</div>
        </div>
        <button
          onClick={onDownloadReport}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-[#06b6d4] hover:bg-[#0891b2] text-[#0c1017] font-semibold text-xs rounded transition-all"
        >
          <Download size={13} />
          <span>Export Markdown</span>
        </button>
      </div>

      <div className="bg-[#131722] border border-[#242936] rounded p-4 font-mono text-xs text-[#cbd5e1] whitespace-pre-wrap leading-relaxed max-h-[500px] overflow-y-auto">
        {data.report_markdown}
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// TAB 9: METHODOLOGY
// -------------------------------------------------------------
function MethodologyView({ data }) {
  const d = data.diagnostics || {};

  return (
    <div className="space-y-4">
      <div className="bg-[#131722] border border-[#242936] rounded p-3.5">
        <div className="text-xs font-semibold text-[#94a3b8] uppercase tracking-wider mb-2">
          Statistical Test Battery & Deep Learning Architecture Specifications
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#242936] text-[#64748b]">
                <th className="py-2 px-3">Methodology / Architecture</th>
                <th className="py-2 px-3">Theoretical Foundation</th>
                <th className="py-2 px-3">Specification / Formula</th>
                <th className="py-2 px-3">Verification Mechanism</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#242936]/60">
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Augmented Dickey-Fuller (ADF)</td>
                <td className="py-2 px-3 text-[#94a3b8]">Stochastic Trend vs Integration</td>
                <td className="py-2 px-3 font-mono">Δy_t = α + βt + γy_{'{t-1}'} + Σ δ_i Δy_{'{t-i}'}</td>
                <td className="py-2 px-3 text-[#10b981] font-semibold">Joint ADF & KPSS Synthesis</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Variational Autoencoder (VAE)</td>
                <td className="py-2 px-3 text-[#94a3b8]">Generative Latent Manifold</td>
                <td className="py-2 px-3 font-mono">ELBO = E[log p(x|z)] - β D_KL(q(z|x) || p(z))</td>
                <td className="py-2 px-3 text-[#10b981] font-semibold">Pre-split Train P95 Threshold</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Transformer Encoder</td>
                <td className="py-2 px-3 text-[#94a3b8]">Multi-Head Self-Attention</td>
                <td className="py-2 px-3 font-mono">Attention(Q,K,V) = softmax(QK^T / √d_k) V</td>
                <td className="py-2 px-3 text-[#10b981] font-semibold">Purged OOS Embargo Horizon</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Inter-Market GCN</td>
                <td className="py-2 px-3 text-[#94a3b8]">Spectral Graph Convolution</td>
                <td className="py-2 px-3 font-mono">H^{'(l+1)'} = σ( D~^{'-1/2'} A~ D~^{'-1/2'} H^{'(l)'} W^{'(l)'} )</td>
                <td className="py-2 px-3 text-[#10b981] font-semibold">Normalized Laplacian Adjacency</td>
              </tr>
              <tr className="hover:bg-[#181c27]">
                <td className="py-2 px-3 font-medium text-white">Rolling-Origin Backtest</td>
                <td className="py-2 px-3 text-[#94a3b8]">Expanding Window Protocol</td>
                <td className="py-2 px-3 font-mono">MASE = MAE_model / MAE_naive(insample)</td>
                <td className="py-2 px-3 text-[#10b981] font-semibold">assert_no_leakage Validation</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
