import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { Maximize2, Minimize2, TrendingUp, Activity, Layers } from 'lucide-react';

export default function TradingViewChart({ data, horizon = 10 }) {
  const chartRef = useRef(null);
  const [timeframe, setTimeframe] = useState('ALL');
  const [chartMode, setChartMode] = useState('PRICE'); // 'PRICE' | 'VOL_CONE' | 'STL'
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (!chartRef.current || !data) return;

    let traces = [];
    let layout = {
      autosize: true,
      paper_bgcolor: '#131722',
      plot_bgcolor: '#131722',
      margin: { l: 45, r: 25, t: 15, b: 25 },
      showlegend: false,
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor: '#1e222d',
        bordercolor: '#2a2e39',
        font: { family: 'IBM Plex Mono, monospace', size: 12, color: '#f8fafc' },
      },
      xaxis: {
        gridcolor: '#1e222d',
        linecolor: '#242936',
        tickcolor: '#242936',
        tickfont: { family: 'IBM Plex Sans, sans-serif', size: 11, color: '#64748b' },
        showspikes: true,
        spikemode: 'across',
        spikesnap: 'cursor',
        spikecolor: '#38bdf8',
        spikethickness: 1,
        spikedash: 'dot',
      },
      yaxis: {
        gridcolor: '#1e222d',
        linecolor: '#242936',
        tickcolor: '#242936',
        tickfont: { family: 'IBM Plex Mono, monospace', size: 11, color: '#64748b' },
        side: 'right',
        showspikes: true,
        spikemode: 'across',
        spikesnap: 'cursor',
        spikecolor: '#38bdf8',
        spikethickness: 1,
        spikedash: 'dot',
      },
    };

    if (chartMode === 'PRICE') {
      const dates = data.price_data.dates;
      const prices = data.price_data.prices;

      // Slice based on timeframe
      let startIndex = 0;
      if (dates.length > 0) {
        const lastDate = new Date(dates[dates.length - 1]);
        if (timeframe === '1M') {
          const cut = new Date(lastDate);
          cut.setMonth(cut.getMonth() - 1);
          startIndex = dates.findIndex((d) => new Date(d) >= cut);
        } else if (timeframe === '6M') {
          const cut = new Date(lastDate);
          cut.setMonth(cut.getMonth() - 6);
          startIndex = dates.findIndex((d) => new Date(d) >= cut);
        } else if (timeframe === '1Y') {
          const cut = new Date(lastDate);
          cut.setFullYear(cut.getFullYear() - 1);
          startIndex = dates.findIndex((d) => new Date(d) >= cut);
        } else if (timeframe === '5Y') {
          const cut = new Date(lastDate);
          cut.setFullYear(cut.getFullYear() - 5);
          startIndex = dates.findIndex((d) => new Date(d) >= cut);
        }
      }
      if (startIndex < 0) startIndex = 0;

      const subDates = dates.slice(startIndex);
      const subPrices = prices.slice(startIndex);

      traces.push({
        x: subDates,
        y: subPrices,
        type: 'scatter',
        mode: 'lines',
        name: data.symbol_code,
        line: { color: '#06b6d4', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(6, 182, 212, 0.06)',
        hovertemplate: 'Price: <b>%{y:,.2f}</b><extra></extra>',
      });
    } else if (chartMode === 'VOL_CONE') {
      const vc = data.volatility_cone;
      layout.showlegend = true;
      layout.legend = {
        x: 0.02,
        y: 0.98,
        bgcolor: 'rgba(19, 23, 34, 0.75)',
        bordercolor: '#242936',
        borderwidth: 1,
        font: { size: 11, color: '#94a3b8' },
      };
      layout.yaxis.ticksuffix = '%';

      // 1. Historical Realized Vol
      traces.push({
        x: vc.hist_dates,
        y: vc.hist_vol,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical 21D Vol',
        line: { color: '#94a3b8', width: 1.5 },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
      });

      // 2. 90% Bootstrap Envelope
      traces.push({
        x: vc.forecast_dates,
        y: vc.forecast_upper,
        type: 'scatter',
        mode: 'lines',
        name: '90% Upper Bound',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      });
      traces.push({
        x: vc.forecast_dates,
        y: vc.forecast_lower,
        type: 'scatter',
        mode: 'lines',
        name: '90% Confidence Interval',
        fill: 'tonexty',
        fillcolor: 'rgba(6, 182, 212, 0.15)',
        line: { width: 0 },
        hoverinfo: 'skip',
      });

      // 3. Forecast Median
      traces.push({
        x: vc.forecast_dates,
        y: vc.forecast_median,
        type: 'scatter',
        mode: 'lines+markers',
        name: `${vc.horizon}D Forecast (ARIMA+GARCH)`,
        line: { color: '#06b6d4', width: 2.2 },
        marker: { size: 4, color: '#06b6d4' },
        hovertemplate: 'Forecast: %{y:.1f}%<extra></extra>',
      });
    } else if (chartMode === 'STL') {
      const stl = data.stl;
      traces.push({
        x: stl.dates,
        y: stl.trend,
        type: 'scatter',
        mode: 'lines',
        name: 'Macro Trend Component',
        line: { color: '#06b6d4', width: 2 },
        hovertemplate: 'Trend: %{y:,.2f}<extra></extra>',
      });
      traces.push({
        x: data.price_data.dates,
        y: data.price_data.prices,
        type: 'scatter',
        mode: 'lines',
        name: 'Actual Price',
        line: { color: '#475569', width: 1 },
        opacity: 0.5,
        hovertemplate: 'Raw: %{y:,.2f}<extra></extra>',
      });
    }

    Plotly.react(chartRef.current, traces, layout, {
      responsive: true,
      displayModeBar: false,
    });

    const handleResize = () => {
      if (chartRef.current) {
        Plotly.Plots.resize(chartRef.current);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [data, timeframe, chartMode]);

  return (
    <div className={`relative flex flex-col bg-[#131722] border border-[#242936] rounded-md overflow-hidden ${isFullscreen ? 'fixed inset-0 z-50 p-4' : 'h-[500px]'}`}>
      {/* TradingView Chart Top Control Bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#0c1017] border-b border-[#242936] text-xs">
        {/* Left: Mode Switcher */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setChartMode('PRICE')}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-colors ${
              chartMode === 'PRICE' ? 'bg-[#06b6d4] text-[#0c1017] font-semibold' : 'text-[#94a3b8] hover:text-white hover:bg-[#1e222d]'
            }`}
          >
            <TrendingUp size={13} />
            <span>Price Chart</span>
          </button>
          <button
            onClick={() => setChartMode('VOL_CONE')}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-colors ${
              chartMode === 'VOL_CONE' ? 'bg-[#06b6d4] text-[#0c1017] font-semibold' : 'text-[#94a3b8] hover:text-white hover:bg-[#1e222d]'
            }`}
          >
            <Activity size={13} />
            <span>Volatility Cone</span>
          </button>
          <button
            onClick={() => setChartMode('STL')}
            className={`px-2.5 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-colors ${
              chartMode === 'STL' ? 'bg-[#06b6d4] text-[#0c1017] font-semibold' : 'text-[#94a3b8] hover:text-white hover:bg-[#1e222d]'
            }`}
          >
            <Layers size={13} />
            <span>Trend Decomposition</span>
          </button>
        </div>

        {/* Center: Live Hover Info / Timeframe Selector */}
        {chartMode === 'PRICE' && (
          <div className="flex items-center gap-1 bg-[#131722] p-0.5 rounded border border-[#242936]">
            {['1M', '6M', '1Y', '5Y', 'ALL'].map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                  timeframe === tf ? 'bg-[#242936] text-[#38bdf8]' : 'text-[#64748b] hover:text-[#e2e8f0]'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        )}

        {/* Right: Fullscreen Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-[#64748b] font-mono hidden sm:inline">
            {data?.display_name} • {data?.symbol_code}
          </span>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 rounded text-[#94a3b8] hover:text-white hover:bg-[#1e222d] transition-colors"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Chart'}
          >
            {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* Main Plot Area */}
      <div ref={chartRef} className="flex-1 w-full h-full" />
    </div>
  );
}
