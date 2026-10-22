import React from 'react';
import { ArrowUpRight, ArrowDownRight, ShieldCheck, ShieldAlert, Award } from 'lucide-react';

export default function HeaderStatsStrip({ data }) {
  if (!data) return null;

  const stats = data.stats || {};
  const diagnostics = data.diagnostics || {};
  const meta = data.meta || {};
  const symbol_code = data.symbol_code || '';
  const display_name = data.display_name || '';

  const currentPrice = Number(stats.current_price ?? stats.latest_price ?? 0);
  const priceChange = Number(stats.price_change ?? 0);
  const priceChangePct = Number(stats.price_change_pct ?? 0);
  const isUp = priceChange >= 0;

  const curr = meta?.currency || 'INR';
  const currSym = curr === 'INR' ? '₹' : curr;

  const vol21d = Number(stats.vol_21d ?? 0);
  const vol63d = Number(stats.vol_63d ?? 0);
  const volForecast = Number(stats.vol_forecast ?? 0);
  const volBandLow = Number(stats.vol_band_low ?? 0);
  const volBandHigh = Number(stats.vol_band_high ?? 0);
  const beta = Number(stats.beta ?? 1.0);
  const rSquared = Number(stats.r_squared ?? 0);
  const sharpe = Number(stats.sharpe ?? 0);
  const maxDd = Number(stats.max_drawdown ?? 0);
  const low52w = Number(stats.low_52w ?? 0);
  const high52w = Number(stats.high_52w ?? 0);
  const pos52wPct = Number(stats.pos_52w_pct ?? 50);

  return (
    <div className="bg-[#0c1017] border-b border-[#1e222d] px-4 py-2.5 select-none">
      {/* Top row: Symbol, Price, and Quick Primary Tags */}
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-2">
        <div className="flex items-baseline gap-3">
          <span className="text-xl font-bold text-white tracking-tight font-mono">
            {symbol_code}
          </span>
          <span className="text-xs text-[#94a3b8] font-sans font-medium truncate max-w-md">
            {display_name}
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-[#181c27] border border-[#242936] text-[#64748b] uppercase tracking-wider">
            {data.asset_mode}
          </span>
        </div>

        {/* Big Price & Daily Change */}
        <div className="flex items-baseline gap-2.5 font-mono">
          <span className="text-2xl font-bold text-white tracking-tight">
            {currSym} {currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <div
            className={`flex items-center gap-0.5 px-2 py-0.5 rounded text-xs font-semibold ${
              isUp ? 'text-[#089981] bg-[#089981]/10 border border-[#089981]/25' : 'text-[#f23645] bg-[#f23645]/10 border border-[#f23645]/25'
            }`}
          >
            {isUp ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}
            <span>{priceChange > 0 ? '+' : ''}{priceChange.toFixed(2)}</span>
            <span>({priceChangePct > 0 ? '+' : ''}{priceChangePct.toFixed(2)}%)</span>
          </div>
        </div>
      </div>

      {/* Bottom row: TradingView Narrow Metrics Ribbon */}
      <div className="flex flex-wrap items-center gap-4 text-xs font-mono border-t border-[#1e222d]/60 pt-2 text-[#94a3b8]">
        {/* Realized Volatility */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">21D Vol:</span>
          <span className="text-white font-semibold">{vol21d.toFixed(1)}%</span>
          <span className="text-[10px] text-[#64748b]">({vol63d.toFixed(1)}% 63D)</span>
        </div>

        <div className="h-3 w-[1px] bg-[#1e222d]" />

        {/* Volatility Forecast Cone */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">Vol Forecast:</span>
          <span className="text-[#06b6d4] font-semibold">{volForecast.toFixed(1)}%</span>
          <span className="text-[10px] text-[#64748b]">[{volBandLow.toFixed(1)}% - {volBandHigh.toFixed(1)}%]</span>
        </div>

        <div className="h-3 w-[1px] bg-[#1e222d]" />

        {/* Market Beta */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">Beta vs ^NSEI:</span>
          <span className="text-white font-semibold">{beta.toFixed(2)}</span>
          <span className="text-[10px] text-[#64748b]">(R² {(rSquared * 100).toFixed(0)}%)</span>
        </div>

        <div className="h-3 w-[1px] bg-[#1e222d]" />

        {/* Sharpe & Max Drawdown */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">Sharpe:</span>
          <span className="text-white font-semibold">{sharpe.toFixed(2)}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">Max DD:</span>
          <span className="text-[#f23645] font-semibold">{(maxDd * 100).toFixed(1)}%</span>
        </div>

        <div className="h-3 w-[1px] bg-[#1e222d]" />

        {/* 52W Range with Mini Bar */}
        <div className="flex items-center gap-2">
          <span className="text-[#64748b] uppercase text-[10px] tracking-wider font-sans">52W Range:</span>
          <span className="text-[11px] text-[#64748b]">{low52w.toFixed(0)}</span>
          <div className="w-16 h-1.5 bg-[#181c27] rounded-full overflow-hidden border border-[#242936] relative">
            <div
              className="h-full bg-[#06b6d4] rounded-full"
              style={{ width: `${Math.max(0, Math.min(100, pos52wPct))}%` }}
            />
          </div>
          <span className="text-[11px] text-white">{high52w.toFixed(0)}</span>
        </div>

        <div className="h-3 w-[1px] bg-[#1e222d]" />

        {/* Stationarity Badge */}
        <div className="flex items-center gap-1">
          {diagnostics.is_stationary ? (
            <span className="flex items-center gap-1 text-[10px] text-[#089981] bg-[#089981]/10 px-1.5 py-0.5 rounded border border-[#089981]/20">
              <ShieldCheck size={11} />
              <span>Stationary I(0)</span>
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-[#f59e0b] bg-[#f59e0b]/10 px-1.5 py-0.5 rounded border border-[#f59e0b]/20">
              <ShieldAlert size={11} />
              <span>Non-Stationary</span>
            </span>
          )}
        </div>

        {/* OOS Skill */}
        {data.forecast_eval && (
          <div className="flex items-center gap-1">
            <span className="flex items-center gap-1 text-[10px] text-[#38bdf8] bg-[#38bdf8]/10 px-1.5 py-0.5 rounded border border-[#38bdf8]/20">
              <Award size={11} />
              <span>MASE: {(Number(data.forecast_eval.overall_mase) || 1.0).toFixed(2)} ({data.forecast_eval.skill_pct > 0 ? '+' : ''}{(Number(data.forecast_eval.skill_pct) || 0).toFixed(1)}% Skill)</span>
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
