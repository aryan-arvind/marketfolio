"""
MarketFolio Backend API.
Lightweight FastAPI server bridging the Python quantitative risk engine to the React TradingView frontend.
Run via: uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

# Ensure project root and src/ are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ui.pipeline import execute_analysis_pipeline
from ui.state import calculate_derived_stats
import ui.readings as readings
from src.risklens.report import build_markdown_report

app = FastAPI(title="MarketFolio Quantitative Engine API", version="2.0.0")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    asset_mode: str = Field(default="Indian Stock / Index")
    asset_input: str = Field(default="RELIANCE.NS")
    benchmark_symbol: str = Field(default="^NSEI")
    horizon: int = Field(default=10, ge=5, le=60)
    force_refresh: bool = Field(default=False)
    max_origins: int = Field(default=12)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return default
    return float(val)


def _serialize_series(s: pd.Series) -> dict[str, list]:
    return {
        "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in s.index],
        "values": [_safe_float(v) for v in s.values],
    }


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "MarketFolio Quantitative API"}


@app.post("/api/analyze")
def analyze_asset(req: AnalysisRequest):
    try:
        raw = execute_analysis_pipeline(
            asset_mode=req.asset_mode,
            asset_input=req.asset_input,
            benchmark_symbol=req.benchmark_symbol,
            horizon=req.horizon,
            force_refresh=req.force_refresh,
            max_origins=req.max_origins,
            status_box=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    prices_s = raw["clean_res"].price_series
    returns_s = raw["clean_res"].log_return_series
    stats_dict = calculate_derived_stats(prices_s, returns_s)
    vol_res = raw["vol_res"]
    stl_res = raw["stl_res"]
    cal_res = raw["cal_res"]
    mkt_res = raw["market_res"]
    eval_res = raw["eval_res"]
    var_res = raw["var_res"]

    # 1. Price series & Returns
    price_dates = [d.strftime("%Y-%m-%d") for d in prices_s.index]
    price_vals = [_safe_float(v) for v in prices_s.values]
    return_vals = [_safe_float(v) for v in returns_s.values]

    # 2. Realized Volatility Cone
    hist_vol = vol_res.realized_vol_series.dropna() * 100.0
    last_vol = _safe_float(vol_res.last_observed_vol_pct) if vol_res.last_observed_vol_pct else (hist_vol.iloc[-1] if len(hist_vol) > 0 else 0.0)
    last_date = prices_s.index[-1]

    # Prepend anchor point
    boot_med = [_safe_float(x) for x in np.insert(vol_res.boot_median, 0, last_vol)]
    boot_lo = [_safe_float(x) for x in np.insert(vol_res.boot_lower_90, 0, last_vol)]
    boot_hi = [_safe_float(x) for x in np.insert(vol_res.boot_upper_90, 0, last_vol)]
    future_dates = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start=last_date, periods=req.horizon + 1)
    ]

    vol_cone = {
        "hist_dates": [d.strftime("%Y-%m-%d") for d in hist_vol.index],
        "hist_vol": [_safe_float(v) for v in hist_vol.values],
        "forecast_dates": future_dates,
        "forecast_median": boot_med,
        "forecast_lower": boot_lo,
        "forecast_upper": boot_hi,
        "horizon": req.horizon,
    }

    # 3. STL Decomposition
    stl_dates = [d.strftime("%Y-%m-%d") for d in stl_res.trend.index]
    stl_data = {
        "dates": stl_dates,
        "trend": [_safe_float(v) for v in stl_res.trend.values],
        "seasonal": [_safe_float(v) for v in stl_res.seasonal.values],
        "resid": [_safe_float(v) for v in stl_res.residual.values],
        "trend_strength": _safe_float(stl_res.trend_strength_f_t),
        "season_strength": _safe_float(stl_res.seasonal_strength_f_s),
    }

    # 4. Calendar Distributions
    dow_df = cal_res.dow_stats.copy()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_df["order"] = dow_df["day_name"].apply(lambda x: dow_order.index(x) if x in dow_order else 99)
    dow_df = dow_df.sort_values("order")
    calendar_dow = [
        {
            "day_name": row["day_name"],
            "mean": _safe_float(row["mean"] * 100.0),
            "ci_lower": _safe_float(row["ci_lower"] * 100.0),
            "ci_upper": _safe_float(row["ci_upper"] * 100.0),
            "count": int(row["count"]),
        }
        for _, row in dow_df.iterrows()
    ]

    month_df = cal_res.month_stats.copy()
    calendar_month = [
        {
            "month_name": row["month_name"],
            "mean": _safe_float(row["mean"] * 100.0),
            "count": int(row["count"]),
        }
        for _, row in month_df.iterrows()
    ]

    # 5. Market Regression & Scatter
    bench_s = raw.get("bench_returns")
    aligned = pd.concat([returns_s.rename("y"), bench_s.rename("x")], axis=1, join="inner").dropna() if bench_s is not None else pd.DataFrame()
    aligned_y = aligned["y"] * 100.0 if not aligned.empty else pd.Series([], dtype=float)
    aligned_x = aligned["x"] * 100.0 if not aligned.empty else pd.Series([], dtype=float)
    step = max(1, len(aligned_x) // 600)

    rolling_beta = (aligned["y"].rolling(63).cov(aligned["x"]) / aligned["x"].rolling(63).var()).dropna() if len(aligned) > 63 else pd.Series([], dtype=float)

    market_data = {
        "beta": _safe_float(mkt_res.arma_beta if mkt_res else 1.0),
        "alpha": _safe_float((mkt_res.arma_alpha_annualized * 100.0) if mkt_res else 0.0),
        "r_squared": _safe_float(mkt_res.ols_r_squared if mkt_res else 0.0),
        "p_value": _safe_float(mkt_res.arma_beta_pvalue if mkt_res else 0.0),
        "is_significant": bool(mkt_res.arma_beta_pvalue < 0.05 if mkt_res else False),
        "scatter_asset": [_safe_float(v) for v in aligned_y.iloc[::step].values],
        "scatter_bench": [_safe_float(v) for v in aligned_x.iloc[::step].values],
        "rolling_dates": [d.strftime("%Y-%m-%d") for d in rolling_beta.index],
        "rolling_beta": [_safe_float(v) for v in rolling_beta.values],
    }

    # 6. Out-of-Sample Backtest Evaluation
    metrics_rows = []
    if eval_res and hasattr(eval_res, "metrics_table") and not eval_res.metrics_table.empty:
        for _, r in eval_res.metrics_table.iterrows():
            metrics_rows.append({
                "model": str(r.get("Model", "")),
                "mae": _safe_float(r.get("MAE", 0.0) * 100.0 if r.get("MAE") else 0.0),
                "rmse": _safe_float(r.get("RMSE", 0.0) * 100.0 if r.get("RMSE") else 0.0),
                "mase": _safe_float(r.get("MASE", 0.0)),
                "improvement": str(r.get("MASE Improvement vs Naive (%)", "")),
            })

    eval_data = {
        "metrics_table": metrics_rows,
        "overall_mase": _safe_float(eval_res.model_metrics.mase if eval_res else 1.0),
        "overall_mae": _safe_float((eval_res.model_metrics.mae * 100.0) if eval_res else 0.0),
        "naive_mae": _safe_float((eval_res.naive_metrics.mae * 100.0) if eval_res else 0.0),
        "skill_pct": _safe_float(eval_res.model_metrics.mase_improvement_pct if eval_res else 0.0),
        "best_model_name": eval_res.model_metrics.model_name if eval_res else "ARIMA",
        "n_origins": eval_res.n_origins_evaluated if eval_res else 0,
        "summary_verdict": eval_res.summary_verdict if eval_res else "",
        "origins": [],
    }

    # 7. Macro VAR & Granger
    macro_data = None
    if var_res:
        macro_data = {
            "available": True,
            "aligned_obs": int(var_res.aligned_obs),
            "selected_lag": int(var_res.selected_lag),
            "granger_causal_factors": [g.causing_variable for g in var_res.granger_tests if g.is_causal_05],
            "granger_details": [
                {
                    "var": g.causing_variable,
                    "f_stat": _safe_float(g.f_stat),
                    "p_val": _safe_float(g.p_value),
                    "conclusion": g.conclusion,
                }
                for g in var_res.granger_tests
            ],
            "var_beats_baseline": bool(var_res.var_beats_baseline),
            "oos_verdict": var_res.out_of_sample_verdict,
        }

    # 8. Diagnostics & Tests
    stat_suite = raw["stat_suite"]
    ret_diag = stat_suite.get("log_returns") if isinstance(stat_suite, dict) else None
    
    lb_ret = raw.get("lb_returns")
    lb_sq = raw.get("lb_sq_returns")
    lb_ret_p = float(lb_ret["lb_pvalue"].iloc[0]) if lb_ret is not None and not lb_ret.empty else 1.0
    lb_sq_p = float(lb_sq["lb_pvalue"].iloc[0]) if lb_sq is not None and not lb_sq.empty else 1.0

    diagnostics = {
        "adf_stat": _safe_float(ret_diag.adf_stat if ret_diag else -3.5),
        "adf_pvalue": _safe_float(ret_diag.adf_pvalue if ret_diag else 0.001),
        "is_stationary": bool(ret_diag.adf_pvalue < 0.05 if ret_diag else True),
        "kpss_stat": _safe_float(ret_diag.kpss_stat if ret_diag else 0.15),
        "kpss_pvalue": _safe_float(ret_diag.kpss_pvalue if ret_diag else 0.10),
        "arch_lm_pvalue": _safe_float(lb_sq_p),
        "lb_pvalue_returns": _safe_float(lb_ret_p),
        "lb_pvalue_sq_returns": _safe_float(lb_sq_p),
    }

    # 9. Professional Readings & Takeaways
    _, _, vol_desc = readings.get_volatility_regime(vol_res, req.horizon) if vol_res else ("", "", "Volatility analysis unavailable.")
    desk_readings = {
        "vol_reading": vol_desc,
        "trend_reading": readings.get_trend_reading(stl_res) if stl_res else "",
        "calendar_reading": readings.get_calendar_reading(cal_res) if cal_res else "",
        "efficiency_reading": readings.get_predictability_reading(raw["arma_search"]) if raw.get("arma_search") else "",
        "macro_reading": readings.get_macro_drivers_reading(var_res) if var_res else "",
        "data_quality_reading": readings.get_data_quality_reading(raw["clean_res"], raw["meta"]) if raw.get("clean_res") else "",
        "desk_takeaway": readings.get_backtest_reading(eval_res) if eval_res else "",
        "executive_summary": str(raw.get("narrative", "")),
        "bullets": readings.get_overview_bullet_list(raw),
    }

    # 10. Research Report Markdown
    report_md = build_markdown_report(raw["assembled_results"], raw["narrative"])

    return {
        "symbol_code": raw["symbol_code"],
        "display_name": raw["display_name"],
        "asset_mode": req.asset_mode,
        "meta": raw["meta"],
        "warnings": raw.get("warnings", []),
        "price_data": {
            "dates": price_dates,
            "prices": price_vals,
            "returns": return_vals,
        },
        "stats": stats_dict,
        "volatility_cone": vol_cone,
        "stl": stl_data,
        "calendar": {
            "dow": calendar_dow,
            "month": calendar_month,
        },
        "market": market_data,
        "forecast_eval": eval_data,
        "macro_var": macro_data,
        "diagnostics": diagnostics,
        "readings": desk_readings,
        "report_markdown": report_md,
    }


@app.post("/api/report/download")
def download_markdown_report(req: AnalysisRequest):
    try:
        raw = execute_analysis_pipeline(
            asset_mode=req.asset_mode,
            asset_input=req.asset_input,
            benchmark_symbol=req.benchmark_symbol,
            horizon=req.horizon,
            force_refresh=False,
            max_origins=req.max_origins,
            status_box=None,
        )
        md_content = build_markdown_report(raw["assembled_results"], raw["narrative"])
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename=MarketFolio_Memo_{raw['symbol_code']}.md"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


from fastapi.staticfiles import StaticFiles

# Mount built frontend dist for seamless single-port local serving
DIST_DIR = BASE_DIR / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
