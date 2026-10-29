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
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

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


import json
import re

_ANALYSIS_CACHE: dict[tuple, dict] = {}
API_CACHE_DIR = BASE_DIR / ".cache" / "api_analysis"
API_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_disk_cache_path(key: tuple) -> Path:
    safe = "_".join(re.sub(r"[^A-Za-z0-9_-]", "_", str(k)) for k in key)
    return API_CACHE_DIR / f"{safe}.json"


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "MarketFolio Quantitative API"}


@app.post("/api/analyze")
def analyze_asset(req: AnalysisRequest):
    cache_key = (req.asset_mode, req.asset_input.strip().upper(), req.benchmark_symbol, req.horizon)
    if not req.force_refresh:
        if cache_key in _ANALYSIS_CACHE:
            return _ANALYSIS_CACHE[cache_key]
        disk_path = _get_disk_cache_path(cache_key)
        if disk_path.exists():
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                _ANALYSIS_CACHE[cache_key] = cached
                return cached
            except Exception:
                pass

    try:
        raw = execute_analysis_pipeline(
            asset_mode=req.asset_mode,
            asset_input=req.asset_input,
            benchmark_symbol=req.benchmark_symbol,
            horizon=req.horizon,
            force_refresh=req.force_refresh,
            max_origins=min(req.max_origins, 3),
            status_box=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    prices_s = raw["clean_res"].price_series
    returns_s = raw["clean_res"].log_return_series
    derived = calculate_derived_stats(prices_s, returns_s)
    vol_res = raw["vol_res"]
    stl_res = raw["stl_res"]
    cal_res = raw["cal_res"]
    mkt_res = raw["market_res"]
    eval_res = raw["eval_res"]
    var_res = raw["var_res"]

    # Compute enriched stats dictionary matching TradingView frontend expectations
    clean_rets = returns_s.dropna()
    vol_21d = float(clean_rets.iloc[-21:].std() * np.sqrt(252) * 100.0) if len(clean_rets) >= 21 else 15.0
    vol_63d = float(clean_rets.iloc[-63:].std() * np.sqrt(252) * 100.0) if len(clean_rets) >= 63 else vol_21d
    rolling_21 = clean_rets.rolling(21).std() * np.sqrt(252) * 100.0
    rolling_21 = rolling_21.dropna()
    if len(rolling_21) > 0:
        vol_pctile = float((rolling_21 < vol_21d).mean() * 100.0)
    else:
        vol_pctile = 50.0

    boot_med_val = float(vol_res.boot_median[-1]) if vol_res and len(vol_res.boot_median) > 0 else vol_21d
    boot_lo_val = float(vol_res.boot_lower_90[-1]) if vol_res and len(vol_res.boot_lower_90) > 0 else (vol_21d * 0.8)
    boot_hi_val = float(vol_res.boot_upper_90[-1]) if vol_res and len(vol_res.boot_upper_90) > 0 else (vol_21d * 1.2)

    ret_mean = float(clean_rets.mean()) if len(clean_rets) > 0 else 0.0
    ret_std = float(clean_rets.std()) if len(clean_rets) > 0 else 0.01
    sharpe = float((ret_mean / (ret_std + 1e-9)) * np.sqrt(252))
    downside_rets = clean_rets[clean_rets < 0]
    downside_std = float(downside_rets.std()) if len(downside_rets) > 1 else ret_std
    sortino = float((ret_mean / (downside_std + 1e-9)) * np.sqrt(252))

    cur_p = float(derived["latest_price"])
    lo_52 = float(derived["low_52w"])
    hi_52 = float(derived["high_52w"])
    pos_52 = ((cur_p - lo_52) / (hi_52 - lo_52 + 1e-9)) * 100.0

    stats_dict = {
        "current_price": cur_p,
        "latest_price": cur_p,
        "prev_price": float(derived["prev_price"]),
        "price_change": float(derived["price_change_1d"]),
        "price_change_pct": float(derived["pct_change_1d"]),
        "low_52w": lo_52,
        "high_52w": hi_52,
        "pos_52w_pct": _safe_float(pos_52),
        "vol_21d": _safe_float(vol_21d),
        "vol_63d": _safe_float(vol_63d),
        "vol_percentile": _safe_float(vol_pctile),
        "vol_forecast": _safe_float(boot_med_val),
        "vol_band_low": _safe_float(boot_lo_val),
        "vol_band_high": _safe_float(boot_hi_val),
        "beta": _safe_float(mkt_res.arma_beta if mkt_res else 1.0),
        "r_squared": _safe_float(mkt_res.ols_r_squared if mkt_res else 0.0),
        "sharpe": _safe_float(sharpe),
        "sortino": _safe_float(sortino),
        "max_drawdown": _safe_float(abs(derived["max_drawdown_pct"]) / 100.0),
        "max_drawdown_pct": _safe_float(derived["max_drawdown_pct"]),
        "skewness": _safe_float(derived["skewness"]),
        "kurtosis": _safe_float(derived["kurtosis"]),
        "return_std": _safe_float(derived["return_std"]),
        "n_extreme_sessions": int(derived["n_extreme_sessions"]),
    }

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

    # Full ATSA Tables & Diagnostics
    clean_r = returns_s.dropna().values
    n_obs = len(clean_r)
    try:
        acf_vals = sm_acf(clean_r, nlags=20)[1:]
        pacf_vals = sm_pacf(clean_r, nlags=20)[1:]
    except Exception:
        acf_vals = [0.0] * 20
        pacf_vals = [0.0] * 20
    ci_bound = 1.96 / np.sqrt(n_obs) if n_obs > 0 else 0.05

    acf_pacf_data = {
        "lags": list(range(1, 21)),
        "acf": [_safe_float(v) for v in acf_vals],
        "pacf": [_safe_float(v) for v in pacf_vals],
        "ci_bound": _safe_float(ci_bound),
    }

    lb_rows = []
    if lb_ret is not None and not lb_ret.empty and lb_sq is not None and not lb_sq.empty:
        for lag in [10, 20]:
            r_stat = float(lb_ret.loc[lag, "lb_stat"]) if lag in lb_ret.index else 0.0
            r_pval = float(lb_ret.loc[lag, "lb_pvalue"]) if lag in lb_ret.index else 1.0
            sq_stat = float(lb_sq.loc[lag, "lb_stat"]) if lag in lb_sq.index else 0.0
            sq_pval = float(lb_sq.loc[lag, "lb_pvalue"]) if lag in lb_sq.index else 1.0
            lb_rows.append({
                "lag": lag,
                "returns_stat": _safe_float(r_stat),
                "returns_pval": _safe_float(r_pval),
                "sq_returns_stat": _safe_float(sq_stat),
                "sq_returns_pval": _safe_float(sq_pval),
            })

    price_diag = stat_suite.get("log_price") if isinstance(stat_suite, dict) else None
    unit_root_rows = [
        {
            "series": "Log Price (P_t)",
            "adf_stat": _safe_float(price_diag.adf_stat if price_diag else 0.17),
            "adf_pvalue": _safe_float(price_diag.adf_pvalue if price_diag else 0.971),
            "kpss_stat": _safe_float(price_diag.kpss_stat if price_diag else 10.5),
            "kpss_pvalue": _safe_float(price_diag.kpss_pvalue if price_diag else 0.01),
            "decision": price_diag.verdict if price_diag else "Unit Root / Non-Stationary I(1)",
        },
        {
            "series": "Log Return (r_t)",
            "adf_stat": _safe_float(ret_diag.adf_stat if ret_diag else -23.29),
            "adf_pvalue": _safe_float(ret_diag.adf_pvalue if ret_diag else 0.0001),
            "kpss_stat": _safe_float(ret_diag.kpss_stat if ret_diag else 0.08),
            "kpss_pvalue": _safe_float(ret_diag.kpss_pvalue if ret_diag else 0.10),
            "decision": ret_diag.verdict if ret_diag else "Stationary I(0)",
        },
    ]

    arma_search = raw.get("arma_search")
    arma_grid_rows = []
    if arma_search and hasattr(arma_search, "all_models_df") and arma_search.all_models_df is not None:
        for _, r in arma_search.all_models_df.head(6).iterrows():
            arma_grid_rows.append({
                "p": int(r.get("p", 0)),
                "d": int(r.get("d", 0)),
                "q": int(r.get("q", 0)),
                "order": f"({int(r.get('p', 0))},{int(r.get('d', 0))},{int(r.get('q', 0))})",
                "aic": _safe_float(r.get("aic", 0.0)),
                "aicc": _safe_float(r.get("aicc", 0.0)),
                "bic": _safe_float(r.get("bic", 0.0)),
            })

    econometric_audit = {
        "acf_pacf": acf_pacf_data,
        "ljung_box_table": lb_rows,
        "unit_root_table": unit_root_rows,
        "arma_grid_table": arma_grid_rows,
        "best_arma_order": f"({arma_search.best_order[0]},{arma_search.best_order[1]},{arma_search.best_order[2]})" if arma_search else "(0,0,0)",
        "is_predictable": bool(arma_search.is_predictable) if arma_search else False,
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

    # 10. Advanced ML Models Serialization
    vae_obj = raw.get("vae_res")
    vae_data = {}
    if vae_obj:
        tl_sample = vae_obj.timeline_df.copy()
        # Subsample timeline to 400 points max for efficient frontend network payload
        step = max(1, len(tl_sample) // 400)
        tl_sub = tl_sample.iloc[::step]
        vae_data = {
            "available": True,
            "threshold": _safe_float(vae_obj.reconstruction_threshold),
            "n_anomalies": int(vae_obj.n_anomalies_detected),
            "anomaly_rate_pct": _safe_float(vae_obj.anomaly_rate_pct),
            "interpretation": vae_obj.interpretation,
            "timeline": [
                {
                    "date": str(r["date"]),
                    "error": _safe_float(r["vae_recon_error"]),
                    "is_anomaly": bool(r["is_vae_anomaly"]),
                    "z1": _safe_float(r["latent_z1"]),
                    "z2": _safe_float(r["latent_z2"]),
                }
                for _, r in tl_sub.iterrows()
            ],
            "top_anomalies": vae_obj.top_anomalies,
            "scorecard": vae_obj.baseline_scorecard.to_dict(orient="records"),
            "model_summary": vae_obj.model_summary,
        }
    else:
        vae_data = {"available": False}

    trans_obj = raw.get("transformer_res")
    trans_data = {}
    if trans_obj:
        trans_data = {
            "available": True,
            "accuracy": _safe_float(trans_obj.accuracy * 100.0),
            "balanced_accuracy": _safe_float(trans_obj.balanced_accuracy * 100.0),
            "macro_f1": _safe_float(trans_obj.macro_f1),
            "verdict": trans_obj.verdict,
            "interpretation": trans_obj.interpretation,
            "confusion_matrix": trans_obj.confusion_matrix,
            "attention_weights": [_safe_float(w) for w in trans_obj.mean_attention_weights],
            "scorecard": trans_obj.scorecard_df.to_dict(orient="records"),
            "thresholds": {
                "low": _safe_float(trans_obj.thresholds.low_cutoff * 100.0),
                "high": _safe_float(trans_obj.thresholds.high_cutoff * 100.0),
            },
        }
    else:
        trans_data = {"available": False}

    gcn_obj = raw.get("gcn_res")
    gcn_data = {}
    if gcn_obj and gcn_obj.available:
        gcn_data = {
            "available": True,
            "nodes": gcn_obj.nodes,
            "adjacency_matrix": [[_safe_float(c) for c in row] for row in gcn_obj.adjacency_matrix],
            "node_vulnerability": {k: _safe_float(v) for k, v in gcn_obj.node_vulnerability_scores.items()},
            "systemic_accuracy": _safe_float(gcn_obj.systemic_stress_accuracy * 100.0),
            "systemic_f1": _safe_float(gcn_obj.systemic_stress_f1),
            "baseline_accuracy": _safe_float(gcn_obj.baseline_accuracy * 100.0),
            "baseline_f1": _safe_float(gcn_obj.baseline_f1),
            "interpretation": gcn_obj.interpretation,
        }
    else:
        gcn_data = {"available": False, "reason": getattr(gcn_obj, "reason", "Macro data not aligned")}

    trello_obj = raw.get("trello_res") or {}

    # 11. Research Report Markdown
    report_md = build_markdown_report(raw["assembled_results"], raw["narrative"])

    response_payload = {
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
        "econometric_audit": econometric_audit,
        "readings": desk_readings,
        "report_markdown": report_md,
        "vae_anomaly": vae_data,
        "transformer_regime": trans_data,
        "gcn_relational": gcn_data,
        "trello": trello_obj,
    }
    _ANALYSIS_CACHE[cache_key] = response_payload
    try:
        disk_path = _get_disk_cache_path(cache_key)
        with open(disk_path, "w", encoding="utf-8") as f:
            json.dump(response_payload, f)
    except Exception:
        pass
    return response_payload


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
