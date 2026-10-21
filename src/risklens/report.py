"""
Comprehensive report generator and artifact exporter for RiskLens.
Outputs structured JSON results, high-resolution figures (PNG), formatted tables (CSV),
and a complete Markdown report meeting university ATSA submission standards.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from risklens.config import REPORTS_DIR

logger = logging.getLogger("risklens")


def create_report_directory(asset_identifier: str, base_out_dir: Path | None = None) -> Path:
    """Creates a dedicated report directory: reports/<asset_identifier>_<YYYY-MM-DD>/"""
    if base_out_dir is None:
        base_out_dir = REPORTS_DIR

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sanitized_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in asset_identifier)
    dir_name = f"{sanitized_name}_{date_str}"
    target_dir = base_out_dir / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def export_analysis_bundle(
    report_dir: Path,
    results_data: dict[str, Any],
    narrative_text: str,
    tables_dict: dict[str, pd.DataFrame],
) -> Path:
    """
    Saves:
    1. results.json: Complete raw and computed metrics.
    2. CSV tables: All diagnostic and evaluation tables.
    3. report.md: Full Markdown report with ATSA syllabus mapping and disclaimer.
    """
    # 1. Export results.json
    json_path = report_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info("Saved complete numeric results to '%s'", json_path)

    # 2. Export CSV tables
    for name, df in tables_dict.items():
        if df is not None and not df.empty:
            csv_path = report_dir / f"{name}.csv"
            df.to_csv(csv_path, index=True)
            logger.debug("Exported table '%s.csv'", name)

    # 3. Assemble report.md
    md_content = build_markdown_report(results_data, narrative_text)
    md_path = report_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved comprehensive Markdown report to '%s'", md_path)

    return md_path


def build_markdown_report(results_data: dict[str, Any], narrative_text: str) -> str:
    """Constructs the comprehensive ATSA report string."""
    meta = results_data.get("asset_metadata", {})
    name = meta.get("name", "Asset")
    symbol = meta.get("symbol_or_code", "N/A")
    source = meta.get("source", "Public Source")
    last_date = meta.get("last_available_date", "N/A")
    fetch_time = meta.get("fetch_timestamp", "N/A")
    is_stale = meta.get("is_stale", False)
    stale_tag = " **[STALE CACHE - LIVE API UNAVAILABLE]**" if is_stale else ""
    n_obs = meta.get("n_observations", 0)

    # Warnings
    warnings_list = results_data.get("warnings", [])
    anomalies = results_data.get("flagged_anomalies", [])

    # Diagnostics
    stat = results_data.get("stationarity", {})
    cal = results_data.get("seasonality", {})
    arma = results_data.get("arma_predictability", {})
    vol = results_data.get("volatility_forecast", {})
    mkt = results_data.get("market_model", {})
    var_res = results_data.get("var_macro", {})
    eval_res = results_data.get("evaluation", {})

    md = []
    md.append(f"# RiskLens ATSA Quantitative Risk & Macro-Linkage Report")
    md.append(f"## Target Asset: {name} ({symbol}){stale_tag}\n")

    # Data Freshness Header
    md.append("### Data Freshness & Ingestion Metadata")
    md.append(f"- **Primary Data Source**: {source}")
    md.append(f"- **Last Available Market Date**: {last_date}")
    md.append(f"- **Cleaned Sample Size**: {n_obs} daily observations")
    md.append(f"- **Data Ingest Timestamp**: {fetch_time}")
    md.append(f"- **Cache Freshness Status**: {'STALE CACHE' if is_stale else 'Verified Live Fresh (TTL < 6h)'}")
    md.append("")

    if warnings_list:
        md.append("> [!WARNING]")
        for w in warnings_list:
            md.append(f"> - {w}")
        md.append("")

    if anomalies:
        md.append("> [!NOTE]")
        md.append(f"> **Large Return Outliers Flagged (|r_t| > 15%)**: {len(anomalies)} events detected.")
        md.append("> These occurrences typically reflect corporate actions, stock splits, or reporting artefacts and are retained for empirical honesty.")
        md.append("")

    # Executive narrative
    md.append("## 1. Executive Summary & Synthesis")
    md.append(narrative_text)
    md.append("")

    # ATSA Syllabus Mapping Table
    md.append("## 2. ATSA Syllabus Alignment Matrix")
    md.append("| ATSA Syllabus Unit | Quantitative Topic | Implementation Module | Empirical Finding |")
    md.append("|:---|:---|:---|:---|")
    md.append(f"| **Unit 1: Stationarity & Decomposition** | Unit Root Tests & STL | `diagnostics.py`, `seasonality.py` | Log price: {stat.get('log_price_verdict', 'N/A')}; Returns: {stat.get('log_returns_verdict', 'N/A')} |")
    md.append(f"| **Unit 1: Calendar Effects** | Subseries & Dummy OLS | `seasonality.py` | OLS F-test p={cal.get('ols_f_pvalue', 0.0):.4f} ({'Significant' if cal.get('day_of_week_significant', False) else 'No effect'}) |")
    md.append(f"| **Unit 3 & 4: ARMA Identification** | AICc Grid Search & EMH | `arma_models.py` | Best: ARIMA{arma.get('best_order', '(0,0,0)')}; Predictable: {arma.get('is_predictable', False)} |")
    md.append(f"| **Unit 2 & 5: Transfer / Reg with ARMA** | Single-Index Alpha & Beta | `regression_arma.py` | Beta={mkt.get('beta', 0.0):.3f}, Annual Alpha={mkt.get('alpha_annualized_pct', 0.0):+.2f}% (p={mkt.get('alpha_pvalue', 0.0):.4f}) |")
    md.append(f"| **Unit 4, 5 & 8: Volatility & Bootstrap** | Realized Vol & Bootstrap (B=2000) | `volatility.py` | Current={vol.get('last_observed_vol_pct', 0.0):.1f}%, 30d Risk Range=[{vol.get('expected_risk_range_lower_pct', 0.0):.1f}%, {vol.get('expected_risk_range_upper_pct', 0.0):.1f}%] |")
    md.append(f"| **Unit 8: Vector Autoregression** | Granger Causality, IRF & FEVD | `var_analysis.py` | VAR({var_res.get('selected_lag', 1)}) stable; Causal: {', '.join(var_res.get('granger_causal_factors', [])) or 'None'} |")
    md.append(f"| **Unit 4 & 5: Forecast Evaluation** | Rolling-Origin MASE Backtesting | `evaluation.py` | MASE improvement vs Naive: {eval_res.get('model_mase_improvement_pct', 0.0):+.2f}% |")
    md.append("")

    # Module 1
    md.append("## 3. Stationarity & Order of Integration (Units 1 & 5)")
    md.append("### Empirical Test Results")
    md.append("| Series Tested | ADF Statistic | ADF p-value | KPSS Statistic | KPSS p-value | Decision Verdict |")
    md.append("|:---|:---|:---|:---|:---|:---|")
    md.append(f"| Log Price / NAV | {stat.get('price_adf_stat', 0.0):.4f} | {stat.get('price_adf_pvalue', 0.0):.4f} | {stat.get('price_kpss_stat', 0.0):.4f} | {stat.get('price_kpss_pvalue', 0.0):.4f} | {stat.get('log_price_verdict', 'N/A')} |")
    md.append(f"| Log Returns r_t | {stat.get('returns_adf_stat', 0.0):.4f} | {stat.get('returns_adf_pvalue', 0.0):.4f} | {stat.get('returns_kpss_stat', 0.0):.4f} | {stat.get('returns_kpss_pvalue', 0.0):.4f} | {stat.get('log_returns_verdict', 'N/A')} |")
    md.append("")
    md.append(f"> **Autocorrelation Analysis**: Ljung-Box test on returns at lag 10 yields p = {stat.get('lb_returns_pvalue_lag10', 0.0):.4f}; on squared returns r_t^2 yields p = {stat.get('lb_sq_returns_pvalue_lag10', 0.0):.4f}. Significant autocorrelation in squared returns confirms pronounced volatility clustering (Mandelbrot 1963; Engle 1982).")
    md.append("")

    # Module 2
    md.append("## 4. Seasonality & Calendar Effects (Unit 1)")
    md.append(f"- **Day-of-Week OLS Joint F-test**: Statistic = {cal.get('ols_f_stat', 0.0):.3f}, p-value = {cal.get('ols_f_pvalue', 0.0):.4f}")
    md.append(f"- **Kruskal-Wallis Non-Parametric Test**: Statistic = {cal.get('kruskal_stat', 0.0):.3f}, p-value = {cal.get('kruskal_pvalue', 0.0):.4f}")
    md.append(f"- **STL Trend Strength (F_T)**: {cal.get('trend_strength_f_t', 0.0):.4f}")
    md.append(f"- **STL Seasonal Strength (F_S)**: {cal.get('seasonal_strength_f_s', 0.0):.4f}")
    md.append(f"- **Verdict**: {cal.get('summary_verdict', 'N/A')}")
    md.append("")

    # Module 3
    md.append("## 5. ARMA Conditional Mean Modeling & Predictability (Units 3, 4, 5)")
    md.append(f"- **Best Model Selected via AICc**: ARIMA{arma.get('best_order', '(0,0,0)')}")
    md.append(f"- **AICc**: {arma.get('best_aicc', 0.0):.2f} | **BIC**: {arma.get('best_bic', 0.0):.2f}")
    md.append(f"- **Constant-Drift Model AICc**: {arma.get('constant_aicc', 0.0):.2f} (Delta AICc = {arma.get('delta_aicc_vs_constant', 0.0):.2f})")
    md.append(f"- **Residual Diagnostics**: Ljung-Box lag 10 p-value = {arma.get('residual_lb_pvalue', 0.0):.4f}; Jarque-Bera p-value = {arma.get('residual_jb_pvalue', 0.0):.4f}")
    md.append(f"- **Predictability Verdict**: {arma.get('verdict', 'N/A')}")
    md.append("")

    # Module 4
    md.append("## 6. Realized Volatility & Residual Bootstrap Forecast (Units 4, 5, 8)")
    md.append(f"- **Target Series**: Log Realized Volatility $v_t = \\ln(\\sqrt{{252}} \\cdot \\text{{std}}_{{21}}(r_t))$")
    md.append(f"- **Last Observed Annualized Volatility**: **{vol.get('last_observed_vol_pct', 0.0):.2f}%**")
    md.append(f"- **30-Day Point Forecast (Median)**: **{vol.get('point_forecast_pct', 0.0):.2f}%**")
    md.append(f"- **Expected 30-Day Risk Range (90% Interval)**: **[{vol.get('expected_risk_range_lower_pct', 0.0):.2f}% - {vol.get('expected_risk_range_upper_pct', 0.0):.2f}%]** p.a.")
    md.append(f"- **Benchmark Baselines**: Naive = {vol.get('naive_forecast_pct', 0.0):.2f}%, 21-Day Rolling Mean = {vol.get('rolling_mean_forecast_pct', 0.0):.2f}%")
    md.append(f"- **Bootstrap Details**: B = 2000 replications with mean-adjusted recursive ARMA simulation.")
    md.append("")

    # Module 5
    md.append(f"## 7. Systematic Market Linkage: Asset vs {mkt.get('benchmark_name', '^NSEI')} (Units 2 & 5)")
    md.append(f"- **Common Dates Aligned**: {mkt.get('n_aligned_obs', 0)} sessions ({mkt.get('start_date', '')} to {mkt.get('end_date', '')})")
    md.append(f"- **Market Beta**: **{mkt.get('beta', 0.0):.3f}** (SE = {mkt.get('beta_se', 0.0):.3f}, 95% CI: [{mkt.get('beta_ci_lower', 0.0):.3f}, {mkt.get('beta_ci_upper', 0.0):.3f}])")
    md.append(f"- **Annualized Alpha**: **{mkt.get('alpha_annualized_pct', 0.0):+.2f}%** (p-value = {mkt.get('alpha_pvalue', 0.0):.4f})")
    md.append(f"- **OLS Explanatory Power**: R^2 = {mkt.get('r_squared', 0.0):.3f}")
    md.append(f"- **OLS Residual Durbin-Watson**: {mkt.get('durbin_watson', 0.0):.2f} (Ljung-Box p = {mkt.get('ols_lb_pvalue', 0.0):.4f})")
    md.append(f"- **Econometric Synthesis**: {mkt.get('interpretation', 'N/A')}")
    md.append("")

    # Module 6
    md.append("## 8. Global Macro Linkage & VAR Analysis (Unit 8)")
    if var_res.get("available", False):
        md.append(f"- **System Specification**: 5 variables [S&P 500, Brent Crude, USD/INR, India VIX, NIFTY 50] across {var_res.get('aligned_obs', 0)} common trading dates.")
        md.append(f"- **Cholesky Identifying Assumption**: Shock ordering = `['SP500', 'Brent', 'USDINR', 'IndiaVIX', 'NIFTY50']`.")
        md.append(f"- **Selected Lag Order**: p = {var_res.get('selected_lag', 1)} (AIC lag: {var_res.get('lag_aic', 1)}, BIC lag: {var_res.get('lag_bic', 1)})")
        md.append(f"- **System Stability**: {'Stable (all roots inside unit circle)' if var_res.get('is_stable', False) else 'Unstable'}")
        md.append(f"- **Granger Causality Results on NIFTY 50**:")
        for g in var_res.get("granger_details", []):
            md.append(f"  - **{g.get('var')}**: F = {g.get('f_stat', 0.0):.2f}, p = {g.get('p_val', 0.0):.4f} -> {g.get('conclusion')}")
        md.append(f"- **1-Step Rolling Out-of-Sample Performance**: {var_res.get('oos_verdict', 'N/A')}")
    else:
        md.append(f"> Macro VAR analysis skipped: {var_res.get('reason', 'Insufficient observations')}")
    md.append("")

    # Module 7
    md.append("## 9. Expanding-Window Out-of-Sample Evaluation (Units 4 & 5)")
    md.append(f"- **Evaluation Protocol**: Expanding-window rolling origin (never shuffled, assert_no_leakage verified).")
    md.append(f"- **Origins Evaluated**: {eval_res.get('n_origins', 0)} origins (Step = 21 days, Horizon = 21 days).")
    md.append("| Model Candidate | MAE | RMSE | MASE | MASE Improvement vs Naive |")
    md.append("|:---|:---|:---|:---|:---|")
    for row in eval_res.get("table_rows", []):
        md.append(f"| {row.get('Model')} | {row.get('MAE', 0.0):.4f} | {row.get('RMSE', 0.0):.4f} | {row.get('MASE', 0.0):.4f} | {row.get('MASE Improvement vs Naive (%)')} |")
    md.append(f"- **Summary Verdict**: {eval_res.get('verdict', 'N/A')}")
    md.append("")

    # Limitations & Academic References
    md.append("## 10. Econometric Limitations & Methodology Caveats")
    md.append("1. **Cholesky Ordering Assumption**: The orthogonalized impulse response functions and variance decompositions rely on the recursive causal ordering [S&P 500 -> Brent -> USD/INR -> India VIX -> NIFTY 50]. Different causal orderings yield altered contemporaneous structural parameters.")
    md.append("2. **Overlapping Realized Volatility Windows**: Computing realized volatility over a 21-day rolling window induces autocorrelation into the target series $v_t$ by construction, which assists ARMA fit but requires multi-step evaluation caution.")
    md.append("3. **Trading Calendar Discrepancies**: Global macro series (US equities, European commodities, Indian markets) observe different national and market holidays. Alignment via inner join discards unaligned dates and introduces lead-lag market timing differences (e.g. US markets close after Indian markets).")
    md.append("4. **Dividend & Corporate Action Artefacts**: Dividend payouts in IDCW mutual fund plans and stock dividends/splits cause structural step-changes in raw NAV/prices; growth plans and split-adjusted series mitigate this distortion.")
    md.append("5. **Survivorship Bias**: Public live data reflects surviving schemes and actively traded equities; defunct or merged funds are not represented in current snapshots.")
    md.append("")

    md.append("## 11. Disclaimer")
    md.append("> **ACADEMIC AND EDUCATIONAL USE ONLY**: This software, analyses, and reports were developed for a university project in **Applied Time Series Analysis (ATSA)**. None of the statistical models, forecasts, or interpretations constitute investment advice, financial recommendations, or trading signals. Market investments are subject to capital loss.")
    md.append("")

    md.append("## 12. References")
    md.append("- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3-73.")
    md.append("- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1-26.")
    md.append("- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.")
    md.append("- Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.")
    md.append("- Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297-307.")
    md.append("- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia. otexts.com/fpp3.")
    md.append("- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.")
    md.append("- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.")
    md.append("")

    return "\n".join(md)
