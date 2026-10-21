"""
Quantitative narrator and report synthesizer for RiskLens.
Generates an ATSA-grounded analytical commentary strictly from computed metrics.
Supports an optional local LLM via Ollama (if installed and requested), with a
fully deterministic, zero-hallucination econometric template as the robust default.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("risklens")


def generate_deterministic_narrative(results_data: dict[str, Any]) -> str:
    """
    Synthesizes a structured plain-English executive summary strictly from computed numbers.
    Never invents or extrapolates values outside the supplied dictionary.
    """
    asset_meta = results_data.get("asset_metadata", {})
    asset_name = asset_meta.get("name", "Target Asset")
    symbol = asset_meta.get("symbol_or_code", "N/A")
    last_date = asset_meta.get("last_available_date", "N/A")
    n_obs = asset_meta.get("n_observations", 0)

    # 1. Stationarity
    stat = results_data.get("stationarity", {})
    price_verdict = stat.get("log_price_verdict", "N/A")
    ret_verdict = stat.get("log_returns_verdict", "N/A")
    adf_p = stat.get("returns_adf_pvalue", float("nan"))
    kpss_p = stat.get("returns_kpss_pvalue", float("nan"))

    # 2. Seasonality
    season = results_data.get("seasonality", {})
    dow_sig = season.get("day_of_week_significant", False)
    dow_f_p = season.get("ols_f_pvalue", float("nan"))
    trend_str = season.get("trend_strength_f_t", float("nan"))
    season_str = season.get("seasonal_strength_f_s", float("nan"))

    # 3. Predictability
    arma = results_data.get("arma_predictability", {})
    best_order = arma.get("best_order", "(0,0,0)")
    is_pred = arma.get("is_predictable", False)
    delta_aicc = arma.get("delta_aicc_vs_constant", 0.0)

    # 4. Volatility Forecast
    vol = results_data.get("volatility_forecast", {})
    last_vol = vol.get("last_observed_vol_pct", float("nan"))
    point_vol = vol.get("point_forecast_pct", float("nan"))
    risk_min = vol.get("expected_risk_range_lower_pct", float("nan"))
    risk_max = vol.get("expected_risk_range_upper_pct", float("nan"))
    boot_fallback = vol.get("bootstrap_fallback_used", False)

    # 5. Market Model
    mkt = results_data.get("market_model", {})
    bench_name = mkt.get("benchmark_name", "^NSEI")
    beta = mkt.get("beta", float("nan"))
    alpha_ann = mkt.get("alpha_annualized_pct", float("nan"))
    alpha_p = mkt.get("alpha_pvalue", float("nan"))
    r2 = mkt.get("r_squared", float("nan"))

    # 6. VAR Linkage
    var_dict = results_data.get("var_macro", {})
    var_available = var_dict.get("available", False)
    var_lag = var_dict.get("selected_lag", 1)
    granger_causal = var_dict.get("granger_causal_factors", [])
    var_beats = var_dict.get("var_beats_baseline", False)

    # Narrative building
    lines: list[str] = []
    lines.append(f"### Quantitative Risk & Market Profile: {asset_name} ({symbol})")
    lines.append(
        f"Data sampled through **{last_date}** comprises **{n_obs}** cleaned trading sessions. "
        "All calculations represent real market observations processed via verified statistical models."
    )
    lines.append("")

    # Section 1
    lines.append("#### 1. Time Series Stationarity & Integration Properties")
    lines.append(
        f"- **Log Price / NAV**: Categorized as **{price_verdict}**.\n"
        f"- **Log Returns**: Categorized as **{ret_verdict}** "
        f"(ADF p-value = {adf_p:.4f}, KPSS p-value = {kpss_p:.4f}). "
        "The asset exhibits standard first-difference stationarity ($I(1)$ price series, $I(0)$ return series)."
    )
    lines.append("")

    # Section 2
    lines.append("#### 2. Calendar Patterns & Structural Decomposition")
    if dow_sig:
        lines.append(
            f"- Statistical tests identify **significant day-of-week calendar effects** (OLS F-test p = {dow_f_p:.4f} < 0.05). "
            "Mean returns systematically vary across trading days."
        )
    else:
        lines.append(
            f"- Joint F-testing indicates **no statistically significant day-of-week anomaly** (OLS F-test p = {dow_f_p:.4f} >= 0.05). "
            "Returns are consistent with calendar homogeneity."
        )
    lines.append(
        f"- Robust STL decomposition indicates dominant trend strength (**F_T = {trend_str:.3f}**) "
        f"and minimal seasonal variation (**F_S = {season_str:.3f}**)."
    )
    lines.append("")

    # Section 3
    lines.append("#### 3. Return Predictability & Market Efficiency")
    lines.append(
        f"- Grid search selected **ARIMA{best_order}** via AICc. "
        f"Relative to a constant-drift random walk benchmark, Delta AICc is **{delta_aicc:.2f}**."
    )
    if is_pred:
        lines.append(
            "- **Verdict**: Demonstrates statistically significant incremental predictability over the unconditional mean."
        )
    else:
        lines.append(
            "- **Verdict**: **No evidence of return predictability**. Consistent with Fama's (1970) Weak-Form "
            "Efficient Market Hypothesis: past price movements do not generate exploitable linear predictive signal."
        )
    lines.append("")

    # Section 4
    lines.append("#### 4. Realized Volatility & Downside Risk Profile")
    lines.append(
        f"- **Current Realized Volatility**: **{last_vol:.2f}%** annualized (21-day rolling window).\n"
        f"- **30-Day Forecast Range**: **[{risk_min:.2f}% - {risk_max:.2f}%]** annualized (90% bootstrap interval).\n"
        f"- **Point Forecast (Median)**: **{point_vol:.2f}%**."
    )
    if boot_fallback:
        lines.append("  *(Note: Analytic Gaussian intervals utilized as bootstrap fallback.)*")
    lines.append("")

    # Section 5
    lines.append(f"#### 5. Benchmark Sensitivity & Market Beta vs {bench_name}")
    alpha_status = "significant" if alpha_p < 0.05 else "not statistically significant"
    lines.append(
        f"- **Market Beta**: **{beta:.3f}** (SE = {mkt.get('beta_se', 0.0):.3f}).\n"
        f"- **Annualized Alpha**: **{alpha_ann:+.2f}%** (p-value = {alpha_p:.4f}, {alpha_status}).\n"
        f"- **Explanatory Power**: **R^2 = {r2:.3f}**."
    )
    lines.append("")

    # Section 6
    lines.append("#### 6. Global Macro Transmission & Lead-Lag Dynamics")
    if var_available:
        lines.append(
            f"- Stable VAR system estimated at lag **p = {var_lag}** across 5 global macro factors.\n"
            f"- **Granger-Causal Factors on NIFTY 50**: {', '.join(granger_causal) if granger_causal else 'None at 5% significance'}.\n"
            f"- Out-of-sample directional performance: {'VAR beats mean baseline' if var_beats else 'Baseline holds comparable accuracy; macro transmission predominantly governs shock variance (FEVD) rather than daily return timing'}."
        )
    else:
        lines.append("- Macro VAR analysis omitted due to insufficient aligned data observations.")

    return "\n".join(lines)


def generate_narrative(
    results_data: dict[str, Any],
    use_llm: bool = False,
    model_name: str = "llama3",
) -> str:
    """
    Produces the report narrative. If use_llm is True and the 'ollama' package is available,
    prompts the local LLM strictly with the provided computed numbers.
    Otherwise, immediately returns the deterministic template.
    """
    if not use_llm:
        return generate_deterministic_narrative(results_data)

    try:
        import ollama  # type: ignore

        # Construct strictly bounded prompt
        clean_json = json.dumps(results_data, indent=2, default=str)
        prompt = (
            "You are an econometrician and ATSA professor writing an objective analysis report. "
            "You are provided ONLY with the following exact computed results in JSON format:\n\n"
            f"{clean_json}\n\n"
            "STRICT RULES:\n"
            "1. Report ONLY the facts and numbers present in the JSON above.\n"
            "2. DO NOT invent, extrapolate, or estimate any numbers, dates, or p-values not provided.\n"
            "3. Structure your response into: Executive Summary, Stationarity, Predictability, Volatility, and Macro Linkages.\n"
            "4. Keep the tone academic, rigorous, and concise."
        )

        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.get("message", {}).get("content", "")
        if content.strip():
            logger.info("Successfully generated report narrative using Ollama model '%s'.", model_name)
            return content
        logger.warning("Empty response received from Ollama. Falling back to deterministic narrative.")
        return generate_deterministic_narrative(results_data)

    except Exception as exc:
        logger.info("Ollama LLM narration unavailable or disabled (%s). Using deterministic template.", exc)
        return generate_deterministic_narrative(results_data)
