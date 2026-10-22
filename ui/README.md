# MarketFolio Presentation Layer: Object to Fields Reference

This document maps all core model output objects and NamedTuples from `src/risklens/` to their exposed fields.

| Object / NamedTuple | Defined In | Exposed Fields | Type & Description |
| :--- | :--- | :--- | :--- |
| **CleanedSeriesResult** | `src/risklens/cleaning.py` | `price_series` | `pd.Series` (DatetimeIndex, cleaned closing price / NAV) |
| | | `log_return_series` | `pd.Series` (DatetimeIndex, $r_t = \ln(P_t/P_{t-1})$) |
| | | `modeling_series` | `pd.Series` (RangeIndex series for statsmodels) |
| | | `dates` | `np.ndarray` (array of string dates corresponding to observations) |
| | | `flagged_anomalies` | `pd.DataFrame` (columns: `date`, `return`, `abs_return`, `note` for $\|r_t\| > 0.15$) |
| | | `n_original_obs` | `int` (raw observations received from provider) |
| | | `n_cleaned_obs` | `int` (observations remaining after alignment and cleaning) |
| | | `n_gaps_filled` | `int` (forward-filled business day gaps $\le 3$ days) |
| | | `n_dropped_after_gap` | `int` (observations dropped due to extended gaps $> 3$ days) |
| **UnitRootTestResult** | `src/risklens/diagnostics.py` | `series_name` | `str` (identifier of the series) |
| | | `adf_stat` | `float` (Augmented Dickey-Fuller test statistic) |
| | | `adf_pvalue` | `float` (MacKinnon approximate p-value for ADF) |
| | | `adf_lags` | `int` (AIC-selected lag order used in ADF regression) |
| | | `adf_nobs` | `int` (sample size used in ADF test) |
| | | `adf_crit_values` | `dict[str, float]` (1%, 5%, 10% critical thresholds) |
| | | `kpss_stat` | `float` (KPSS test statistic) |
| | | `kpss_pvalue` | `float` (KPSS asymptotic p-value) |
| | | `kpss_lags` | `int` (Hobijn bandwidth lag truncation) |
| | | `kpss_crit_values` | `dict[str, float]` (10%, 5%, 2.5%, 1% critical thresholds) |
| | | `kpss_truncated` | `bool` (whether p-value hit table bounds 0.01 or 0.10) |
| | | `verdict` | `str` (`"Stationary (I(0))"`, `"Difference Stationary (I(1))"`, etc.) |
| | | `explanation` | `str` (econometric interpretation of joint ADF/KPSS decision) |
| **CalendarEffectResult** | `src/risklens/seasonality.py` | `ols_f_stat` | `float` (joint F-statistic on day-of-week dummy variables) |
| | | `ols_f_pvalue` | `float` (p-value for joint day-of-week significance) |
| | | `kruskal_stat` | `float` (Kruskal-Wallis non-parametric H-statistic) |
| | | `kruskal_pvalue` | `float` (p-value for Kruskal-Wallis test across weekdays) |
| | | `is_day_of_week_significant` | `bool` (True if joint F or Kruskal $p < 0.05$) |
| | | `summary_verdict` | `str` (plain-text conclusion on calendar effects) |
| | | `dow_stats` | `pd.DataFrame` (columns: `day_name`, `count`, `mean`, `std`, `se`, `ci_lower`, `ci_upper`) |
| | | `month_stats` | `pd.DataFrame` (columns: `month_name`, `count`, `mean`, `std`, `se`, `ci_lower`, `ci_upper`) |
| **STLDecompositionResult** | `src/risklens/seasonality.py` | `trend_strength_f_t` | `float` ($F_T = \max(0, 1 - \frac{\text{Var}(R)}{\text{Var}(T+R)})$) |
| | | `seasonal_strength_f_s` | `float` ($F_S = \max(0, 1 - \frac{\text{Var}(R)}{\text{Var}(S+R)})$) |
| | | `trend` | `pd.Series` (extracted STL trend component) |
| | | `seasonal` | `pd.Series` (extracted 21-day seasonal cycle component) |
| | | `residual` | `pd.Series` (remainder / irregular component) |
| | | `verdict` | `str` (qualitative summary of trend vs seasonal domination) |
| **ARMAModelSummary** | `src/risklens/arma_models.py` | `order` | `tuple[int, int, int]` ((p, d, q) order) |
| | | `aic`, `aicc`, `bic` | `float` (information criteria metrics) |
| | | `params`, `bse`, `pvalues` | `pd.Series` (coefficient estimates, standard errors, p-values) |
| | | `conf_int` | `pd.DataFrame` (95% parameter confidence intervals) |
| | | `lb_df` | `pd.DataFrame` (Ljung-Box diagnostic table across lags) |
| | | `jb_stat`, `jb_pvalue` | `float` (Jarque-Bera residual normality statistic and p-value) |
| | | `jb_skew`, `jb_kurtosis` | `float` (residual sample skewness and kurtosis) |
| | | `residuals` | `pd.Series` (standardized model residuals) |
| | | `fittedvalues` | `pd.Series` (in-sample fitted conditional mean values) |
| | | `model_result` | `ARIMAResults` (statsmodels internal result wrapper) |
| **GridSearchResult** | `src/risklens/arma_models.py` | `best_order` | `tuple[int, int, int]` (lowest AICc order) |
| | | `best_model` | `ARMAModelSummary` (summary of the optimal model) |
| | | `constant_model` | `ARMAModelSummary` (summary of constant-drift baseline (0,0,0)) |
| | | `all_models_df` | `pd.DataFrame` (comparison table with order, AICc, BIC, convergence) |
| | | `failed_orders` | `list[tuple[int, int, int]]` (orders that encountered non-convergence) |
| | | `predictability_verdict` | `str` (EMH weak-form predictability finding) |
| | | `is_predictable` | `bool` (True if $\Delta\text{AICc} < -2.0$ vs constant model) |
| **VolatilityForecastResult**| `src/risklens/volatility.py` | `realized_vol_series` | `pd.Series` (21-day rolling annualized realized volatility) |
| | | `log_vol_series` | `pd.Series` (natural log of realized volatility: $\ln(\sigma_t)$) |
| | | `d_order` | `int` (integration order determined by ADF/KPSS on log vol) |
| | | `best_order` | `tuple[int, int, int]` (selected ARIMA order for log volatility) |
| | | `last_observed_vol_pct` | `float` (latest rolling realized volatility in percent p.a.) |
| | | `horizon_days` | `int` (forecast horizon in trading days, e.g. 5 to 60) |
| | | `analytic_point_pred` | `np.ndarray` (point forecasts in natural percentage space) |
| | | `analytic_lower_90`, `upper_90` | `np.ndarray` (analytic normal 90% confidence envelope) |
| | | `boot_median` | `np.ndarray` (bootstrap median path across $B=2000$ iterations) |
| | | `boot_lower_90`, `upper_90` | `np.ndarray` (5th and 95th empirical bootstrap percentiles) |
| | | `boot_failed_or_fallback` | `bool` (True if simulation failed or fell back to analytic) |
| | | `fallback_reason` | `str` (explanation if bootstrap fallback was engaged) |
| | | `expected_risk_range_pct` | `tuple[float, float]` ((lower_90, upper_90) at final horizon day) |
| | | `point_forecast_pct` | `float` (median expected volatility at final horizon day) |
| | | `naive_forecast_pct` | `float` (random walk baseline: constant volatility) |
| | | `rolling_mean_forecast_pct` | `float` (21-day historical moving average baseline) |
| | | `model_fit` | `ARIMAResults` (statsmodels fitted model on log volatility) |
| **MarketRegressionResult** | `src/risklens/regression_arma.py` | `asset_name`, `benchmark_name`| `str` (asset and benchmark tickers) |
| | | `n_aligned_obs` | `int` (overlapping trading days evaluated) |
| | | `start_date`, `end_date` | `str` (sample time window) |
| | | `ols_alpha_daily`, `annualized` | `float` (OLS intercept, daily and annualized $\times 252$) |
| | | `ols_alpha_pvalue` | `float` (HAC p-value for OLS alpha) |
| | | `ols_beta`, `ols_beta_se_hac` | `float` (OLS slope with Newey-West HAC standard error) |
| | | `ols_beta_pvalue` | `float` (p-value for OLS beta $\ne 0$) |
| | | `ols_r_squared`, `ols_dw_stat` | `float` ($R^2$ fit quality and Durbin-Watson autocorrelation stat) |
| | | `ols_lb_pvalue` | `float` (Ljung-Box p-value on OLS residuals) |
| | | `best_arma_order` | `tuple[int, int, int]` (ARMA order of disturbance process) |
| | | `arma_alpha_daily`, `annualized`| `float` (SARIMAX disturbance-adjusted alpha) |
| | | `arma_alpha_se`, `pvalue` | `float` (standard error and p-value of SARIMAX alpha) |
| | | `arma_alpha_ci` | `tuple[float, float]` (95% confidence interval for SARIMAX alpha) |
| | | `arma_beta`, `arma_beta_se` | `float` (SARIMAX disturbance-adjusted beta and SE) |
| | | `arma_beta_pvalue`, `ci` | `float`, `tuple[float, float]` (p-value and 95% CI for beta) |
| | | `arma_aicc` | `float` (AICc of SARIMAX specification) |
| | | `sarimax_fit` | `SARIMAXResults` (statsmodels fitted object) |
| | | `interpretation` | `str` (plain-language summary of beta and alpha findings) |
| **GrangerTestResult** | `src/risklens/var_analysis.py` | `causing_variable`, `target_variable` | `str` (macro input driver and response asset, e.g. NIFTY 50) |
| | | `f_stat`, `p_value` | `float` (Wald/F statistic and p-value for restriction test) |
| | | `df_num`, `df_denom` | `int` (degrees of freedom for F distribution) |
| | | `is_causal_05` | `bool` (True if $p < 0.05$, indicating lead-lag predictive power) |
| | | `conclusion` | `str` (plain-text verdict on directional transmission) |
| **VARAnalysisResult** | `src/risklens/var_analysis.py` | `aligned_obs` | `int` (5-variable synchronized observation count) |
| | | `start_date`, `end_date` | `str` (sample window) |
| | | `stationarity_table` | `pd.DataFrame` (ADF test summary on each macro variable) |
| | | `lag_order_aic`, `bic` | `int` (information criteria optimal lags) |
| | | `selected_lag` | `int` (parsimonious lag selected for VAR estimation) |
| | | `is_stable` | `bool` (True if all companion matrix roots lie strictly inside unit circle) |
| | | `roots` | `np.ndarray` (moduli of dynamic eigenvalues) |
| | | `granger_tests` | `list[GrangerTestResult]` (causality results for each macro variable) |
| | | `fevd_nifty_horizon10` | `pd.DataFrame` (10-day forecast error variance decomposition table) |
| | | `var_rmse_1step`, `mae_1step` | `float` (out-of-sample 1-step VAR prediction errors) |
| | | `baseline_rmse_1step`, `mae` | `float` (competing random walk baseline errors) |
| | | `var_beats_baseline` | `bool` (True if VAR RMSE < baseline RMSE) |
| | | `out_of_sample_verdict` | `str` (forecast performance summary) |
| | | `var_fit` | `VARResults` (statsmodels fitted VAR model) |
| | | `cholesky_order` | `list[str]` (causal ordering for orthogonalized IRF) |
| **RollingOriginMetrics** | `src/risklens/evaluation.py` | `model_name` | `str` (name of model or baseline) |
| | | `mae` | `float` (Mean Absolute Error across origins) |
| | | `rmse` | `float` (Root Mean Squared Error across origins) |
| | | `mase` | `float` (Mean Absolute Scaled Error relative to in-sample naive) |
| | | `mase_improvement_pct` | `float` (percentage improvement over naive baseline: $(1 - \text{MASE})\times 100$) |
| **BacktestResult** | `src/risklens/evaluation.py` | `target_name` | `str` (name of target series, e.g. Volatility) |
| | | `n_origins_evaluated` | `int` (number of expanding-window out-of-sample origins) |
| | | `failed_origins_count` | `int` (origins skipped due to numerical non-convergence) |
| | | `horizon` | `int` (evaluation forecast horizon in days) |
| | | `initial_window`, `step_size`| `int` (burn-in training observations and step size) |
| | | `metrics_table` | `pd.DataFrame` (comparison table: Model vs Naive vs Rolling Mean) |
| | | `model_metrics` | `RollingOriginMetrics` (primary model performance) |
| | | `naive_metrics` | `RollingOriginMetrics` (naive random walk baseline) |
| | | `rolling_mean_metrics` | `RollingOriginMetrics` (historical rolling mean baseline) |
| | | `summary_verdict` | `str` (plain-text conclusion on out-of-sample skill) |
