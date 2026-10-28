"""
MarketFolio Tab: Volatility Regime Transformer Classifier.
Visualizes multi-head self-attention sequence attribution, out-of-sample confusion matrix,
loss histories, and rigorous comparative scorecards against Martingale and Logistic baselines.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.theme import ACCENT_CYAN, ACCENT_TEAL, COLOR_BORDER, COLOR_GAIN, COLOR_LOSS, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Volatility Regime Transformer tab."""
    render_method_tag("Transformer Encoder · Multi-Head Self-Attention Regime Classification")

    trans_res = data.get("transformer_res")
    if not trans_res:
        st.info("Transformer Volatility Classifier did not produce results for this asset run.")
        return

    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Out-of-Sample Macro F1",
            value=f"{trans_res.macro_f1:.4f}",
            subtext="Harmonic mean of class F1s",
            value_color=ACCENT_CYAN,
        )
    with c2:
        render_kpi_card(
            title="Balanced Accuracy",
            value=f"{trans_res.balanced_accuracy * 100.0:.1f}%",
            subtext="Adjusted for regime prevalence",
        )
    with c3:
        render_kpi_card(
            title="Test Accuracy",
            value=f"{trans_res.accuracy * 100.0:.1f}%",
            subtext="Exact match ratio on test split",
        )
    with c4:
        # Delta vs persistence baseline
        persist_f1 = next((b.macro_f1 for b in trans_res.baselines if "Persistence" in b.name), 0.0)
        delta_persist = trans_res.macro_f1 - persist_f1
        render_kpi_card(
            title="Delta vs Persistence",
            value=f"{delta_persist:+.4f}",
            subtext="vs Last-Known Regime Martingale",
            value_color=COLOR_GAIN if delta_persist >= 0 else COLOR_LOSS,
        )

    # Dynamic Analytical Interpretation
    render_takeaway_line(
        takeaway=f"{trans_res.verdict}: {trans_res.interpretation}",
        detail=f"Evaluated on {len(trans_res.y_test_true)} chronological out-of-sample trading windows with strict embargo purging.",
    )

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Row 1: Self-Attention Temporal Weights & Confusion Matrix
    col_attn, col_cm = st.columns([1.2, 1.0])

    with col_attn:
        st.markdown('<div class="mf-card-title">Multi-Head Self-Attention Temporal Attribution</div>', unsafe_allow_html=True)
        attns = trans_res.mean_attention_weights
        lookback_lags = [f"t-{len(attns) - 1 - i}" if i < len(attns) - 1 else "t (Current)" for i in range(len(attns))]

        fig_attn = go.Figure()
        fig_attn.add_trace(
            go.Bar(
                x=lookback_lags,
                y=attns,
                marker=dict(
                    color=attns,
                    colorscale=[[0, "#0e7490"], [0.5, "#06b6d4"], [1, "#38bdf8"]],
                ),
                text=[f"{a*100.0:.1f}%" for a in attns],
                textposition="auto",
            )
        )
        fig_attn.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0c1017",
            plot_bgcolor="#111827",
            margin=dict(l=30, r=20, t=30, b=30),
            height=320,
            xaxis=dict(gridcolor="#1e293b", title="Temporal Lookback Lag"),
            yaxis=dict(gridcolor="#1e293b", title="Mean Attention Weight"),
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption("Attention weights identify which past trading days the Transformer emphasized when forming future regime projections.")

    with col_cm:
        st.markdown('<div class="mf-card-title">Out-of-Sample Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(trans_res.confusion_matrix)
        labels = ["Low Vol", "Normal Vol", "High Vol"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#111827"], [0.5, "#0e7490"], [1, "#06b6d4"]],
                text=[[str(val) for val in row] for row in cm],
                texttemplate="%{text}",
                textfont=dict(size=14, color="#ffffff"),
            )
        )
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0c1017",
            plot_bgcolor="#111827",
            margin=dict(l=30, r=20, t=30, b=30),
            height=320,
            xaxis=dict(title="Predicted Regime", gridcolor="#1e293b"),
            yaxis=dict(title="True Regime", gridcolor="#1e293b", autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonal elements represent correct out-of-sample regime classifications.")

    # Row 2: Comparative Scorecard Table
    st.markdown('<div class="mf-card-title" style="margin-top:1.5rem;">Academic Baseline Comparison Scorecard</div>', unsafe_allow_html=True)
    st.dataframe(
        trans_res.scorecard_df,
        use_container_width=True,
        hide_index=True,
    )

    # Class-Level Performance Metrics
    st.markdown('<div class="mf-card-title" style="margin-top:1.5rem;">Per-Regime Detailed Classification Report</div>', unsafe_allow_html=True)
    cr = trans_res.class_report
    report_rows = []
    for regime_name in ["Low Vol", "Normal Vol", "High Vol"]:
        if regime_name in cr:
            row_data = cr[regime_name]
            report_rows.append({
                "Regime": regime_name,
                "Precision": f"{row_data.get('precision', 0.0) * 100.0:.1f}%",
                "Recall": f"{row_data.get('recall', 0.0) * 100.0:.1f}%",
                "F1-Score": f"{row_data.get('f1-score', 0.0):.4f}",
                "Support (Windows)": int(row_data.get("support", 0)),
            })

    if report_rows:
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
