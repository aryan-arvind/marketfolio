"""
MarketFolio Tab: Deep Variational Autoencoder (VAE) Anomaly Engine.
Visualizes multi-channel financial reconstruction error, P95 train threshold,
latent space manifold, top historical market dislocations, and comparative baseline scorecard.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.theme import ACCENT_CYAN, ACCENT_TEAL, COLOR_BORDER, COLOR_CARD, COLOR_LOSS, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the VAE Anomaly Engine tab."""
    render_method_tag("Variational Autoencoder (VAE) · Multi-Channel Reconstruction Anomaly Detection")

    vae_res = data.get("vae_res")
    if not vae_res:
        st.info("VAE Anomaly Engine did not produce results for this asset run.")
        return

    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Reconstruction Threshold",
            value=f"{vae_res.reconstruction_threshold:.4f}",
            subtext="Train Set P95 Percentile cutoff",
            value_color=ACCENT_CYAN,
        )
    with c2:
        render_kpi_card(
            title="Total Anomalies Flagged",
            value=f"{vae_res.n_anomalies_detected}",
            subtext=f"{vae_res.anomaly_rate_pct:.1f}% of total sequence windows",
            value_color=COLOR_LOSS if vae_res.n_anomalies_detected > 0 else ACCENT_TEAL,
        )
    with c3:
        render_kpi_card(
            title="Latent Dimension",
            value=f"d_z = {vae_res.model_summary.get('latent_dim', 4)}",
            subtext=f"Gaussian Prior N(0, I)",
        )
    with c4:
        render_kpi_card(
            title="Input Window",
            value=f"{vae_res.model_summary.get('lookback', 15)} Days",
            subtext=f"{vae_res.model_summary.get('input_dim', 120)} features flattened",
        )

    # Narrative Interpretation
    render_takeaway_line(
        takeaway=vae_res.interpretation,
        detail=f"Reconstruction loss measures deviation from the multi-channel manifold learned during the pre-split training period.",
    )

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Chart 1: Temporal Reconstruction Loss Timeline vs Train P95 Threshold
    st.markdown('<div class="mf-card-title">Multi-Channel Reconstruction Error vs Anomaly Threshold</div>', unsafe_allow_html=True)
    df = vae_res.timeline_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    # Normal points line
    fig.add_trace(
        go.Scatter(
            x=df["date_dt"],
            y=df["vae_recon_error"],
            mode="lines",
            name="VAE Reconstruction Error",
            line=dict(color="#38bdf8", width=1.5),
        )
    )

    # Anomaly threshold horizontal line
    fig.add_trace(
        go.Scatter(
            x=[df["date_dt"].min(), df["date_dt"].max()],
            y=[vae_res.reconstruction_threshold, vae_res.reconstruction_threshold],
            mode="lines",
            name=f"Train P95 Threshold ({vae_res.reconstruction_threshold:.4f})",
            line=dict(color="#f43f5e", width=2, dash="dash"),
        )
    )

    # Anomalies markers
    anom_points = df[df["is_vae_anomaly"]]
    if not anom_points.empty:
        fig.add_trace(
            go.Scatter(
                x=anom_points["date_dt"],
                y=anom_points["vae_recon_error"],
                mode="markers",
                name="Flagged Anomaly Windows",
                marker=dict(color="#e11d48", size=8, symbol="diamond", line=dict(color="#ffffff", width=1)),
                text=[f"Date: {d}<br>Error: {e:.4f}<br>Return: {r:.2f}%" for d, e, r in zip(anom_points["date"], anom_points["vae_recon_error"], anom_points["return_pct"])],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0c1017",
        plot_bgcolor="#111827",
        margin=dict(l=40, r=20, t=30, b=30),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#1e293b", title="Calendar Date"),
        yaxis=dict(gridcolor="#1e293b", title="Reconstruction MSE Loss"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2: 2D Latent Space Projection & Comparative Baseline Scorecard
    col_latent, col_score = st.columns([1, 1])

    with col_latent:
        st.markdown('<div class="mf-card-title">Latent Manifold Projection (z₁ vs z₂)</div>', unsafe_allow_html=True)
        fig_lat = go.Figure()

        normal_latent = df[~df["is_vae_anomaly"]]
        anom_latent = df[df["is_vae_anomaly"]]

        fig_lat.add_trace(
            go.Scatter(
                x=normal_latent["latent_z1"],
                y=normal_latent["latent_z2"],
                mode="markers",
                name="Normal Regime Windows",
                marker=dict(color="#06b6d4", size=5, opacity=0.4),
            )
        )
        if not anom_latent.empty:
            fig_lat.add_trace(
                go.Scatter(
                    x=anom_latent["latent_z1"],
                    y=anom_latent["latent_z2"],
                    mode="markers",
                    name="Structural Dislocation Windows",
                    marker=dict(color="#f43f5e", size=8, symbol="x", line=dict(width=1.5)),
                    text=anom_latent["date"],
                    hoverinfo="text",
                )
            )

        fig_lat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0c1017",
            plot_bgcolor="#111827",
            margin=dict(l=30, r=20, t=30, b=30),
            height=340,
            xaxis=dict(gridcolor="#1e293b", title="Latent Mean μ_z₁"),
            yaxis=dict(gridcolor="#1e293b", title="Latent Mean μ_z₂"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_score:
        st.markdown('<div class="mf-card-title">Out-of-Sample Anomaly Scorecard vs Baselines</div>', unsafe_allow_html=True)
        st.dataframe(
            vae_res.baseline_scorecard,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<div style="font-size:0.78rem; color:#94a3b8; line-height:1.4; margin-top:0.5rem;">'
            '<b>Analytical takeaway:</b> Univariate 3-Sigma filters flag only sudden single-day price jumps, '
            'whereas the Deep VAE detects structural covariance breakdowns across momentum, volatility clustering, '
            'and higher moments that persist across multi-day lookback windows.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Table of Top Flagged Historical Anomalies
    if vae_res.top_anomalies:
        st.markdown('<div class="mf-card-title" style="margin-top:1.5rem;">Top Flagged Market Dislocation Windows</div>', unsafe_allow_html=True)
        st.caption(
            "ℹ️ **Academic Interpretability Clarification:** The VAE detects empirical reconstruction loss exceeding the train P95 threshold. "
            "Historical event annotations are post-hoc calendar references provided for external market context, not model-inferred causal deductions."
        )
        top_df = pd.DataFrame(vae_res.top_anomalies)
        top_df = top_df.rename(
            columns={
                "date": "Date",
                "split": "Data Split",
                "return_pct": "Daily Return %",
                "reconstruction_error": "Recon MSE",
                "threshold_ratio": "Loss / Threshold",
                "matched_event": "Historical Reference (Calendar Context / Non-Causal)",
                "contextual_reference": "Historical Reference (Calendar Context / Non-Causal)",
                "is_zscore_confirmed": "3σ Z-Score",
                "is_iso_confirmed": "Isolation Forest",
            }
        )
        st.dataframe(top_df, use_container_width=True, hide_index=True)
