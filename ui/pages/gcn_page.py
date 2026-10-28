"""
MarketFolio Tab: Inter-Market Graph Convolutional Network (GCN).
Visualizes relational graph topology across 6 asset nodes, normalized Laplacian adjacency,
systemic stress classification metrics, and node vulnerability / centrality scores.
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
    """Renders the Inter-Market GCN tab."""
    render_method_tag("Graph Convolutional Network (GCN) · Inter-Market Systemic Risk & Topology")

    gcn_res = data.get("gcn_res")
    if not gcn_res or not gcn_res.available:
        reason = getattr(gcn_res, "reason", "Insufficient synchronized multi-asset observations.")
        st.info(f"Inter-Market GCN not available for this run: {reason}")
        return

    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Systemic F1-Score",
            value=f"{gcn_res.systemic_stress_f1:.4f}",
            subtext="GCN out-of-sample macro F1",
            value_color=ACCENT_CYAN,
        )
    with c2:
        render_kpi_card(
            title="Systemic Accuracy",
            value=f"{gcn_res.systemic_stress_accuracy * 100.0:.1f}%",
            subtext=f"vs Baseline {gcn_res.baseline_accuracy * 100.0:.1f}%",
        )
    with c3:
        # Top systemic node
        top_node = max(gcn_res.node_vulnerability_scores, key=gcn_res.node_vulnerability_scores.get)
        render_kpi_card(
            title="Primary Propagator",
            value=top_node,
            subtext=f"{gcn_res.node_vulnerability_scores[top_node]:.1f}% transmission weight",
            value_color=COLOR_LOSS,
        )
    with c4:
        render_kpi_card(
            title="Graph Nodes",
            value=f"{len(gcn_res.nodes)} Assets",
            subtext="Synchronized cross-market system",
        )

    # Analytical Interpretation
    render_takeaway_line(
        takeaway=gcn_res.interpretation,
        detail="Graph spectral convolution models spatial feature aggregation across interconnected markets under joint shock transmission.",
    )

    if getattr(gcn_res, "embargo_days", 0) > 0 and getattr(gcn_res, "train_date_range", None):
        st.caption(
            f"🛡️ **Zero-Leakage Guarantee:** An explicit {gcn_res.embargo_days}-day embargo gap was applied between "
            f"Training ({gcn_res.train_date_range[0]} to {gcn_res.train_date_range[1]}) and Testing ({gcn_res.test_date_range[0]} to {gcn_res.test_date_range[1]}) "
            f"so forward systemic volatility labels never consume test period information."
        )

    st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

    col_adj, col_nodes = st.columns([1.1, 1.0])

    with col_adj:
        st.markdown('<div class="mf-card-title">Normalized Relational Adjacency Matrix (Â)</div>', unsafe_allow_html=True)
        adj_mat = np.array(gcn_res.adjacency_matrix)
        nodes = gcn_res.nodes

        fig_adj = go.Figure(
            data=go.Heatmap(
                z=adj_mat,
                x=nodes,
                y=nodes,
                colorscale=[[0, "#111827"], [0.5, "#0e7490"], [1, "#06b6d4"]],
                text=[[f"{v:.2f}" for v in row] for row in adj_mat],
                texttemplate="%{text}",
                textfont=dict(size=11, color="#ffffff"),
            )
        )
        fig_adj.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0c1017",
            plot_bgcolor="#111827",
            margin=dict(l=30, r=20, t=30, b=30),
            height=340,
            xaxis=dict(gridcolor="#1e293b", tickangle=-25),
            yaxis=dict(gridcolor="#1e293b", autorange="reversed"),
        )
        st.plotly_chart(fig_adj, use_container_width=True)

    with col_nodes:
        st.markdown('<div class="mf-card-title">Node Systemic Vulnerability & Transmission Ranking</div>', unsafe_allow_html=True)
        scores_sorted = sorted(gcn_res.node_vulnerability_scores.items(), key=lambda x: x[1], reverse=True)
        node_labels = [x[0] for x in scores_sorted]
        node_vals = [x[1] for x in scores_sorted]

        fig_nodes = go.Figure()
        fig_nodes.add_trace(
            go.Bar(
                x=node_vals,
                y=node_labels,
                orientation="h",
                marker=dict(
                    color=node_vals,
                    colorscale=[[0, "#0e7490"], [1, "#f43f5e"]],
                ),
                text=[f"{v:.1f}%" for v in node_vals],
                textposition="auto",
            )
        )
        fig_nodes.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0c1017",
            plot_bgcolor="#111827",
            margin=dict(l=30, r=20, t=30, b=30),
            height=340,
            xaxis=dict(title="Systemic Vulnerability Score (%)", gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", autorange="reversed"),
        )
        st.plotly_chart(fig_nodes, use_container_width=True)

    # Comparative scorecard
    st.markdown('<div class="mf-card-title" style="margin-top:1.5rem;">Systemic Risk Classification vs Baseline</div>', unsafe_allow_html=True)
    gcn_scorecard = pd.DataFrame([
        {
            "Architecture": "InterMarket GCN (2-Layer Spectral)",
            "Accuracy": f"{gcn_res.systemic_stress_accuracy * 100.0:.2f}%",
            "Macro F1": f"{gcn_res.systemic_stress_f1:.4f}",
            "Delta F1": f"{(gcn_res.systemic_stress_f1 - gcn_res.baseline_f1):+.4f}",
            "Mechanism": "Dynamic graph neighborhood aggregation + normalized Laplacian",
        },
        {
            "Architecture": "Majority Static Baseline",
            "Accuracy": f"{gcn_res.baseline_accuracy * 100.0:.2f}%",
            "Macro F1": f"{gcn_res.baseline_f1:.4f}",
            "Delta F1": "0.0000 (Reference)",
            "Mechanism": "Naive static class prediction without relational topology",
        },
    ])
    st.dataframe(gcn_scorecard, use_container_width=True, hide_index=True)
