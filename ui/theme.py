"""
MarketFolio Design System and Plotly Theme.
Near-black blue-gray palette, crisp cards, 1px borders, single teal accent.
Tabular numerals, no emojis, no glow, clean institutional desk aesthetic.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Color Tokens
BG_BASE = "#0b0f19"
BG_SIDEBAR = "#0c1322"
BG_CARD = "#111827"
BG_CARD_MUTED = "#151e32"
BG_INPUT = "#0f172a"
BORDER_COLOR = "#1e293b"
BORDER_CARD = "#23314a"
BORDER_HIGHLIGHT = "#334155"

TEXT_PRIMARY = "#f8fafc"
TEXT_SECONDARY = "#cbd5e1"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"

ACCENT_TEAL = "#06b6d4"
ACCENT_TEAL_DIM = "rgba(6, 182, 212, 0.15)"
COLOR_GAIN = "#10b981"
COLOR_GAIN_DIM = "rgba(16, 185, 129, 0.12)"
COLOR_LOSS = "#ef4444"
COLOR_LOSS_DIM = "rgba(239, 68, 68, 0.12)"
COLOR_WARNING = "#f59e0b"
COLOR_WARNING_DIM = "rgba(245, 158, 11, 0.12)"

GRID_COLOR = "#1e293b"

# Global CSS Theme
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: {TEXT_PRIMARY};
}}

/* Hide default Streamlit chrome */
#MainMenu, header, footer {{
    visibility: hidden !important;
    height: 0px !important;
}}
.stDeployButton {{
    display: none !important;
}}
div[data-testid="stToolbar"] {{
    display: none !important;
}}
div[data-testid="stDecoration"] {{
    display: none !important;
}}
div[data-testid="stStatusWidget"] {{
    visibility: hidden !important;
}}

/* App Container */
.stApp {{
    background-color: {BG_BASE} !important;
}}
[data-testid="stAppViewContainer"] {{
    background-color: {BG_BASE} !important;
    padding-top: 1rem !important;
}}
[data-testid="stSidebar"] {{
    background-color: {BG_SIDEBAR} !important;
    border-right: 1px solid {BORDER_COLOR} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: transparent;
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 0px;
    margin-bottom: 1.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    height: 38px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {TEXT_MUTED};
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    text-transform: none;
    transition: all 0.15s ease-in-out;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT_PRIMARY};
    background-color: rgba(255, 255, 255, 0.02);
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT_TEAL} !important;
    border-bottom: 2px solid {ACCENT_TEAL} !important;
    font-weight: 600 !important;
    background-color: transparent !important;
}}

/* Typography utilities */
.tabular-numbers {{
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}}

/* Header Strip Component */
.mf-header-strip {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_CARD};
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}}
.mf-header-left {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
}}
.mf-ticker {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.02em;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.mf-name {{
    font-size: 0.95rem;
    color: {TEXT_MUTED};
    font-weight: 400;
}}
.mf-header-price-block {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
}}
.mf-price {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}}
.mf-change {{
    font-size: 0.92rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}}
.mf-change-pos {{
    color: {COLOR_GAIN};
}}
.mf-change-neg {{
    color: {COLOR_LOSS};
}}

.mf-stats-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    padding-top: 0.5rem;
    border-top: 1px solid {BORDER_COLOR};
    width: 100%;
}}
.mf-stat-item {{
    display: flex;
    gap: 0.4rem;
}}
.mf-stat-label {{
    color: {TEXT_DIM};
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}}
.mf-stat-value {{
    color: {TEXT_SECONDARY};
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
}}

/* Card Container */
.mf-card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_CARD};
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
}}
.mf-card-title {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEXT_DIM};
    margin-bottom: 0.4rem;
}}
.mf-card-val {{
    font-size: 1.45rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}}
.mf-card-sub {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin-top: 0.25rem;
}}

/* Badge & Chip Styles */
.mf-badge {{
    display: inline-flex;
    align-items: center;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.mf-badge-live {{
    background-color: {COLOR_GAIN_DIM};
    color: {COLOR_GAIN};
    border: 1px solid rgba(16, 185, 129, 0.3);
}}
.mf-badge-stale {{
    background-color: {COLOR_WARNING_DIM};
    color: {COLOR_WARNING};
    border: 1px solid rgba(245, 158, 11, 0.3);
}}
.mf-badge-teal {{
    background-color: {ACCENT_TEAL_DIM};
    color: {ACCENT_TEAL};
    border: 1px solid rgba(6, 182, 212, 0.3);
}}
.mf-badge-neutral {{
    background-color: rgba(148, 163, 184, 0.1);
    color: {TEXT_MUTED};
    border: 1px solid {BORDER_CARD};
}}

/* Takeaway Line */
.mf-takeaway {{
    background-color: {BG_CARD_MUTED};
    border-left: 3px solid {ACCENT_TEAL};
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.95rem;
    font-size: 0.86rem;
    color: {TEXT_SECONDARY};
    margin: 0.65rem 0 1rem 0;
    line-height: 1.45;
}}

/* Method Caption */
.mf-method-tag {{
    font-size: 0.76rem;
    color: {TEXT_DIM};
    font-weight: 500;
    margin-top: -0.75rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}}

/* Standby Banner */
.mf-standby {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_CARD};
    border-radius: 8px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 800px;
}}

/* Table Overrides */
table {{
    font-size: 0.82rem !important;
    color: {TEXT_SECONDARY} !important;
}}
th {{
    background-color: {BG_CARD_MUTED} !important;
    color: {TEXT_DIM} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid {BORDER_COLOR} !important;
}}
td {{
    border-bottom: 1px solid {BORDER_COLOR} !important;
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
}}

/* Expander styling */
[data-testid="stExpander"] {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_CARD} !important;
    border-radius: 6px !important;
    margin-top: 0.75rem !important;
}}
[data-testid="stExpander"] details summary {{
    font-size: 0.82rem !important;
    color: {TEXT_MUTED} !important;
    padding: 0.6rem 0.8rem !important;
}}
[data-testid="stExpander"] details summary:hover {{
    color: {TEXT_PRIMARY} !important;
}}
</style>
"""


def apply_theme() -> None:
    """Configures the default Plotly template for MarketFolio."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=BG_BASE,
        plot_bgcolor=BG_BASE,
        font=dict(
            family="'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif",
            color=TEXT_MUTED,
            size=12,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_HIGHLIGHT,
            linecolor=BORDER_COLOR,
            tickcolor=BORDER_COLOR,
            tickfont=dict(size=10, family="'IBM Plex Mono', monospace"),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_HIGHLIGHT,
            linecolor=BORDER_COLOR,
            tickcolor=BORDER_COLOR,
            tickfont=dict(size=10, family="'IBM Plex Mono', monospace"),
            showgrid=True,
        ),
        hoverlabel=dict(
            bgcolor=BG_CARD_MUTED,
            bordercolor=BORDER_HIGHLIGHT,
            font=dict(
                family="'IBM Plex Mono', monospace",
                size=12,
                color=TEXT_PRIMARY,
            ),
        ),
        margin=dict(l=40, r=20, t=25, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_MUTED),
        ),
    )
    pio.templates["marketfolio_dark"] = template
    pio.templates.default = "marketfolio_dark"
