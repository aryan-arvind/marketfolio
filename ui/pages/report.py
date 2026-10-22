"""
MarketFolio Tab 7: Research Note & Report Generation.
Compiles a structured institutional memo with findings, diagnostics, and data provenance.
"""

from __future__ import annotations

import re
from typing import Any
import streamlit as st

from risklens.report import build_markdown_report
from ui.components import render_method_tag
from ui.theme import ACCENT_TEAL, TEXT_MUTED


def _clean_narrative(text: str) -> str:
    """Cleans duplicated names, broken math formatting, and legacy names in narrative."""
    # Fix duplicated symbols e.g. "SYMBOL (SYMBOL)" -> "SYMBOL"
    cleaned = re.sub(r"([A-Za-z0-9_\.\^]+)\s*\(\1\)", r"\1", text)
    # Fix broken math duplicates e.g. "I(1)I(1)" -> "I(1)"
    cleaned = cleaned.replace("I(1)I(1)", "I(1)").replace("I(0)I(0)", "I(0)")
    # Replace any legacy RiskLens branding with MarketFolio
    cleaned = cleaned.replace("RiskLens", "MarketFolio").replace("risklens", "marketfolio")
    return cleaned


def render(data: dict[str, Any]) -> None:
    """Renders the Report tab."""
    render_method_tag("Institutional Research Note Synthesis & Audit Package")

    symbol = data["symbol_code"]
    display_name = data["display_name"]
    assembled = data["assembled_results"]
    raw_narrative = data["narrative"]

    clean_narr = _clean_narrative(raw_narrative)
    report_md = build_markdown_report(assembled, clean_narr)
    clean_report_md = _clean_narrative(report_md)

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        st.download_button(
            label="Download Research Memo",
            data=clean_report_md,
            file_name=f"marketfolio_memo_{symbol}.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption("Standard Markdown format with full parameter disclosure, LaTeX formulas, and diagnostic logs.")

    st.markdown("---")

    # Render clean research note container
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #23314a; border-radius: 8px;
                    padding: 2rem 2.5rem; margin-top: 1rem; color: #cbd5e1; line-height: 1.65;">
            {clean_report_md}
        </div>
        """,
        unsafe_allow_html=True,
    )
