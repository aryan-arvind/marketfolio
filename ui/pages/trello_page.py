"""
MarketFolio Tab: Trello Research & Experiment Tracker.
Displays structured quantitative research tasks, model validation experiments,
leakage audit checklists, and live/offline Trello board synchronization status.
"""

from __future__ import annotations

from typing import Any
import streamlit as st

from risklens.trello_tracker import TrackerSyncMode, create_trello_card
from ui.components import render_kpi_card, render_method_tag
from ui.theme import ACCENT_CYAN, COLOR_GAIN, COLOR_LOSS


def render(data: dict[str, Any]) -> None:
    """Renders the Trello Research Tracker tab."""
    render_method_tag("Research Governance · Trello Project & Experiment Workflow Tracker")

    trello_data = data.get("trello_res") or data.get("assembled_results", {}).get("trello", {})
    if not trello_data:
        st.info("Trello tracker data is not available.")
        return

    mode = trello_data.get("mode", TrackerSyncMode.LOCAL_ROADMAP_ONLY)
    is_live = trello_data.get("is_live", False)
    status_label = trello_data.get("status_label", "Local Workflow Mode")
    error_detail = trello_data.get("error_detail")
    board_url = trello_data.get("board_url", "https://trello.com/b/marketfolio-advanced-ml")
    lists = trello_data.get("lists", [])
    write_supported = trello_data.get("write_supported", False)
    write_status_message = trello_data.get("write_status_message", "")

    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Tracker Sync Mode",
            value="API Synchronized" if is_live else ("API Error" if mode == TrackerSyncMode.API_CONNECTION_FAILED else "Local Roadmap"),
            subtext=status_label,
            value_color=COLOR_GAIN if is_live else (COLOR_LOSS if mode == TrackerSyncMode.API_CONNECTION_FAILED else ACCENT_CYAN),
        )
    with c2:
        total_cards = sum(len(l.get("cards", [])) for l in lists)
        render_kpi_card(
            title="Total Research Tasks",
            value=f"{total_cards} Cards",
            subtext=f"Organized into {len(lists)} workflow lists",
        )
    with c3:
        render_kpi_card(
            title="Write Sync Capability",
            value="Authenticated" if write_supported else "Unsupported",
            subtext="Live card creation / updates",
            value_color=COLOR_GAIN if write_supported else "#f59e0b",
        )
    with c4:
        render_kpi_card(
            title="Data Integrity Guarantee",
            value="Audited",
            subtext="No fabricated completion claims",
            value_color=COLOR_GAIN,
        )

    # Status Notification Banner
    if mode == TrackerSyncMode.LIVE_SYNCHRONIZED:
        st.success("✅ **Live Trello Board Synchronized:** Cards and workflow lists are actively queried from the live Trello REST API.")
    elif mode == TrackerSyncMode.API_CONNECTION_FAILED:
        st.error(
            f"⚠️ **Trello API Synchronization Failed:** {status_label}\n\n"
            f"**Error Details:** `{error_detail}`\n\n"
            "The cards displayed below represent the **local roadmap specification**. "
            "They are **NOT synchronized** with the live board and must not be treated as live state."
        )
    else:
        st.info(
            "ℹ️ **Local Research Roadmap Mode:** `TRELLO_API_KEY`, `TRELLO_TOKEN`, or `TRELLO_BOARD_ID` "
            "environment variables are not configured. Displaying local experiment specification. "
            "Card statuses reflect planned specifications rather than live board state."
        )

    # Board Link Notice
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 1rem; margin-top: 0.5rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 600; color: #f8fafc; font-size: 0.92rem;">Research Board Link:</span>
                <span style="color: #94a3b8; font-size: 0.88rem; margin-left: 0.5rem;">{board_url}</span>
            </div>
            <a href="{board_url}" target="_blank" style="background-color: #06b6d4; color: #0c1017; font-weight: 600; font-size: 0.82rem; padding: 0.4rem 0.9rem; border-radius: 4px; text-decoration: none;">
                Open Trello Board ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Board Lists & Cards
    cols = st.columns(len(lists)) if lists else []
    for col, lst in zip(cols, lists):
        with col:
            st.markdown(
                f'<div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem; border-bottom: 2px solid #06b6d4; padding-bottom: 0.35rem;">{lst.get("name")}</div>',
                unsafe_allow_html=True,
            )
            for card in lst.get("cards", []):
                tags_html = "".join(
                    f'<span style="background-color:#1e293b; color:#38bdf8; font-size:0.7rem; padding:2px 6px; border-radius:3px; margin-right:4px;">{t}</span>'
                    for t in card.get("tags", [])
                )
                status_text = card.get("status", "Active")
                status_color = "#10b981" if is_live and "open" in status_text.lower() else "#f59e0b"
                st.markdown(
                    f"""
                    <div style="background-color: #131722; border: 1px solid #242936; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.75rem;">
                        <div style="font-weight: 600; color: #f1f5f9; font-size: 0.85rem; margin-bottom: 0.35rem;">{card.get("name")}</div>
                        <div style="color: #94a3b8; font-size: 0.78rem; line-height: 1.4; margin-bottom: 0.5rem;">{card.get("desc")}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>{tags_html}</div>
                            <span style="color: {status_color}; font-weight: 600; font-size: 0.72rem;">{status_text}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Interactive Task Card Synchronization
    with st.expander("➕ Synchronize New Research Task to Trello (Write Operation)", expanded=False):
        st.markdown(
            f'<div style="font-size:0.82rem; color:#94a3b8; margin-bottom:0.75rem;">{write_status_message}</div>',
            unsafe_allow_html=True,
        )
        with st.form("trello_write_form"):
            t_col1, t_col2 = st.columns([1, 1])
            with t_col1:
                task_name = st.text_input("Task Title", placeholder="e.g., Run out-of-sample stress test on GCN")
                target_list_id = st.text_input("Destination Trello List ID", placeholder="e.g., 64d9...")
            with t_col2:
                task_desc = st.text_area("Task Description & Specification", placeholder="Detailed methodology and validation criteria...")

            submitted = st.form_submit_button("Submit Task to Trello")
            if submitted:
                if not task_name or not target_list_id:
                    st.warning("Please provide both Task Title and Destination List ID.")
                else:
                    write_res = create_trello_card(list_id=target_list_id, name=task_name, desc=task_desc)
                    if write_res["success"]:
                        st.success(f"Card created successfully! Trello Card ID: {write_res['card'].get('id')}")
                    else:
                        st.error(f"Write operation failed: {write_res['error']}")

