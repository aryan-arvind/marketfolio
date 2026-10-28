"""
MarketFolio Trello Research & Experiment Tracker.
Provides structured tracking for quantitative research tasks, model validation
experiments, and academic verification milestones.

Supports:
1. Live authenticated Trello API read synchronization (when API keys are provided).
2. Live authenticated Trello card creation and update write operations.
3. Explicit separation between Live Synchronized data, API connection failures, and offline local specifications.
4. Academic integrity compliance: Never fabricates completion statuses on static dictionaries.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import requests

from risklens.config import HTTP_TIMEOUT_SECONDS

logger = logging.getLogger("risklens")


class TrackerSyncMode:
    LIVE_SYNCHRONIZED = "LIVE_SYNCHRONIZED"
    API_CONNECTION_FAILED = "API_CONNECTION_FAILED"
    LOCAL_ROADMAP_ONLY = "LOCAL_ROADMAP_ONLY"


DEFAULT_WORKFLOW_LISTS = [
    {
        "name": "Research Roadmap & Specs",
        "cards": [
            {
                "id": "card-001",
                "name": "Quantile Threshold Calibration on Training Slices Only",
                "desc": "Ensure all volatility regime boundaries and VAE anomaly thresholds are fit strictly on pre-split training observations to prevent lookahead bias.",
                "status": "Planned / Unverified Specification",
                "tags": ["Academic Integrity", "Leakage Prevention"],
            },
            {
                "id": "card-002",
                "name": "Purge & Embargo Window Enforcement",
                "desc": "Enforce H-day embargo gap between chronological Train, Validation, and Test splits to eliminate forward-looking label overlap.",
                "status": "Planned / Unverified Specification",
                "tags": ["Validation", "Time Series"],
            },
        ],
    },
    {
        "name": "Model Architecture & Baselines",
        "cards": [
            {
                "id": "card-003",
                "name": "TimeSeriesVAE Anomaly Engine",
                "desc": "Multi-channel reconstruction error vs. Rolling Z-Score (3-Sigma) and Isolation Forest baselines.",
                "status": "Planned / Unverified Specification",
                "tags": ["Deep Learning", "VAE", "Anomaly"],
            },
            {
                "id": "card-004",
                "name": "Volatility Transformer with Attention Pooling",
                "desc": "Multi-head self-attention regime classification benchmarked against Majority Class, Martingale Persistence, and Logistic Regression.",
                "status": "Planned / Unverified Specification",
                "tags": ["Transformer", "Self-Attention", "Classification"],
            },
            {
                "id": "card-005",
                "name": "InterMarketGCN Relational Topology",
                "desc": "Kipf-Welling spectral convolution across 6 core macro nodes with normalized Laplacian adjacency.",
                "status": "Planned / Unverified Specification",
                "tags": ["GCN", "Relational Risk", "Systemic"],
            },
        ],
    },
    {
        "name": "Verification & Evidence",
        "cards": [
            {
                "id": "card-006",
                "name": "Unit & Integration Test Suite",
                "desc": "Regression tests for zero lookahead leakage, explicit train boundary verification, and metric fidelity.",
                "status": "Planned / Unverified Specification",
                "tags": ["Pytest", "CI/CD"],
            },
            {
                "id": "card-007",
                "name": "Dynamic Failure Mode Narrative Test",
                "desc": "Verify that narrative honestly reports underperformance when the learned neural model fails to beat martingale persistence.",
                "status": "Planned / Unverified Specification",
                "tags": ["Interpretability", "Finance"],
            },
        ],
    },
]


def fetch_trello_research_board() -> dict[str, Any]:
    """
    Attempts to fetch live Trello board data if TRELLO_API_KEY, TRELLO_TOKEN,
    and TRELLO_BOARD_ID are provided in environment variables.

    Returns a structured dictionary with explicit sync mode:
    - LIVE_SYNCHRONIZED: Successfully fetched live cards from Trello API.
    - API_CONNECTION_FAILED: API credentials provided but HTTP request failed or timed out.
    - LOCAL_ROADMAP_ONLY: API credentials missing; displaying local roadmap specification.
    """
    api_key = os.environ.get("TRELLO_API_KEY")
    token = os.environ.get("TRELLO_TOKEN")
    board_id = os.environ.get("TRELLO_BOARD_ID")
    custom_board_url = os.environ.get("TRELLO_BOARD_URL", "https://trello.com/b/marketfolio-advanced-ml")

    # Case 1: Credentials not supplied -> Local roadmap specification only
    if not api_key or not token or not board_id:
        return {
            "mode": TrackerSyncMode.LOCAL_ROADMAP_ONLY,
            "is_live": False,
            "status_label": "Local Research Roadmap (Offline Specification / API Keys Unset)",
            "error_detail": None,
            "board_id": None,
            "board_url": custom_board_url,
            "lists": DEFAULT_WORKFLOW_LISTS,
            "write_supported": False,
            "write_status_message": "Write operations unsupported: TRELLO_API_KEY and TRELLO_TOKEN environment variables not configured.",
        }

    # Case 2: Credentials supplied -> Attempt live synchronization
    try:
        url = f"https://api.trello.com/1/boards/{board_id}/lists"
        params = {
            "cards": "open",
            "card_fields": "name,desc,labels,due,closed",
            "key": api_key,
            "token": token,
        }
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            raw_lists = resp.json()
            parsed_lists = []
            for lst in raw_lists:
                cards = [
                    {
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "desc": c.get("desc", ""),
                        "status": "Live Open Card",
                        "tags": [lb.get("name") for lb in c.get("labels", []) if lb.get("name")],
                    }
                    for c in lst.get("cards", [])
                ]
                parsed_lists.append({"name": lst.get("name"), "cards": cards})

            logger.info("Successfully fetched live Trello board '%s' with %d lists.", board_id, len(parsed_lists))
            return {
                "mode": TrackerSyncMode.LIVE_SYNCHRONIZED,
                "is_live": True,
                "status_label": "Live Trello API Synchronized",
                "error_detail": None,
                "board_id": board_id,
                "board_url": custom_board_url,
                "lists": parsed_lists,
                "write_supported": True,
                "write_status_message": "Live write operations authenticated and supported.",
            }
        else:
            err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Trello API returned status %d. Refusing to disguise failure as live state.", resp.status_code)
            return {
                "mode": TrackerSyncMode.API_CONNECTION_FAILED,
                "is_live": False,
                "status_label": f"API Connection Failed (HTTP {resp.status_code}) - Displaying Unsynchronized Local Roadmap",
                "error_detail": err_msg,
                "board_id": board_id,
                "board_url": custom_board_url,
                "lists": DEFAULT_WORKFLOW_LISTS,
                "write_supported": False,
                "write_status_message": f"Write operations unavailable due to API connection failure: HTTP {resp.status_code}",
            }
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("Trello API connection error: %s. Refusing to disguise failure as live state.", exc)
        return {
            "mode": TrackerSyncMode.API_CONNECTION_FAILED,
            "is_live": False,
            "status_label": f"API Connection Error ({type(exc).__name__}) - Displaying Unsynchronized Local Roadmap",
            "error_detail": err_msg,
            "board_id": board_id,
            "board_url": custom_board_url,
            "lists": DEFAULT_WORKFLOW_LISTS,
            "write_supported": False,
            "write_status_message": f"Write operations unavailable due to connection error: {exc}",
        }


def create_trello_card(
    list_id: str,
    name: str,
    desc: str = "",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Creates a new card on a live Trello list via authenticated POST request.
    If credentials are not configured, returns an explicit error payload
    without throwing unhandled exceptions.
    """
    api_key = os.environ.get("TRELLO_API_KEY")
    token = os.environ.get("TRELLO_TOKEN")

    if not api_key or not token:
        return {
            "success": False,
            "card": None,
            "error": "Trello write synchronization unsupported in local mode: TRELLO_API_KEY and TRELLO_TOKEN environment variables must be configured.",
        }

    url = "https://api.trello.com/1/cards"
    params = {
        "idList": list_id,
        "name": name,
        "desc": desc,
        "key": api_key,
        "token": token,
    }
    try:
        resp = requests.post(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            card_data = resp.json()
            return {"success": True, "card": card_data, "error": None}
        return {
            "success": False,
            "card": None,
            "error": f"Trello API returned HTTP {resp.status_code}: {resp.text}",
        }
    except Exception as exc:
        return {"success": False, "card": None, "error": f"Trello API request error: {exc}"}


def update_trello_card(
    card_id: str,
    name: str | None = None,
    desc: str | None = None,
    closed: bool | None = None,
    id_list: str | None = None,
) -> dict[str, Any]:
    """
    Updates an existing Trello card (name, description, closed/archived, or destination list)
    via authenticated PUT request.
    """
    api_key = os.environ.get("TRELLO_API_KEY")
    token = os.environ.get("TRELLO_TOKEN")

    if not api_key or not token:
        return {
            "success": False,
            "card": None,
            "error": "Trello write synchronization unsupported in local mode: TRELLO_API_KEY and TRELLO_TOKEN environment variables must be configured.",
        }

    url = f"https://api.trello.com/1/cards/{card_id}"
    params: dict[str, Any] = {"key": api_key, "token": token}
    if name is not None:
        params["name"] = name
    if desc is not None:
        params["desc"] = desc
    if closed is not None:
        params["closed"] = "true" if closed else "false"
    if id_list is not None:
        params["idList"] = id_list

    try:
        resp = requests.put(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            card_data = resp.json()
            return {"success": True, "card": card_data, "error": None}
        return {
            "success": False,
            "card": None,
            "error": f"Trello API returned HTTP {resp.status_code}: {resp.text}",
        }
    except Exception as exc:
        return {"success": False, "card": None, "error": f"Trello API request error: {exc}"}

