"""
Pytest configuration and shared fixtures for RiskLens.
Configures network test skipping by default.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: mark test as requiring live public internet access"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("-m"):
        skip_network = pytest.mark.skip(reason="Network test skipped by default. Run with -m network or via smoke_test.py.")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
