"""Streamlit monitoring dashboard (read-only, no live activation)."""

from rl_trader.monitoring.dashboard.loaders import DashboardPaths, build_overview, compute_health
from rl_trader.monitoring.dashboard.safety import (
    LIVE_ACTIVATION_BLOCKED_MESSAGE,
    SIMULATED_BANNER,
    mode_label,
    redact_mapping,
)

__all__ = [
    "LIVE_ACTIVATION_BLOCKED_MESSAGE",
    "SIMULATED_BANNER",
    "DashboardPaths",
    "build_overview",
    "compute_health",
    "mode_label",
    "redact_mapping",
]
