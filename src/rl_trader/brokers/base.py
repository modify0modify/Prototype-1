"""Shared broker adapter interface.

Research logic must not import live adapters. Orchestration code selects
paper or live implementations behind this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.engine import RiskResult


@dataclass(frozen=True, slots=True)
class BrokerFill:
    """Normalized fill / acceptance record from a broker adapter."""

    portfolio: TargetPortfolio
    filled_at: datetime
    mode: str
    metadata: dict[str, Any]


class BrokerAdapter(Protocol):
    """Shared interface for paper and live broker adapters."""

    mode: str

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Submit a risk-approved target portfolio for execution/simulation."""
        ...
