"""Risk decision statuses and immutable decision records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np

from rl_trader.portfolio.targets import TargetPortfolio


class RiskDecision(StrEnum):
    """Contract outcomes: approved | resized | rejected."""

    APPROVED = "approved"
    RESIZED = "resized"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Full audit record for every risk decision."""

    decision_status: RiskDecision
    original_order: tuple[float, ...]
    approved_order: tuple[float, ...] | None
    violated_rules: tuple[str, ...]
    reason: str
    timestamp: datetime
    configuration_version: str
    model_identifier: str
    cancel_pending_orders: bool = False
    pending_order_ids_cancelled: tuple[str, ...] = ()
    kill_switch_active: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document (timestamps ISO-8601)."""
        payload = asdict(self)
        payload["decision_status"] = str(self.decision_status)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Engine return value — decision + optional resized portfolio + record."""

    decision: RiskDecision
    portfolio: TargetPortfolio | None
    reason: str
    record: DecisionRecord
    details: dict[str, Any] | None = None

    @property
    def status(self) -> RiskDecision:
        """Alias matching the public contract naming."""
        return self.decision


def weights_tuple(weights: np.ndarray | TargetPortfolio | None) -> tuple[float, ...] | None:
    """Stable float tuple for decision records."""
    if weights is None:
        return None
    if isinstance(weights, TargetPortfolio):
        arr = np.asarray(weights.weights, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    return tuple(float(x) for x in arr)
