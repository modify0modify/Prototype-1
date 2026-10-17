"""Per-step environment audit records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StepLogRecord:
    """One environment step — suitable for research audit trails."""

    timestamp: str
    observation_id: int
    action: tuple[float, ...]
    pre_trade_positions: tuple[float, ...]
    requested_positions: tuple[float, ...]
    approved_simulated_positions: tuple[float, ...]
    trades: tuple[float, ...]
    costs: float
    portfolio_value: float
    return_: float
    drawdown: float
    reward: float
    termination_reason: str
    rejected_trade: bool = False
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (``return_`` keyed as ``return``)."""
        payload = asdict(self)
        payload["return"] = payload.pop("return_")
        return payload


def as_float_tuple(values: np.ndarray | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Convert an array-like to a plain float tuple."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return tuple(float(x) for x in arr)
