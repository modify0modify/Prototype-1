"""Fundamental data stubs (research path; separate from price features)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    """Point-in-time fundamental fields for a symbol."""

    symbol: str
    as_of: date
    fields: dict[str, Any]


def load_fundamentals(symbol: str, as_of: date) -> FundamentalSnapshot:
    """Load point-in-time fundamentals (stub)."""
    raise NotImplementedError(
        f"load_fundamentals({symbol!r}, {as_of.isoformat()}) is a stub; "
        "use only data available as-of the decision date (no look-ahead)."
    )
