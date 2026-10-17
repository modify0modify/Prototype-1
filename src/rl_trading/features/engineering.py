"""Feature engineering stubs.

Look-ahead bias note
--------------------
Every transform must be causal: features at time ``t`` may use only data with
timestamps ``<= t`` (or ``< t`` for strict bar-close decisions). Rolling
windows, normalization, and train-set statistics must be fit on past data only
and applied forward — never fit on the full sample then applied in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declarative feature pipeline specification (stub)."""

    name: str
    lookback: int
    causal_only: bool = True


class FeatureBuilder(Protocol):
    """Builds model-ready feature matrices from validated market data."""

    def fit(self, frame: Any) -> FeatureBuilder:
        """Fit transform parameters on a training window only."""
        ...

    def transform(self, frame: Any) -> Any:
        """Apply a fitted transform without peeking ahead."""
        ...


class StubFeatureBuilder:
    """Placeholder feature builder."""

    def fit(self, frame: Any) -> StubFeatureBuilder:
        """Record that fit was called; no real statistics computed."""
        _ = frame
        return self

    def transform(self, frame: Any) -> Any:
        """Raise until real feature transforms are implemented."""
        raise NotImplementedError(
            "StubFeatureBuilder.transform: implement causal feature transforms. "
            "Do not silently fill missing inputs."
        )
