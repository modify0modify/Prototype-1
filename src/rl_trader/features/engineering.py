"""Feature engineering stubs (causal transforms only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declarative feature definition stored with checkpoints."""

    name: str
    lookback: int
    version: str = "1"
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
