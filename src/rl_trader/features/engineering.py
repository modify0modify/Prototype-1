"""Feature engineering interfaces and shared FeatureSpec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declarative feature definition stored with checkpoints / manifests."""

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
