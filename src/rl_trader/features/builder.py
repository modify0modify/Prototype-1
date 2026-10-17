"""Causal feature builder wrapper around the feature pipeline."""

from __future__ import annotations

import pandas as pd

from rl_trader.config.models import FeatureSettings
from rl_trader.features.pipeline import build_feature_matrix


class CausalFeatureBuilder:
    """Look-ahead-safe feature builder wrapping :func:`build_feature_matrix`."""

    def __init__(
        self,
        settings: FeatureSettings,
        *,
        symbol: str = "ASSET",
        benchmark: pd.DataFrame | None = None,
    ) -> None:
        self.settings = settings
        self.symbol = symbol
        self.benchmark = benchmark
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> CausalFeatureBuilder:
        """Record fit; no global statistics that would leak future info."""
        if frame.empty:
            raise ValueError("cannot fit on empty frame")
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform using only causal feature calculations."""
        if not self._fitted:
            raise RuntimeError("CausalFeatureBuilder.transform called before fit")
        features, _specs = build_feature_matrix(
            frame,
            self.settings,
            symbol=self.symbol,
            benchmark_ohlcv=self.benchmark,
        )
        return features


# Backward-compatible alias used by earlier skeleton code / tests
StubFeatureBuilder = CausalFeatureBuilder
