"""Observation / action space builders for ``PortfolioEnv``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from rl_trader.environment.episode import EpisodeData


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """Byte offsets into the flat observation vector."""

    n_assets: int
    n_market: int
    n_fundamental: int
    n_regime: int
    recent_returns_lookback: int

    @property
    def size(self) -> int:
        """Total observation dimension."""
        # market + fund + regime + cash + positions + recent returns + drawdown
        return (
            self.n_market
            + self.n_fundamental
            + self.n_regime
            + 1
            + self.n_assets
            + self.recent_returns_lookback
            + 1
        )

    @property
    def slice_market(self) -> slice:
        start = 0
        return slice(start, start + self.n_market)

    @property
    def slice_fundamental(self) -> slice:
        start = self.n_market
        return slice(start, start + self.n_fundamental)

    @property
    def slice_regime(self) -> slice:
        start = self.n_market + self.n_fundamental
        return slice(start, start + self.n_regime)

    @property
    def index_cash(self) -> int:
        return self.n_market + self.n_fundamental + self.n_regime

    @property
    def slice_positions(self) -> slice:
        start = self.index_cash + 1
        return slice(start, start + self.n_assets)

    @property
    def slice_recent_returns(self) -> slice:
        start = self.index_cash + 1 + self.n_assets
        return slice(start, start + self.recent_returns_lookback)

    @property
    def index_drawdown(self) -> int:
        return self.size - 1


def build_observation_layout(
    episode: EpisodeData, *, recent_returns_lookback: int
) -> ObservationLayout:
    """Derive layout from episode feature widths."""
    if recent_returns_lookback < 1:
        raise ValueError("recent_returns_lookback must be >= 1")
    return ObservationLayout(
        n_assets=episode.n_assets,
        n_market=episode.n_market,
        n_fundamental=episode.n_fundamental,
        n_regime=episode.n_regime,
        recent_returns_lookback=recent_returns_lookback,
    )


def make_action_space(n_assets: int, max_weight: float) -> spaces.Box:
    """Continuous target weights in ``[0, max_weight]`` per asset."""
    return spaces.Box(
        low=0.0,
        high=float(max_weight),
        shape=(n_assets,),
        dtype=np.float32,
    )


def make_observation_space(layout: ObservationLayout) -> spaces.Box:
    """Unbounded float observation box (finite values required at runtime)."""
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(layout.size,),
        dtype=np.float32,
    )


def pack_observation(
    layout: ObservationLayout,
    *,
    market: np.ndarray,
    fundamental: np.ndarray,
    regime: np.ndarray,
    cash_frac: float,
    positions: np.ndarray,
    recent_returns: np.ndarray,
    drawdown: float,
) -> np.ndarray:
    """Assemble a flat float32 observation vector."""
    obs = np.zeros(layout.size, dtype=np.float32)
    obs[layout.slice_market] = np.asarray(market, dtype=np.float32)
    obs[layout.slice_fundamental] = np.asarray(fundamental, dtype=np.float32)
    obs[layout.slice_regime] = np.asarray(regime, dtype=np.float32)
    obs[layout.index_cash] = np.float32(cash_frac)
    obs[layout.slice_positions] = np.asarray(positions, dtype=np.float32)
    rr = np.asarray(recent_returns, dtype=np.float32).reshape(-1)
    if rr.shape[0] != layout.recent_returns_lookback:
        raise ValueError(
            f"recent_returns length {rr.shape[0]} != {layout.recent_returns_lookback}"
        )
    obs[layout.slice_recent_returns] = rr
    obs[layout.index_drawdown] = np.float32(drawdown)
    return obs
