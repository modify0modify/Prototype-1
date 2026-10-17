"""Market data + feature snapshot providers for the paper-trading loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import numpy as np

from rl_trader.environment.synthetic import make_synthetic_episode


class StaleMarketDataError(RuntimeError):
    """Raised when market/feature data is stale or incomplete — safe shutdown."""


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Latest bars, quotes, and agent observation for one cycle."""

    symbols: tuple[str, ...]
    prices: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    price_timestamps: tuple[datetime, ...]
    observation: np.ndarray
    as_of: datetime
    feature_as_of: datetime
    prior_prices: np.ndarray | None = None


class MarketDataProvider(Protocol):
    """Obtain latest market data and updated features."""

    def latest(self) -> MarketSnapshot:
        """Return the newest usable snapshot (may be stale — caller validates)."""
        ...


class FixtureMarketDataProvider:
    """Deterministic synthetic feed for tests and ``use_fixture_market_data``."""

    def __init__(
        self,
        *,
        n_assets: int = 2,
        bar_age_seconds: float = 5.0,
        now: datetime | None = None,
        seed: int = 0,
        symbols: tuple[str, ...] | None = None,
    ) -> None:
        self.n_assets = n_assets
        self.bar_age_seconds = bar_age_seconds
        self._now = now
        self.seed = seed
        self._symbols = symbols
        self._episode = make_synthetic_episode(
            n_bars=40, n_assets=n_assets, seed=seed, price_path="drift"
        )

    def latest(self) -> MarketSnapshot:
        """Build a snapshot from the last synthetic bar."""
        now = self._now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        bar_ts = now - timedelta(seconds=self.bar_age_seconds)
        ep = self._episode
        t = ep.n_bars - 1
        prices = np.asarray(ep.prices[t], dtype=np.float64).reshape(-1)
        prior = np.asarray(ep.prices[t - 1], dtype=np.float64).reshape(-1)
        symbols = self._symbols or tuple(str(s) for s in ep.symbols)
        if len(symbols) != prices.shape[0]:
            symbols = tuple(f"A{i}" for i in range(prices.shape[0]))
        half = prices * 0.0005
        bids = prices - half
        asks = prices + half
        # Compact observation: market features + zeros for portfolio state.
        market = np.asarray(ep.market_features[t], dtype=np.float64).reshape(-1)
        fund = np.asarray(ep.fundamental_features[t], dtype=np.float64).reshape(-1)
        regime = np.asarray(ep.regime_features[t], dtype=np.float64).reshape(-1)
        obs = np.concatenate(
            [
                market,
                fund,
                regime,
                np.array([1.0], dtype=np.float64),  # cash frac placeholder
                np.zeros(prices.shape[0], dtype=np.float64),
            ]
        ).astype(np.float32)
        stamps = tuple(bar_ts for _ in symbols)
        return MarketSnapshot(
            symbols=symbols,
            prices=prices,
            bids=bids,
            asks=asks,
            price_timestamps=stamps,
            observation=obs,
            as_of=now,
            feature_as_of=bar_ts,
            prior_prices=prior,
        )


def assert_snapshot_fresh(
    snapshot: MarketSnapshot,
    *,
    max_age_seconds: float,
    now: datetime | None = None,
) -> None:
    """Raise :class:`StaleMarketDataError` when any bar is too old or incomplete."""
    clock = now or snapshot.as_of
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)
    if snapshot.prices.size == 0 or len(snapshot.symbols) == 0:
        raise StaleMarketDataError("market snapshot incomplete: empty symbols/prices")
    if not np.isfinite(snapshot.prices).all():
        raise StaleMarketDataError("market snapshot incomplete: non-finite prices")
    if snapshot.observation.size == 0 or not np.isfinite(snapshot.observation).all():
        raise StaleMarketDataError("feature snapshot incomplete or non-finite")
    ages = [
        (clock - ts.astimezone(UTC)).total_seconds()
        if ts.tzinfo
        else (clock - ts.replace(tzinfo=UTC)).total_seconds()
        for ts in snapshot.price_timestamps
    ]
    max_age = max(ages) if ages else float("inf")
    feature_age = (
        (clock - snapshot.feature_as_of.astimezone(UTC)).total_seconds()
        if snapshot.feature_as_of.tzinfo
        else (clock - snapshot.feature_as_of.replace(tzinfo=UTC)).total_seconds()
    )
    if max_age > max_age_seconds or feature_age > max_age_seconds:
        raise StaleMarketDataError(
            f"stale market data: bar_age={max_age:.1f}s feature_age={feature_age:.1f}s "
            f"max_allowed={max_age_seconds:.1f}s — shutting down safely"
        )
