"""Deterministic benchmark strategies (research baselines — not investment advice)."""

from __future__ import annotations

import numpy as np

from rl_trader.benchmarks.strategy import BaseStrategy
from rl_trader.config.models import BenchmarkSettings
from rl_trader.environment.actions import project_target_weights


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _normalize_long(raw: np.ndarray) -> np.ndarray:
    """Non-negative weights scaled so sum ≤ 1 (cash residual allowed)."""
    w = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return _zeros(w.size)
    if total > 1.0:
        w = w / total
    return w


def _drift_hold_weights(
    *,
    entry_weights: np.ndarray,
    entry_prices: np.ndarray,
    prices_t: np.ndarray,
) -> np.ndarray:
    """Map fixed share holdings (from entry weights) into current weights."""
    if np.any(entry_prices <= 0) or not np.isfinite(entry_prices).all():
        return _zeros(entry_weights.size)
    if not np.isfinite(prices_t).all() or np.any(prices_t <= 0):
        return _zeros(entry_weights.size)
    asset_value = entry_weights * (prices_t / entry_prices)
    cash0 = max(0.0, 1.0 - float(np.sum(entry_weights)))
    total = cash0 + float(np.sum(asset_value))
    if total <= 0.0 or not np.isfinite(total):
        return _zeros(entry_weights.size)
    return np.asarray(asset_value / total, dtype=np.float64)


class CashOnlyStrategy(BaseStrategy):
    """Hold 100% cash."""

    name = "cash_only"
    required_lookback = 1

    def target_weights(self, t: int) -> np.ndarray:
        _ = t
        prices = self._require_prepared()
        return _zeros(prices.shape[1])


class BuyAndHoldSpyStrategy(BaseStrategy):
    """Buy SPY once (respecting max_weight) and hold; weights drift with price."""

    name = "buy_and_hold_spy"
    required_lookback = 1
    enforce_max_weight = False

    def __init__(self, spy_symbol: str = "SPY", max_weight: float = 1.0) -> None:
        super().__init__()
        self.spy_symbol = spy_symbol.upper()
        self.max_weight = max_weight
        self._spy_index: int | None = None
        self._entry_t: int | None = None
        self._entry_weights: np.ndarray | None = None
        self._entry_prices: np.ndarray | None = None

    def prepare(self, prices: np.ndarray, symbols: tuple[str, ...]) -> None:
        super().prepare(prices, symbols)
        upper = [s.upper() for s in symbols]
        self._spy_index = upper.index(self.spy_symbol) if self.spy_symbol in upper else None
        self._entry_t = None
        self._entry_weights = None
        self._entry_prices = None

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if self._spy_index is None:
            return _zeros(n)
        if self._entry_t is None:
            raw = _zeros(n)
            raw[self._spy_index] = 1.0
            entry = project_target_weights(raw, max_weight=self.max_weight)
            self._entry_t = t
            self._entry_weights = entry
            self._entry_prices = prices[t].copy()
            return entry.copy()
        assert self._entry_weights is not None and self._entry_prices is not None
        return _drift_hold_weights(
            entry_weights=self._entry_weights,
            entry_prices=self._entry_prices,
            prices_t=prices[t],
        )


class EqualWeightBuyAndHoldStrategy(BaseStrategy):
    """Equal-dollar buy at first bar (max_weight capped); hold shares thereafter."""

    name = "equal_weight_buy_and_hold"
    required_lookback = 1
    enforce_max_weight = False

    def __init__(self, max_weight: float = 1.0) -> None:
        super().__init__()
        self.max_weight = max_weight
        self._entry_t: int | None = None
        self._entry_weights: np.ndarray | None = None
        self._entry_prices: np.ndarray | None = None

    def prepare(self, prices: np.ndarray, symbols: tuple[str, ...]) -> None:
        super().prepare(prices, symbols)
        self._entry_t = None
        self._entry_weights = None
        self._entry_prices = None

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if self._entry_t is None:
            raw = np.full(n, 1.0 / n, dtype=np.float64)
            entry = project_target_weights(raw, max_weight=self.max_weight)
            self._entry_t = t
            self._entry_weights = entry
            self._entry_prices = prices[t].copy()
            return entry.copy()
        assert self._entry_weights is not None and self._entry_prices is not None
        return _drift_hold_weights(
            entry_weights=self._entry_weights,
            entry_prices=self._entry_prices,
            prices_t=prices[t],
        )


class MovingAverageTrendStrategy(BaseStrategy):
    """Long assets with price above their trailing MA; equal among signals."""

    name = "moving_average_trend"

    def __init__(self, window: int = 50) -> None:
        super().__init__()
        if window < 2:
            raise ValueError("ma window must be >= 2")
        self.window = window
        self.required_lookback = window

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if t + 1 < self.window:
            return _zeros(n)
        window = prices[t - self.window + 1 : t + 1]
        ma = np.mean(window, axis=0)
        signal = prices[t] > ma
        if not np.any(signal):
            return _zeros(n)
        raw = signal.astype(np.float64)
        raw = raw / float(np.sum(raw))
        return _normalize_long(raw)


class CrossSectionalMomentumStrategy(BaseStrategy):
    """Long the top-k assets by trailing return (equal among selected)."""

    name = "cross_sectional_momentum"

    def __init__(self, lookback: int = 20, top_k: int = 3) -> None:
        super().__init__()
        if lookback < 2:
            raise ValueError("momentum lookback must be >= 2")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.lookback = lookback
        self.top_k = top_k
        self.required_lookback = lookback + 1

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if t < self.lookback:
            return _zeros(n)
        past = prices[t - self.lookback]
        now = prices[t]
        if np.any(past <= 0) or not np.isfinite(past).all() or not np.isfinite(now).all():
            return _zeros(n)
        mom = now / past - 1.0
        k = min(self.top_k, n)
        # Select top-k; ties broken by lower index via argsort stable behavior.
        order = np.argsort(-mom, kind="stable")
        chosen = order[:k]
        raw = _zeros(n)
        raw[chosen] = 1.0 / k
        return _normalize_long(raw)


class MeanReversionStrategy(BaseStrategy):
    """Long assets with short-term z-score below -threshold (equal among signals)."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 5, entry_z: float = 1.0) -> None:
        super().__init__()
        if lookback < 2:
            raise ValueError("mean-reversion lookback must be >= 2")
        self.lookback = lookback
        self.entry_z = float(entry_z)
        self.required_lookback = lookback

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if t + 1 < self.lookback:
            return _zeros(n)
        window = prices[t - self.lookback + 1 : t + 1]
        mu = np.mean(window, axis=0)
        sigma = np.std(window, axis=0, ddof=0)
        sigma = np.where(sigma < 1e-12, np.nan, sigma)
        z = (prices[t] - mu) / sigma
        signal = np.isfinite(z) & (z <= -self.entry_z)
        if not np.any(signal):
            return _zeros(n)
        raw = signal.astype(np.float64)
        raw = raw / float(np.sum(raw))
        return _normalize_long(raw)


class InverseVolatilityStrategy(BaseStrategy):
    """Risk-parity style: weights proportional to inverse trailing volatility."""

    name = "inverse_volatility"

    def __init__(self, vol_lookback: int = 20, vol_floor: float = 1e-6) -> None:
        super().__init__()
        if vol_lookback < 5:
            raise ValueError("vol_lookback must be >= 5")
        self.vol_lookback = vol_lookback
        self.vol_floor = float(vol_floor)
        self.required_lookback = vol_lookback + 1

    def target_weights(self, t: int) -> np.ndarray:
        prices = self._require_prepared()
        n = prices.shape[1]
        if t < self.vol_lookback:
            return _zeros(n)
        window = prices[t - self.vol_lookback : t + 1]
        rets = window[1:] / window[:-1] - 1.0
        if not np.isfinite(rets).all():
            return _zeros(n)
        vol = np.std(rets, axis=0, ddof=0)
        vol = np.maximum(vol, self.vol_floor)
        inv = 1.0 / vol
        raw = inv / float(np.sum(inv))
        return _normalize_long(raw)


def build_default_strategies(
    settings: BenchmarkSettings,
    *,
    max_weight: float = 1.0,
) -> list[BaseStrategy]:
    """Instantiate the standard baseline suite from config knobs."""
    return [
        CashOnlyStrategy(),
        BuyAndHoldSpyStrategy(spy_symbol=settings.spy_symbol, max_weight=max_weight),
        EqualWeightBuyAndHoldStrategy(max_weight=max_weight),
        MovingAverageTrendStrategy(window=settings.ma_window),
        CrossSectionalMomentumStrategy(
            lookback=settings.momentum_lookback, top_k=settings.momentum_top_k
        ),
        MeanReversionStrategy(
            lookback=settings.mean_reversion_lookback,
            entry_z=settings.mean_reversion_z,
        ),
        InverseVolatilityStrategy(
            vol_lookback=settings.vol_lookback,
            vol_floor=settings.risk_parity_vol_floor,
        ),
    ]
