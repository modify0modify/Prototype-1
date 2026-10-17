"""Unit / integration-style tests for deterministic benchmarks (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_trader.benchmarks import (
    CashOnlyStrategy,
    PricePanel,
    build_default_strategies,
    compute_metrics,
    make_synthetic_price_panel,
    resolve_env_settings_for_benchmarks,
    run_all_benchmarks,
    run_strategy,
    write_benchmark_report,
)
from rl_trader.benchmarks.strategies import (
    BuyAndHoldSpyStrategy,
    CrossSectionalMomentumStrategy,
    EqualWeightBuyAndHoldStrategy,
    InverseVolatilityStrategy,
    MeanReversionStrategy,
    MovingAverageTrendStrategy,
)
from rl_trader.config import get_settings
from rl_trader.config.models import BenchmarkSettings, EnvironmentSettings

REQUIRED = {
    "cash_only",
    "buy_and_hold_spy",
    "equal_weight_buy_and_hold",
    "moving_average_trend",
    "cross_sectional_momentum",
    "mean_reversion",
    "inverse_volatility",
}


def _env(**overrides: object) -> EnvironmentSettings:
    base: dict[str, object] = {
        "initial_cash": 100_000.0,
        "max_weight": 0.5,
        "commission_bps": 1.0,
        "bid_ask_spread_bps": 2.0,
        "slippage_bps": 1.0,
        "market_impact_coeff": 0.05,
        "execution_delay_bars": 0,
        "allow_same_bar_execution": True,
    }
    base.update(overrides)
    return EnvironmentSettings(**base)  # type: ignore[arg-type]


def test_build_default_strategies_names() -> None:
    names = {s.name for s in build_default_strategies(BenchmarkSettings())}
    assert names == REQUIRED


def test_cash_only_has_near_zero_costs_and_flat_equity() -> None:
    panel = make_synthetic_price_panel(n_bars=40, seed=1)
    result = run_strategy(CashOnlyStrategy(), panel, _env(commission_bps=10.0))
    assert result.metrics.number_of_trades == 0
    assert result.metrics.transaction_costs == pytest.approx(0.0)
    assert result.metrics.cumulative_return == pytest.approx(0.0)
    assert result.metrics.exposure == pytest.approx(0.0)


def test_buy_and_hold_spy_targets_spy_column() -> None:
    panel = make_synthetic_price_panel(n_bars=30, symbols=("QQQ", "SPY", "AAPL"), seed=2)
    strat = BuyAndHoldSpyStrategy(spy_symbol="SPY")
    strat.prepare(panel.prices, panel.symbols)
    w = strat.target_weights(5)
    assert w[1] == pytest.approx(1.0)
    assert float(np.sum(w)) == pytest.approx(1.0)


def test_equal_weight_buy_and_hold_drifts_with_prices() -> None:
    prices = np.array(
        [
            [100.0, 100.0],
            [110.0, 100.0],
            [121.0, 100.0],
        ],
        dtype=np.float64,
    )
    panel = PricePanel(
        timestamps=np.array(["d0", "d1", "d2"], dtype=object),
        prices=prices,
        symbols=("A", "B"),
    )
    strat = EqualWeightBuyAndHoldStrategy(max_weight=1.0)
    strat.prepare(panel.prices, panel.symbols)
    w0 = strat.target_weights(0)
    assert np.allclose(w0, [0.5, 0.5])
    w2 = strat.target_weights(2)
    # A appreciated; B flat → A weight > B
    assert w2[0] > w2[1]
    assert float(np.sum(w2)) == pytest.approx(1.0)


def test_buy_and_hold_spy_low_turnover_after_entry() -> None:
    panel = make_synthetic_price_panel(n_bars=40, seed=11)
    result = run_strategy(
        BuyAndHoldSpyStrategy(spy_symbol="SPY", max_weight=0.25),
        panel,
        _env(
            max_weight=0.25,
            commission_bps=0.0,
            bid_ask_spread_bps=0.0,
            slippage_bps=0.0,
            market_impact_coeff=0.0,
        ),
    )
    # Zero-cost B&H: a single initial buy (one asset leg), no daily rebalancing.
    assert result.metrics.number_of_trades == 1


def test_ma_trend_requires_lookback_then_signals() -> None:
    # Constant then jump above MA
    n = 20
    prices = np.full((n, 1), 100.0)
    prices[-1, 0] = 150.0
    panel = PricePanel(
        timestamps=np.arange(n),
        prices=prices,
        symbols=("A",),
    )
    strat = MovingAverageTrendStrategy(window=5)
    strat.prepare(panel.prices, panel.symbols)
    assert np.allclose(strat.target_weights(3), [0.0])
    assert strat.target_weights(n - 1)[0] == pytest.approx(1.0)


def test_momentum_selects_top_k() -> None:
    prices = np.array(
        [
            [100.0, 100.0, 100.0],
            [101.0, 110.0, 105.0],
            [102.0, 121.0, 110.0],
        ]
    )
    strat = CrossSectionalMomentumStrategy(lookback=2, top_k=1)
    strat.prepare(prices, ("A", "B", "C"))
    w = strat.target_weights(2)
    assert w[1] == pytest.approx(1.0)  # B has highest 2-bar return


def test_mean_reversion_buys_depressed_name() -> None:
    # Asset 0 dips at end; asset 1 flat
    prices = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
            [70.0, 100.0],
        ]
    )
    strat = MeanReversionStrategy(lookback=5, entry_z=1.0)
    strat.prepare(prices, ("A", "B"))
    w = strat.target_weights(4)
    assert w[0] > 0.0
    assert w[1] == pytest.approx(0.0)


def test_inverse_volatility_prefers_lower_vol() -> None:
    rng = np.random.default_rng(0)
    n = 40
    low = 100 * np.cumprod(1 + rng.normal(0, 0.002, size=n))
    high = 100 * np.cumprod(1 + rng.normal(0, 0.03, size=n))
    prices = np.column_stack([low, high])
    strat = InverseVolatilityStrategy(vol_lookback=20)
    strat.prepare(prices, ("LOW", "HIGH"))
    w = strat.target_weights(n - 1)
    assert w[0] > w[1]


def test_shared_harness_all_strategies_produce_metrics() -> None:
    panel = make_synthetic_price_panel(n_bars=100, seed=3)
    env = _env()
    bench = BenchmarkSettings(
        ma_window=10,
        momentum_lookback=10,
        momentum_top_k=2,
        mean_reversion_lookback=5,
        vol_lookback=10,
    )
    results = run_all_benchmarks(panel, env_settings=env, benchmark_settings=bench)
    assert {r.metrics.strategy for r in results} == REQUIRED
    # Shared assumptions identical across strategies
    assumption_keys = {
        "initial_cash",
        "max_weight",
        "commission_bps",
        "bid_ask_spread_bps",
        "slippage_bps",
        "market_impact_coeff",
        "cost_model",
    }
    first = results[0].assumptions
    for r in results:
        for key in assumption_keys:
            assert r.assumptions[key] == first[key]
        m = r.metrics
        for field in (
            "cumulative_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "turnover",
            "transaction_costs",
            "number_of_trades",
            "win_rate",
            "profit_factor",
            "exposure",
            "worst_day",
            "best_day",
        ):
            assert hasattr(m, field)
        assert m.n_days > 0
        assert m.ending_capital > 0


def test_costs_match_env_settings_not_backtest_only() -> None:
    panel = make_synthetic_price_panel(n_bars=60, seed=4)
    env = _env(commission_bps=25.0, slippage_bps=25.0)
    bench = BenchmarkSettings(
        use_environment_costs=True,
        ma_window=5,
        momentum_lookback=5,
        vol_lookback=10,
    )
    resolved = resolve_env_settings_for_benchmarks(
        environment=env, benchmarks=bench, initial_cash=50_000.0
    )
    assert resolved.commission_bps == 25.0
    assert resolved.initial_cash == 50_000.0
    results = run_all_benchmarks(panel, env_settings=resolved, benchmark_settings=bench)
    spy = next(r for r in results if r.metrics.strategy == "buy_and_hold_spy")
    assert spy.metrics.transaction_costs > 0.0


def test_write_benchmark_report(tmp_path: Path) -> None:
    panel = make_synthetic_price_panel(n_bars=50, seed=5)
    bench = BenchmarkSettings(ma_window=5, momentum_lookback=5, vol_lookback=10)
    results = run_all_benchmarks(panel, env_settings=_env(), benchmark_settings=bench)
    json_path, md_path = write_benchmark_report(results, tmp_path, run_id="test")
    assert json_path.exists()
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "Not investment advice" in text
    assert "No RL training" in text
    assert "cash_only" in text


def test_compute_metrics_basic() -> None:
    equity = np.array([100.0, 110.0, 105.0, 120.0])
    rets = equity[1:] / equity[:-1] - 1.0
    m = compute_metrics(
        strategy="toy",
        equity=equity,
        daily_returns=rets,
        turnovers=np.array([0.1, 0.0, 0.2]),
        costs=np.array([1.0, 0.0, 2.0]),
        n_trades=3,
        exposures=np.array([1.0, 1.0, 0.5]),
        starting_capital=100.0,
    )
    assert m.cumulative_return == pytest.approx(0.2)
    assert m.transaction_costs == pytest.approx(3.0)
    assert m.number_of_trades == 3
    assert m.best_day >= m.worst_day


def test_research_yaml_loads_benchmarks(monkeypatch: object) -> None:
    for key in ("APP_MODE", "LIVE_TRADING_ENABLED", "BROKER_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "research.yaml")
    assert settings.benchmarks.ma_window == 50
    assert settings.benchmarks.use_environment_costs is True
