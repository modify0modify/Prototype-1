"""Leakage-detection tests: future prices must not affect historical features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_trader.config.models import FeatureSettings
from rl_trader.features import FundamentalEvent, run_feature_pipeline
from rl_trader.features.pipeline import build_feature_matrix
from rl_trader.features.technical import compute_technical_features


def _synthetic_ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    open_ = close + rng.normal(0, 0.2, size=n)
    volume = rng.integers(1_000_000, 2_000_000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_future_price_mutation_does_not_change_past_features() -> None:
    settings = FeatureSettings(
        include_portfolio_state=False,
        include_fundamentals=False,
        vol_percentile_window=40,
        drawdown_window=30,
        rolling_high_window=30,
        regime_trend_window=20,
    )
    base = _synthetic_ohlcv(100)
    bench = _synthetic_ohlcv(100, seed=1)

    features_full, _ = build_feature_matrix(
        base, settings, symbol="AAPL", benchmark_ohlcv=bench
    )

    cut = 60
    cut_ts = base.index[cut]
    mutated = base.copy()
    mutated.iloc[cut + 1 :, mutated.columns.get_loc("close")] *= 3.0
    mutated.iloc[cut + 1 :, mutated.columns.get_loc("adj_close")] = mutated["close"].iloc[
        cut + 1 :
    ]
    mutated.iloc[cut + 1 :, mutated.columns.get_loc("high")] *= 3.0

    features_mut, _ = build_feature_matrix(
        mutated, settings, symbol="AAPL", benchmark_ohlcv=bench
    )

    past_cols = list(features_full.columns)
    left = features_full.loc[:cut_ts, past_cols]
    right = features_mut.loc[:cut_ts, past_cols]
    assert left.isna().equals(right.isna())
    both = left.notna() & right.notna()
    assert np.allclose(
        left.to_numpy()[both.to_numpy()],
        right.to_numpy()[both.to_numpy()],
        rtol=1e-10,
        atol=1e-10,
    )


def test_prefix_frame_matches_full_frame_prefix() -> None:
    """Features on data[:t] equal features(full)[:t] for causal transforms."""
    settings = FeatureSettings(
        include_regime=False,
        include_portfolio_state=False,
        include_fundamentals=False,
    )
    full = _synthetic_ohlcv(80)
    cut = 50
    prefix = full.iloc[: cut + 1]

    feat_full = compute_technical_features(full, settings)
    feat_prefix = compute_technical_features(prefix, settings)

    shared = feat_prefix.index
    a = feat_full.loc[shared]
    b = feat_prefix
    assert a.isna().equals(b.isna())
    mask = a.notna() & b.notna()
    assert np.allclose(
        a.to_numpy()[mask.to_numpy()],
        b.to_numpy()[mask.to_numpy()],
        rtol=1e-10,
        atol=1e-10,
    )


def test_fundamentals_not_backfilled_before_publication() -> None:
    settings = FeatureSettings(
        include_regime=False,
        include_portfolio_state=False,
        include_fundamentals=True,
    )
    ohlcv = _synthetic_ohlcv(40)
    pub = ohlcv.index[20].date()
    events = [
        FundamentalEvent(
            publication_date=pub,
            revenue_growth=0.15,
            estimated_intrinsic_value=120.0,
        )
    ]
    features, _ = build_feature_matrix(
        ohlcv, settings, symbol="MSFT", fundamental_events=events
    )
    before = features.loc[features.index.date < pub, "revenue_growth"]
    assert before.isna().all()
    on_or_after = features.loc[features.index.date >= pub, "revenue_growth"]
    assert np.allclose(on_or_after.to_numpy(dtype=float), 0.15)


def test_pipeline_preserves_row_count(tmp_path: Path) -> None:
    settings = FeatureSettings(
        output_dir=tmp_path / "feat",
        manifest_dir=tmp_path / "feat",
        quality_report_dir=tmp_path / "qual",
        include_portfolio_state=True,
        include_fundamentals=True,
        include_regime=True,
        vol_percentile_window=30,
    )
    ohlcv = _synthetic_ohlcv(90)
    bench = _synthetic_ohlcv(90, seed=2)
    result = run_feature_pipeline(
        ohlcv, settings, symbol="NVDA", benchmark_ohlcv=bench
    )
    assert len(result.features) == len(ohlcv)
    assert result.manifest.row_policy.startswith("keep_all_rows")
    assert result.quality.n_rows == len(ohlcv)
    assert result.feature_path.is_file()
    assert result.manifest_path.is_file()
    assert result.quality_path.is_file()
    # Quality report includes missingness + distributions
    assert "return_1d" in result.quality.missing_by_column
    assert "return_1d" in result.quality.distribution_by_column


def test_drop_na_rows_config_rejected() -> None:
    with pytest.raises(Exception, match="drop_na_rows"):
        FeatureSettings(drop_na_rows=True)
