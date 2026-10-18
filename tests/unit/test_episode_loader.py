"""Real-data EpisodeData loader: fixtures only (no network)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import split_episode_by_dates
from rl_trader.config.models import DataSettings, FeatureSettings, TrainingSettings
from rl_trader.data.episode_loader import (
    EpisodeDataError,
    load_episode_from_processed,
)


def _business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _write_ohlcv(
    processed_dir: Path,
    symbol: str,
    days: list[date],
    *,
    seed: int,
    start_tag: date,
    end_tag: date,
) -> Path:
    rng = np.random.default_rng(seed)
    n = len(days)
    rets = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days])
    frame = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 2_000_000, size=n),
            "adjustment_factor": 1.0,
            "open_adj": close * 0.99,
            "high_adj": close * 1.01,
            "low_adj": close * 0.98,
            "close_adj": close,
        },
        index=idx,
    )
    out_dir = processed_dir / "ohlcv"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{start_tag.isoformat()}_{end_tag.isoformat()}.parquet"
    frame.to_parquet(path)
    return path


def _compact_feature_settings(features_dir: Path) -> FeatureSettings:
    return FeatureSettings(
        lookback_bars=5,
        normalize=False,
        causal_only=True,
        feature_version="1",
        benchmark_symbol="SPY",
        output_dir=features_dir,
        manifest_dir=features_dir,
        quality_report_dir=features_dir / "quality",
        return_windows=(1, 5),
        ma_windows=(5,),
        rsi_period=5,
        macd_fast=3,
        macd_slow=6,
        macd_signal=3,
        bollinger_period=5,
        atr_period=5,
        volatility_window=5,
        downside_vol_window=5,
        volume_zscore_window=5,
        drawdown_window=10,
        rolling_high_window=10,
        regime_trend_window=5,
        regime_vol_window=5,
        vol_percentile_window=10,
        include_portfolio_state=False,
        include_fundamentals=False,
        include_regime=True,
        drop_na_rows=False,
    )


def _training_windows() -> TrainingSettings:
    return TrainingSettings(
        algorithm="ppo",
        model_id="unit-real",
        total_timesteps=64,
        seed=0,
        device="cpu",
        train_start=date(2020, 1, 2),
        train_end=date(2020, 3, 31),
        val_start=date(2020, 4, 1),
        val_end=date(2020, 4, 30),
        test_start=date(2020, 5, 1),
        test_end=date(2020, 5, 29),
        normalize_features=True,
        evaluate_test_after_training=True,
    )


def test_load_episode_from_labelled_ohlcv_fixtures(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    features_dir = tmp_path / "features"
    days = _business_days(date(2020, 1, 2), date(2020, 5, 29))
    assert len(days) >= 40
    for i, sym in enumerate(("SPY", "QQQ")):
        _write_ohlcv(
            processed,
            sym,
            days,
            seed=10 + i,
            start_tag=date(2020, 1, 2),
            end_tag=date(2020, 5, 29),
        )

    data = DataSettings(
        universe=("SPY", "QQQ"),
        processed_dir=processed,
        survivorship_bias_acknowledged=True,
    )
    features = _compact_feature_settings(features_dir)
    training = _training_windows()

    result = load_episode_from_processed(
        data, features, training, build_features_if_missing=True
    )
    episode = result.episode
    assert episode.symbols == ("SPY", "QQQ")
    assert episode.n_bars >= 40
    assert episode.prices.shape == (episode.n_bars, 2)
    assert episode.n_market > 0
    assert episode.n_regime > 0
    assert all(src == "built_from_ohlcv" for src in result.feature_sources.values())

    splits = split_episode_by_dates(episode, training)
    assert splits.train.n_bars >= 2
    assert splits.validation.n_bars >= 2
    assert splits.test.n_bars >= 2

    scaler = FeatureScaler().fit(splits.train, split="train")
    assert scaler.fitted_on == "train"
    mean_before = scaler.mean_.copy()
    _ = scaler.transform(splits.validation)
    assert np.allclose(scaler.mean_, mean_before)


def test_refuse_missing_ohlcv(tmp_path: Path) -> None:
    data = DataSettings(
        universe=("SPY", "QQQ"),
        processed_dir=tmp_path / "processed",
        survivorship_bias_acknowledged=True,
    )
    features = _compact_feature_settings(tmp_path / "features")
    training = _training_windows()
    with pytest.raises(EpisodeDataError, match="missing cleaned OHLCV"):
        load_episode_from_processed(data, features, training)


def test_refuse_incomplete_date_coverage(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    # Only January — does not cover val/test windows.
    days = _business_days(date(2020, 1, 2), date(2020, 1, 31))
    for i, sym in enumerate(("SPY", "QQQ")):
        _write_ohlcv(
            processed,
            sym,
            days,
            seed=20 + i,
            start_tag=date(2020, 1, 2),
            end_tag=date(2020, 1, 31),
        )
    data = DataSettings(
        universe=("SPY", "QQQ"),
        processed_dir=processed,
        survivorship_bias_acknowledged=True,
    )
    features = _compact_feature_settings(tmp_path / "features")
    training = _training_windows()
    with pytest.raises(EpisodeDataError, match="before test_end|stale|incomplete|≥2"):
        load_episode_from_processed(data, features, training)


def test_require_feature_parquet_refuses_when_missing(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    days = _business_days(date(2020, 1, 2), date(2020, 5, 29))
    for i, sym in enumerate(("SPY", "QQQ")):
        _write_ohlcv(
            processed,
            sym,
            days,
            seed=30 + i,
            start_tag=date(2020, 1, 2),
            end_tag=date(2020, 5, 29),
        )
    data = DataSettings(
        universe=("SPY", "QQQ"),
        processed_dir=processed,
        survivorship_bias_acknowledged=True,
    )
    features = _compact_feature_settings(tmp_path / "features")
    training = _training_windows()
    with pytest.raises(EpisodeDataError, match="missing features"):
        load_episode_from_processed(
            data, features, training, build_features_if_missing=False
        )
