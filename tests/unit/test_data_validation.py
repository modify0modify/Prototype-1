"""Unit tests for OHLCV validation detections (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rl_trader.data.cleaning import clean_ohlcv
from rl_trader.data.validation import validate_ohlcv_frame

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "market_data" / "spy_sample.csv"


def _load_base() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE, parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return frame.sort_index()


def test_clean_sorts_and_adds_adjusted_columns() -> None:
    raw = _load_base()
    # Unsorted + duplicate timestamp
    shuffled = pd.concat([raw.iloc[::-1], raw.iloc[[0]]])
    result = clean_ohlcv(shuffled)
    assert result.frame.index.is_monotonic_increasing
    assert result.duplicates_removed == 1
    assert "close_adj" in result.frame.columns
    assert "adjustment_factor" in result.frame.columns
    # With adj_close == close, factor should be 1
    assert float((result.frame["adjustment_factor"] - 1.0).abs().max()) < 1e-9


def test_clean_applies_adjustment_factor() -> None:
    raw = _load_base()
    raw = raw.copy()
    raw["adj_close"] = raw["close"] * 0.5
    cleaned = clean_ohlcv(raw).frame
    assert cleaned["close_adj"].iloc[0] == pytest.approx(raw["close"].iloc[0] * 0.5)
    assert cleaned["open_adj"].iloc[0] == pytest.approx(raw["open"].iloc[0] * 0.5)


def test_detect_duplicate_timestamps() -> None:
    frame = _load_base()
    frame = pd.concat([frame, frame.iloc[[1]]])
    report = validate_ohlcv_frame(frame, allow_missing_bars=True)
    assert report.metadata["duplicate_index_rows"] >= 2
    assert any("duplicated" in e for e in report.errors)


def test_detect_missing_trading_dates() -> None:
    frame = _load_base()
    # Drop 2023-01-06 (Friday) to create a gap inside the week
    frame = frame.drop(pd.Timestamp("2023-01-06", tz="UTC"))
    report = validate_ohlcv_frame(
        frame,
        start=date(2023, 1, 3),
        end=date(2023, 1, 13),
        allow_missing_bars=False,
    )
    assert report.metadata["missing_trading_dates_count"] >= 1
    assert any("missing" in e for e in report.errors)
    # Cleaning must not invent the missing day
    cleaned = clean_ohlcv(frame).frame
    assert pd.Timestamp("2023-01-06", tz="UTC") not in cleaned.index


def test_detect_zero_and_negative_prices() -> None:
    frame = _load_base().copy()
    frame.iloc[0, frame.columns.get_loc("close")] = 0.0
    frame.iloc[1, frame.columns.get_loc("open")] = -1.0
    report = validate_ohlcv_frame(frame, allow_missing_bars=True)
    assert report.metadata["zero_or_negative_or_nan_price_rows"] >= 1
    assert any("zero/negative" in e for e in report.errors)


def test_detect_negative_volume() -> None:
    frame = _load_base().copy()
    frame.iloc[2, frame.columns.get_loc("volume")] = -100
    report = validate_ohlcv_frame(frame, allow_missing_bars=True)
    assert report.metadata["negative_volume_rows"] == 1
    assert any("negative volume" in e for e in report.errors)


def test_detect_large_price_jump() -> None:
    frame = _load_base().copy()
    # 50% jump
    frame.iloc[4, frame.columns.get_loc("close")] = frame.iloc[3]["close"] * 1.5
    frame.iloc[4, frame.columns.get_loc("adj_close")] = frame.iloc[4]["close"]
    report = validate_ohlcv_frame(frame, jump_threshold=0.20, allow_missing_bars=True)
    assert report.metadata["large_jump_count"] >= 1
    assert any("jump_threshold" in w for w in report.warnings)


def test_validate_none_frame() -> None:
    report = validate_ohlcv_frame(None)
    assert report.ok is False
    assert "None" in report.errors[0]
