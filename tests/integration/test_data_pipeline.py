"""Integration tests for the data pipeline with a fixture provider (no network)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from rl_trader.config.models import DataSettings
from rl_trader.data import FixtureProvider, run_data_pipeline
from rl_trader.data.schema import ADJUSTMENT_METHOD

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "market_data" / "spy_sample.csv"


def _fixture_frame() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE, parse_dates=["timestamp"]).set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index, tz="UTC")
    return frame


def test_pipeline_keeps_raw_and_cleaned_separate(tmp_path: Path) -> None:
    settings = DataSettings(
        universe=("SPY",),
        provider="yfinance",  # factory unused — we inject FixtureProvider
        start_date=date(2023, 1, 3),
        end_date=date(2023, 1, 13),
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        cache_dir=tmp_path / "cache",
        quality_report_dir=tmp_path / "reports",
        allow_missing_bars=False,
        jump_threshold=0.5,
    )
    provider = FixtureProvider({"SPY": _fixture_frame()})
    result = run_data_pipeline(
        settings,
        provider=provider,
        use_cache=True,
    )
    assert len(result.results) == 1
    item = result.results[0]

    assert item.raw_path.is_file()
    assert item.cleaned_path.is_file()
    assert item.cache_parquet.is_file()
    assert item.quality_report_path.is_file()

    # Paths must stay in different trees
    assert "raw" in item.raw_path.parts
    assert "processed" in item.cleaned_path.parts
    assert item.raw_path != item.cleaned_path

    raw = pd.read_parquet(item.raw_path)
    cleaned = pd.read_parquet(item.cleaned_path)
    assert "adjustment_factor" not in raw.columns
    assert "close_adj" in cleaned.columns
    assert "open_adj" in cleaned.columns

    # Raw must not be modified by cleaning (row count may drop only via duplicate drop
    # in cleaned; raw archive is the provider snapshot)
    assert len(raw) == len(_fixture_frame())

    quality = item.quality.to_dict()
    assert quality["provider"] == "fixture"
    assert quality["timezone"] == "UTC"
    assert quality["adjustment_method"] == ADJUSTMENT_METHOD
    assert "requested_start" in quality
    assert "downloaded_at" in quality
    assert "validation" in quality
    assert quality["raw_path"] == str(item.raw_path)
    assert quality["cleaned_path"] == str(item.cleaned_path)


def test_pipeline_cache_hit_does_not_require_network(tmp_path: Path) -> None:
    settings = DataSettings(
        universe=("SPY",),
        start_date=date(2023, 1, 3),
        end_date=date(2023, 1, 13),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        cache_dir=tmp_path / "cache",
        quality_report_dir=tmp_path / "reports",
        allow_missing_bars=False,
        jump_threshold=0.5,
    )
    provider = FixtureProvider({"SPY": _fixture_frame()})
    first = run_data_pipeline(settings, provider=provider, use_cache=True)
    # Second run with a provider that would fail if called
    class BoomProvider:
        name = "fixture"

        def fetch_daily_ohlcv(self, symbol: str, start: date, end: date):  # type: ignore[no-untyped-def]
            raise AssertionError("provider should not be called on cache hit")

    second = run_data_pipeline(settings, provider=BoomProvider(), use_cache=True)
    assert first.results[0].cache_parquet == second.results[0].cache_parquet
    assert second.results[0].cleaned_path.is_file()


def test_pipeline_records_metadata_sidecar(tmp_path: Path) -> None:
    settings = DataSettings(
        universe=("SPY",),
        start_date=date(2023, 1, 3),
        end_date=date(2023, 1, 13),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        cache_dir=tmp_path / "cache",
        quality_report_dir=tmp_path / "reports",
        allow_missing_bars=False,
    )
    provider = FixtureProvider({"SPY": _fixture_frame()})
    item = run_data_pipeline(settings, provider=provider).results[0]
    sidecar = item.raw_path.parent / item.raw_path.name.replace(".parquet", ".meta.json")
    assert sidecar.is_file()
    text = sidecar.read_text(encoding="utf-8")
    assert "fixture" in text
    assert "requested_start" in text
    assert "downloaded_at" in text
