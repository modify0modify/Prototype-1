"""Download request helpers and a fixture/stub provider for offline use."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from rl_trader.data.provider import DownloadMeta, OHLCVDownload
from rl_trader.data.schema import RAW_OHLCV_COLUMNS


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Parameters for a historical download job."""

    symbols: tuple[str, ...]
    start: date
    end: date
    destination: Path


class StubDownloader:
    """Placeholder downloader — does not contact any network service."""

    def download(self, request: DownloadRequest) -> Path:
        """Raise until a real provider adapter is used via the pipeline."""
        raise NotImplementedError(
            "StubDownloader: use rl_trader.data.pipeline.run_data_pipeline "
            "with YFinanceProvider (or a test fixture provider)."
        )


class FixtureProvider:
    """``MarketDataProvider`` that loads OHLCV from a local CSV/parquet fixture.

    Used by unit/integration tests — no network access.
    """

    name = "fixture"

    def __init__(self, frame_by_symbol: dict[str, pd.DataFrame]) -> None:
        self._frames = {k.upper(): v.copy() for k, v in frame_by_symbol.items()}

    def fetch_daily_ohlcv(self, symbol: str, start: date, end: date) -> OHLCVDownload:
        """Return a sliced fixture frame with download metadata."""
        key = symbol.upper()
        if key not in self._frames:
            raise KeyError(f"no fixture data for symbol {key!r}")
        frame = self._frames[key].copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        elif frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        frame.index = pd.DatetimeIndex(
            [ts.normalize() for ts in frame.index], tz="UTC", name="timestamp"
        )
        for col in RAW_OHLCV_COLUMNS:
            if col not in frame.columns and col == "adj_close":
                frame["adj_close"] = frame["close"]
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        sliced = frame.loc[mask, list(RAW_OHLCV_COLUMNS)].sort_index()
        meta = DownloadMeta(
            provider=self.name,
            symbol=key,
            requested_start=start,
            requested_end=end,
            downloaded_at=datetime.now(UTC),
            timezone="UTC",
            rows=int(len(sliced)),
            source_detail="FixtureProvider",
        )
        return OHLCVDownload(frame=sliced, meta=meta)
