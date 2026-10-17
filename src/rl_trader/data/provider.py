"""Abstract market-data provider interface (replaceable without downstream changes)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class DownloadMeta:
    """Provenance recorded with every download / cache artifact."""

    provider: str
    symbol: str
    requested_start: date
    requested_end: date
    downloaded_at: datetime
    timezone: str = "UTC"
    rows: int = 0
    source_detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable metadata."""
        return {
            "provider": self.provider,
            "symbol": self.symbol,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "downloaded_at": self.downloaded_at.astimezone(UTC).isoformat(),
            "timezone": self.timezone,
            "rows": self.rows,
            "source_detail": self.source_detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> DownloadMeta:
        """Restore metadata from JSON."""
        rows_raw = data.get("rows", 0)
        if not isinstance(rows_raw, int):
            rows_raw = int(str(rows_raw))
        return cls(
            provider=str(data["provider"]),
            symbol=str(data["symbol"]),
            requested_start=date.fromisoformat(str(data["requested_start"])),
            requested_end=date.fromisoformat(str(data["requested_end"])),
            downloaded_at=datetime.fromisoformat(str(data["downloaded_at"])),
            timezone=str(data.get("timezone", "UTC")),
            rows=rows_raw,
            source_detail=str(data.get("source_detail", "")),
        )


@dataclass(frozen=True, slots=True)
class OHLCVDownload:
    """Raw OHLCV frame plus download provenance."""

    frame: pd.DataFrame
    meta: DownloadMeta


class MarketDataProvider(Protocol):
    """Historical daily OHLCV provider — implementations must be swappable."""

    name: str

    def fetch_daily_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> OHLCVDownload:
        """Fetch daily bars for ``symbol`` in ``[start, end]`` (inclusive intent).

        Returns a DataFrame indexed by timezone-aware UTC timestamps with
        columns at least ``open, high, low, close, volume`` and ``adj_close``
        when the upstream source provides it. Must not mutate caller state.
        """
        ...
