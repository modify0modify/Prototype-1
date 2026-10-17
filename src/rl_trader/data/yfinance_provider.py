"""Yahoo Finance historical provider via the ``yfinance`` library."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from rl_trader.data.provider import DownloadMeta, OHLCVDownload
from rl_trader.data.schema import RAW_OHLCV_COLUMNS


class YFinanceProvider:
    """``MarketDataProvider`` backed by yfinance (easily replaceable)."""

    name = "yfinance"

    def fetch_daily_ohlcv(self, symbol: str, start: date, end: date) -> OHLCVDownload:
        """Download daily OHLCV from Yahoo Finance.

        yfinance's ``end`` parameter is exclusive; we add one day so the
        requested end date is included when the market was open.
        """
        import yfinance as yf

        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")

        ticker = yf.Ticker(symbol.upper())
        # end is exclusive in yfinance history()
        hist = ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=False,
        )
        if hist.empty:
            frame = pd.DataFrame(columns=list(RAW_OHLCV_COLUMNS))
            frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        else:
            frame = _normalize_yfinance_frame(hist)

        meta = DownloadMeta(
            provider=self.name,
            symbol=symbol.upper(),
            requested_start=start,
            requested_end=end,
            downloaded_at=datetime.now(UTC),
            timezone="UTC",
            rows=int(len(frame)),
            source_detail="yfinance.Ticker.history(auto_adjust=False)",
        )
        return OHLCVDownload(frame=frame, meta=meta)


def _normalize_yfinance_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """Map yfinance columns to the canonical raw schema in UTC."""
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    # yfinance may already lower-case depending on version
    cols = {
        c: rename.get(c, rename.get(c.title(), c.lower().replace(" ", "_")))
        for c in hist.columns
    }
    frame = hist.rename(columns=cols).copy()
    for required in ("open", "high", "low", "close", "volume"):
        if required not in frame.columns:
            raise ValueError(f"yfinance response missing column {required!r}")
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]

    frame = frame.loc[:, list(RAW_OHLCV_COLUMNS)]
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        # Session dates from Yahoo are US equity calendar dates; store as UTC midnight.
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    # Normalize to midnight UTC on the calendar date
    index = pd.DatetimeIndex([ts.normalize() for ts in index], tz="UTC", name="timestamp")
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.sort_index()
    return frame
