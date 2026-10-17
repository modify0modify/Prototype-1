"""Local download cache — raw bytes/frames preserved with provenance metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from rl_trader.data.provider import DownloadMeta, OHLCVDownload


@dataclass(frozen=True, slots=True)
class CachePaths:
    """Paths for a cached download artifact pair."""

    parquet: Path
    meta: Path


def cache_key(provider: str, symbol: str, start: date, end: date) -> str:
    """Stable cache key for a provider/symbol/range request."""
    return f"{provider}__{symbol.upper()}__{start.isoformat()}_{end.isoformat()}"


def paths_for(cache_dir: Path, provider: str, symbol: str, start: date, end: date) -> CachePaths:
    """Return parquet + sidecar metadata paths under ``cache_dir``."""
    stem = cache_key(provider, symbol, start, end)
    return CachePaths(
        parquet=cache_dir / f"{stem}.parquet",
        meta=cache_dir / f"{stem}.meta.json",
    )


def load_cached(
    cache_dir: Path, provider: str, symbol: str, start: date, end: date
) -> OHLCVDownload | None:
    """Load a cached download if both parquet and metadata exist."""
    paths = paths_for(cache_dir, provider, symbol, start, end)
    if not paths.parquet.is_file() or not paths.meta.is_file():
        return None
    frame = pd.read_parquet(paths.parquet)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.index.name = "timestamp"
    meta = DownloadMeta.from_dict(json.loads(paths.meta.read_text(encoding="utf-8")))
    return OHLCVDownload(frame=frame, meta=meta)


def save_cached(cache_dir: Path, download: OHLCVDownload) -> CachePaths:
    """Persist frame + metadata; does not modify the in-memory frame."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = paths_for(
        cache_dir,
        download.meta.provider,
        download.meta.symbol,
        download.meta.requested_start,
        download.meta.requested_end,
    )
    download.frame.to_parquet(paths.parquet)
    paths.meta.write_text(
        json.dumps(download.meta.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
