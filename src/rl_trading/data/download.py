"""Historical market-data download stubs.

Look-ahead bias note
--------------------
Download windows must end at or before the intended decision timestamp.
Never use a bar that would not have been available at decision time.

Missing-data policy
-------------------
Do **not** silently fill gaps. Missing bars must surface as validation errors
or explicit, logged imputation decisions — never hidden forward-fills in this
pipeline without a recorded audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Parameters for a historical download job."""

    symbols: tuple[str, ...]
    start: date
    end: date
    destination: Path


class MarketDataDownloader(Protocol):
    """Interface for provider-backed downloaders (stub)."""

    def download(self, request: DownloadRequest) -> Path:
        """Fetch bars into ``request.destination`` and return the path written."""
        ...


class StubDownloader:
    """Placeholder downloader — does not contact any network service."""

    def download(self, request: DownloadRequest) -> Path:
        """Raise until a real provider adapter is implemented."""
        raise NotImplementedError(
            "StubDownloader: wire a data provider adapter before downloading. "
            "API keys must come from RL_TRADING_DATA_API_* env vars."
        )
