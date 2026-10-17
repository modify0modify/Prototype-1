"""Historical market-data download stubs (research path; no broker contact).

Look-ahead bias note: download windows must end at or before the decision
timestamp. Missing-data policy: do **not** silently fill gaps.
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
            "API keys must come from RL_TRADER_DATA_API_* env vars."
        )
