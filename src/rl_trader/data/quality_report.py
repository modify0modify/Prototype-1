"""Data-quality report artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_trader.data.provider import DownloadMeta
from rl_trader.data.schema import ADJUSTMENT_METHOD
from rl_trader.data.validation import ValidationReport


@dataclass(slots=True)
class DataQualityReport:
    """Structured quality report for one symbol download."""

    symbol: str
    provider: str
    downloaded_at: str
    requested_start: str
    requested_end: str
    timezone: str
    raw_path: str
    cleaned_path: str
    cache_path: str
    n_rows_raw: int
    n_rows_cleaned: int
    duplicates_removed: int
    adjustment_method: str
    validation: dict[str, Any]
    ok: bool
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return asdict(self)


def build_quality_report(
    *,
    meta: DownloadMeta,
    validation: ValidationReport,
    raw_path: Path,
    cleaned_path: Path,
    cache_path: Path,
    n_rows_raw: int,
    n_rows_cleaned: int,
    duplicates_removed: int,
    adjustment_method: str = ADJUSTMENT_METHOD,
) -> DataQualityReport:
    """Assemble a quality report from pipeline outputs."""
    return DataQualityReport(
        symbol=meta.symbol,
        provider=meta.provider,
        downloaded_at=meta.downloaded_at.astimezone(UTC).isoformat(),
        requested_start=meta.requested_start.isoformat(),
        requested_end=meta.requested_end.isoformat(),
        timezone=meta.timezone,
        raw_path=str(raw_path),
        cleaned_path=str(cleaned_path),
        cache_path=str(cache_path),
        n_rows_raw=n_rows_raw,
        n_rows_cleaned=n_rows_cleaned,
        duplicates_removed=duplicates_removed,
        adjustment_method=adjustment_method,
        validation=validation.to_dict(),
        ok=validation.ok,
    )


def write_quality_report(report: DataQualityReport, directory: Path) -> Path:
    """Write ``{symbol}_quality.json`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.symbol}_quality.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
