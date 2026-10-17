"""Orchestrate download → cache → raw → cleaned → validate → quality report."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from rl_trader.config.models import DataSettings
from rl_trader.data.cache import load_cached, paths_for, save_cached
from rl_trader.data.cleaning import clean_ohlcv
from rl_trader.data.provider import MarketDataProvider, OHLCVDownload
from rl_trader.data.quality_report import (
    DataQualityReport,
    build_quality_report,
    write_quality_report,
)
from rl_trader.data.validation import ValidationReport, validate_ohlcv_frame
from rl_trader.data.yfinance_provider import YFinanceProvider


@dataclass(frozen=True, slots=True)
class SymbolPipelineResult:
    """Per-symbol pipeline outputs."""

    symbol: str
    raw_path: Path
    cleaned_path: Path
    cache_parquet: Path
    quality_report_path: Path
    validation: ValidationReport
    quality: DataQualityReport


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Multi-symbol pipeline summary."""

    results: tuple[SymbolPipelineResult, ...]

    @property
    def ok(self) -> bool:
        """True when every symbol validation passed."""
        return all(r.validation.ok for r in self.results)


def get_provider(name: str) -> MarketDataProvider:
    """Factory for configured providers (swap here without touching callers)."""
    if name == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"unsupported market data provider: {name!r}")


def default_date_range(settings: DataSettings) -> tuple[date, date]:
    """Resolve start/end from settings or a 2-year lookback ending today (UTC date)."""
    end = settings.end_date or date.today()
    start = settings.start_date or (end - timedelta(days=365 * 2))
    if end < start:
        raise ValueError("data.end_date must be >= data.start_date")
    return start, end


def run_symbol_pipeline(
    symbol: str,
    settings: DataSettings,
    provider: MarketDataProvider,
    *,
    start: date,
    end: date,
    use_cache: bool = True,
) -> SymbolPipelineResult:
    """Run the full daily OHLCV pipeline for one symbol."""
    symbol = symbol.upper()
    cache_dir = settings.cache_dir
    raw_dir = settings.raw_dir / provider.name
    cleaned_dir = settings.processed_dir / "ohlcv"
    report_dir = settings.quality_report_dir

    cached = load_cached(cache_dir, provider.name, symbol, start, end) if use_cache else None
    if cached is None:
        download = provider.fetch_daily_ohlcv(symbol, start, end)
        cache_paths = save_cached(cache_dir, download)
    else:
        download = cached
        cache_paths = paths_for(cache_dir, provider.name, symbol, start, end)

    # Preserve raw unchanged (copy from cache into raw archive)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_parquet = raw_dir / f"{symbol}_{start.isoformat()}_{end.isoformat()}.parquet"
    raw_meta = raw_dir / f"{symbol}_{start.isoformat()}_{end.isoformat()}.meta.json"
    if not raw_parquet.exists():
        shutil.copy2(cache_paths.parquet, raw_parquet)
    raw_meta.write_text(
        json.dumps(download.meta.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cleaning = clean_ohlcv(download.frame)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"{symbol}_{start.isoformat()}_{end.isoformat()}.parquet"
    cleaning.frame.to_parquet(cleaned_path)

    validation = validate_ohlcv_frame(
        cleaning.frame,
        start=start,
        end=end,
        jump_threshold=settings.jump_threshold,
        allow_missing_bars=settings.allow_missing_bars,
    )
    if cleaning.warnings:
        validation.warnings.extend(cleaning.warnings)
    validation.metadata["duplicates_removed"] = cleaning.duplicates_removed
    validation.metadata["adjustment_method"] = cleaning.adjustment_method

    quality = build_quality_report(
        meta=download.meta,
        validation=validation,
        raw_path=raw_parquet,
        cleaned_path=cleaned_path,
        cache_path=cache_paths.parquet,
        n_rows_raw=int(len(download.frame)),
        n_rows_cleaned=int(len(cleaning.frame)),
        duplicates_removed=cleaning.duplicates_removed,
        adjustment_method=cleaning.adjustment_method,
    )
    quality_path = write_quality_report(quality, report_dir)

    return SymbolPipelineResult(
        symbol=symbol,
        raw_path=raw_parquet,
        cleaned_path=cleaned_path,
        cache_parquet=cache_paths.parquet,
        quality_report_path=quality_path,
        validation=validation,
        quality=quality,
    )


def run_data_pipeline(
    settings: DataSettings,
    *,
    provider: MarketDataProvider | None = None,
    start: date | None = None,
    end: date | None = None,
    symbols: tuple[str, ...] | None = None,
    use_cache: bool = True,
) -> PipelineResult:
    """Download/validate the configured universe (or an override symbol list)."""
    resolved_start, resolved_end = default_date_range(settings)
    if start is not None:
        resolved_start = start
    if end is not None:
        resolved_end = end
    universe = symbols or settings.universe
    active_provider = provider or get_provider(settings.provider)

    results = [
        run_symbol_pipeline(
            symbol,
            settings,
            active_provider,
            start=resolved_start,
            end=resolved_end,
            use_cache=use_cache,
        )
        for symbol in universe
    ]
    return PipelineResult(results=tuple(results))


# Re-export for type checkers / callers that only need the download type
__all__ = [
    "OHLCVDownload",
    "PipelineResult",
    "SymbolPipelineResult",
    "default_date_range",
    "get_provider",
    "run_data_pipeline",
    "run_symbol_pipeline",
]
