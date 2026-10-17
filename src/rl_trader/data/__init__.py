"""Historical data download, cleaning, and validation (research-only)."""

from rl_trader.data.cleaning import CleaningResult, clean_ohlcv
from rl_trader.data.download import DownloadRequest, FixtureProvider, StubDownloader
from rl_trader.data.pipeline import (
    PipelineResult,
    SymbolPipelineResult,
    get_provider,
    run_data_pipeline,
)
from rl_trader.data.provider import DownloadMeta, MarketDataProvider, OHLCVDownload
from rl_trader.data.quality_report import DataQualityReport
from rl_trader.data.validation import ValidationReport, validate_ohlcv_frame

__all__ = [
    "CleaningResult",
    "DataQualityReport",
    "DownloadMeta",
    "DownloadRequest",
    "FixtureProvider",
    "MarketDataProvider",
    "OHLCVDownload",
    "PipelineResult",
    "StubDownloader",
    "SymbolPipelineResult",
    "ValidationReport",
    "clean_ohlcv",
    "get_provider",
    "run_data_pipeline",
    "validate_ohlcv_frame",
]
