"""Historical data download, cleaning, and validation (research-only)."""

from rl_trader.data.cleaning import CleaningResult, clean_ohlcv
from rl_trader.data.download import DownloadRequest, FixtureProvider, StubDownloader
from rl_trader.data.episode_loader import (
    EpisodeDataError,
    EpisodeLoadResult,
    find_cleaned_ohlcv,
    load_episode_from_processed,
    required_feature_warmup_bars,
)
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
    "EpisodeDataError",
    "EpisodeLoadResult",
    "FixtureProvider",
    "MarketDataProvider",
    "OHLCVDownload",
    "PipelineResult",
    "StubDownloader",
    "SymbolPipelineResult",
    "ValidationReport",
    "clean_ohlcv",
    "find_cleaned_ohlcv",
    "get_provider",
    "load_episode_from_processed",
    "required_feature_warmup_bars",
    "run_data_pipeline",
    "validate_ohlcv_frame",
]
