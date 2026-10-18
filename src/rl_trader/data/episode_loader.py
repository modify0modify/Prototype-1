"""Build ``EpisodeData`` from cleaned OHLCV (+ features) for real-data training.

Never synthesizes prices. Missing, incomplete, or stale processed artifacts raise
``EpisodeDataError`` with an actionable message (download / build-features).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rl_trader.config.models import DataSettings, FeatureSettings, TrainingSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.features.fundamental_features import FUNDAMENTAL_COLUMNS
from rl_trader.features.pipeline import build_feature_matrix
from rl_trader.features.portfolio_state import portfolio_feature_specs
from rl_trader.features.regime import regime_feature_specs
from rl_trader.features.technical import technical_feature_specs


class EpisodeDataError(RuntimeError):
    """Processed market data cannot be loaded into an episode panel."""


@dataclass(frozen=True, slots=True)
class EpisodeLoadResult:
    """Episode panel plus provenance for logging / manifests."""

    episode: EpisodeData
    ohlcv_paths: dict[str, Path]
    feature_sources: dict[str, str]  # symbol -> "parquet" | "built_from_ohlcv"
    index_start: date
    index_end: date


def find_cleaned_ohlcv(processed_dir: Path, symbol: str) -> Path:
    """Return the newest cleaned OHLCV parquet for ``symbol`` under processed_dir."""
    symbol = symbol.upper()
    matches = sorted((processed_dir / "ohlcv").glob(f"{symbol}_*.parquet"))
    if not matches:
        matches = sorted(processed_dir.glob(f"**/ohlcv/{symbol}_*.parquet"))
    if not matches:
        raise EpisodeDataError(
            f"missing cleaned OHLCV for {symbol} under {processed_dir / 'ohlcv'}; "
            "run: python scripts/download_data.py --config <config> "
            "(or make download-data). Refusing synthetic fallback."
        )
    return matches[-1]


def find_feature_parquet(features_dir: Path, symbol: str) -> Path | None:
    """Return ``{SYMBOL}_features.parquet`` if present."""
    path = features_dir / f"{symbol.upper()}_features.parquet"
    return path if path.is_file() else None


def _frame_dates(index: pd.Index) -> list[date]:
    out: list[date] = []
    for ts in index:
        if isinstance(ts, date) and not isinstance(ts, pd.Timestamp):
            out.append(ts)
        else:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("UTC")
            out.append(stamp.date())
    return out


def _as_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize an index to a UTC ``DatetimeIndex``."""
    dt = pd.DatetimeIndex(index)
    if dt.tz is None:
        return dt.tz_localize("UTC")
    return dt.tz_convert("UTC")


def _load_ohlcv_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        if "timestamp" in frame.columns:
            frame = frame.set_index("timestamp")
        else:
            raise EpisodeDataError(f"OHLCV at {path} has no DatetimeIndex or timestamp column")
    frame.index = _as_utc_index(frame.index)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _close_series(ohlcv: pd.DataFrame) -> pd.Series:
    if "close_adj" in ohlcv.columns:
        return ohlcv["close_adj"].astype(float)
    if "close" in ohlcv.columns:
        return ohlcv["close"].astype(float)
    raise EpisodeDataError("OHLCV frame missing close / close_adj")


def _technical_names(settings: FeatureSettings) -> list[str]:
    return [s.name for s in technical_feature_specs(settings)]


def _regime_names(settings: FeatureSettings) -> list[str]:
    return [s.name for s in regime_feature_specs(settings)]


def _fundamental_names(settings: FeatureSettings) -> list[str]:
    if not settings.include_fundamentals:
        return []
    return list(FUNDAMENTAL_COLUMNS)


def _portfolio_names(settings: FeatureSettings) -> set[str]:
    if not settings.include_portfolio_state:
        return set()
    return {s.name for s in portfolio_feature_specs(settings)}


def _ensure_columns(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name in names:
        if name in frame.columns:
            out[name] = pd.to_numeric(frame[name], errors="coerce")
        else:
            out[name] = np.nan
    return out


def _build_or_load_features(
    symbol: str,
    ohlcv: pd.DataFrame,
    settings: FeatureSettings,
    *,
    benchmark_ohlcv: pd.DataFrame,
    features_dir: Path,
    build_if_missing: bool,
) -> tuple[pd.DataFrame, str]:
    existing = find_feature_parquet(features_dir, symbol)
    if existing is not None:
        feats = pd.read_parquet(existing)
        if not isinstance(feats.index, pd.DatetimeIndex):
            raise EpisodeDataError(f"feature parquet {existing} lacks DatetimeIndex")
        feats.index = _as_utc_index(feats.index)
        return feats.sort_index(), "parquet"

    if not build_if_missing:
        raise EpisodeDataError(
            f"missing features for {symbol} under {features_dir}; "
            "run: python scripts/build_features.py --config <config> "
            "(or make build-features). Refusing synthetic fallback."
        )

    features, _specs = build_feature_matrix(
        ohlcv,
        settings,
        symbol=symbol,
        benchmark_ohlcv=benchmark_ohlcv,
        portfolio_state=None,
        fundamental_events=None,
        valuation=None,
    )
    return features, "built_from_ohlcv"


def _validate_split_coverage(
    calendar: list[date],
    training: TrainingSettings,
) -> None:
    if not calendar:
        raise EpisodeDataError("aligned OHLCV calendar is empty")

    cal_start, cal_end = calendar[0], calendar[-1]
    if cal_start > training.train_start:
        raise EpisodeDataError(
            f"processed OHLCV starts on {cal_start.isoformat()}, after train_start "
            f"{training.train_start.isoformat()}; re-download with a start date that "
            "covers training.train_start (see data.start_date / --start). "
            "Refusing synthetic fallback."
        )
    if cal_end < training.test_end:
        raise EpisodeDataError(
            f"processed OHLCV ends on {cal_end.isoformat()}, before test_end "
            f"{training.test_end.isoformat()} (stale/incomplete); re-download with an "
            "end date covering training.test_end (data.end_date / --end). "
            "Refusing synthetic fallback."
        )

    def _count(start: date, end: date) -> int:
        return sum(1 for d in calendar if start <= d <= end)

    counts = {
        "train": _count(training.train_start, training.train_end),
        "val": _count(training.val_start, training.val_end),
        "test": _count(training.test_start, training.test_end),
    }
    short = {k: v for k, v in counts.items() if v < 2}
    if short:
        raise EpisodeDataError(
            "train/val/test windows need ≥2 trading bars each after aligning the "
            f"universe; got {counts}. Check training.* date splits vs downloaded "
            "history. Refusing synthetic fallback."
        )


def _finite_fill(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite values with 0 (warm-up / missing fundamentals)."""
    return np.where(np.isfinite(arr), arr, 0.0)


def load_episode_from_processed(
    data: DataSettings,
    features: FeatureSettings,
    training: TrainingSettings,
    *,
    universe: tuple[str, ...] | None = None,
    build_features_if_missing: bool = True,
) -> EpisodeLoadResult:
    """Load an aligned multi-asset ``EpisodeData`` from processed artifacts.

    Parameters
    ----------
    data / features / training:
        Config blocks (paths, universe, feature knobs, train/val/test dates).
    universe:
        Optional override (defaults to ``data.universe``).
    build_features_if_missing:
        When True, build causal features from cleaned OHLCV if parquet files are
        absent. When False, require existing feature parquets.
    """
    symbols = tuple(s.upper() for s in (universe or data.universe))
    if len(symbols) < 1:
        raise EpisodeDataError("universe must contain at least one symbol")

    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    ohlcv_paths: dict[str, Path] = {}
    for symbol in symbols:
        path = find_cleaned_ohlcv(data.processed_dir, symbol)
        ohlcv_paths[symbol] = path
        ohlcv_by_symbol[symbol] = _load_ohlcv_frame(path)

    # Full intersection calendar (warm-up history before train_start is kept for
    # feature construction, then the episode panel is sliced to the split span).
    full_index: pd.DatetimeIndex | None = None
    for frame in ohlcv_by_symbol.values():
        idx = pd.DatetimeIndex(frame.index)
        full_index = idx if full_index is None else full_index.intersection(idx)
    if full_index is None or len(full_index) < 2:
        raise EpisodeDataError(
            "universe OHLCV intersection has fewer than 2 shared sessions; "
            "check cleaned parquet coverage. Refusing synthetic fallback."
        )
    full_index = full_index.sort_values()

    panel_start = pd.Timestamp(training.train_start, tz="UTC")
    panel_end = pd.Timestamp(training.test_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    episode_index = full_index[(full_index >= panel_start) & (full_index <= panel_end)]
    if len(episode_index) < 2:
        raise EpisodeDataError(
            f"no shared bars in [{training.train_start} .. {training.test_end}] "
            "after intersecting the universe; re-download data covering the "
            "configured train/val/test windows. Refusing synthetic fallback."
        )

    calendar = _frame_dates(episode_index)
    _validate_split_coverage(calendar, training)

    prices = np.column_stack(
        [
            _close_series(ohlcv_by_symbol[sym].reindex(episode_index)).to_numpy(
                dtype=np.float64
            )
            for sym in symbols
        ]
    )
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise EpisodeDataError(
            "aligned close prices contain non-finite or non-positive values; "
            "fix or re-download cleaned OHLCV. Refusing synthetic fallback."
        )

    bench_symbol = features.benchmark_symbol.upper()
    if bench_symbol in ohlcv_by_symbol:
        bench_full = ohlcv_by_symbol[bench_symbol].reindex(full_index)
    else:
        # Benchmark may sit outside the tradable universe — still required for regime.
        try:
            bench_path = find_cleaned_ohlcv(data.processed_dir, bench_symbol)
        except EpisodeDataError as exc:
            raise EpisodeDataError(
                f"benchmark {bench_symbol} OHLCV required for features/regime but missing; "
                f"{exc}"
            ) from exc
        bench_full = _load_ohlcv_frame(bench_path).reindex(full_index)

    feature_sources: dict[str, str] = {}
    tech_names = _technical_names(features)
    fund_names = _fundamental_names(features)
    regime_names = _regime_names(features) if features.include_regime else []
    skip_cols = _portfolio_names(features)

    market_blocks: list[np.ndarray] = []
    fund_blocks: list[np.ndarray] = []
    regime_block: np.ndarray | None = None
    regime_source_symbol = bench_symbol if bench_symbol in symbols else symbols[0]

    for symbol in symbols:
        ohlcv_full = ohlcv_by_symbol[symbol].reindex(full_index)
        feats, source = _build_or_load_features(
            symbol,
            ohlcv_full,
            features,
            benchmark_ohlcv=bench_full,
            features_dir=features.output_dir,
            build_if_missing=build_features_if_missing,
        )
        feature_sources[symbol] = source
        feats = feats.reindex(episode_index)
        keep = [c for c in feats.columns if c not in skip_cols]
        feats = feats.loc[:, keep]

        tech = _ensure_columns(feats, tech_names)
        market_blocks.append(tech.to_numpy(dtype=np.float64))

        if fund_names:
            fund = _ensure_columns(feats, fund_names)
            fund_blocks.append(fund.to_numpy(dtype=np.float64))

        if regime_names and regime_block is None and symbol == regime_source_symbol:
            regime_block = _ensure_columns(feats, regime_names).to_numpy(dtype=np.float64)

    if regime_names and regime_block is None:
        feats, source = _build_or_load_features(
            bench_symbol,
            bench_full,
            features,
            benchmark_ohlcv=bench_full,
            features_dir=features.output_dir,
            build_if_missing=build_features_if_missing,
        )
        feature_sources[bench_symbol] = source
        feats = feats.reindex(episode_index)
        regime_block = _ensure_columns(feats, regime_names).to_numpy(dtype=np.float64)

    market = _finite_fill(np.concatenate(market_blocks, axis=1))
    if fund_blocks:
        fundamental = _finite_fill(np.concatenate(fund_blocks, axis=1))
    else:
        fundamental = np.zeros((len(episode_index), 0), dtype=np.float64)
    if regime_block is None:
        regime = np.zeros((len(episode_index), 0), dtype=np.float64)
    else:
        regime = _finite_fill(regime_block)

    timestamps = np.array(calendar, dtype=object)
    episode = EpisodeData(
        timestamps=timestamps,
        prices=prices,
        market_features=market,
        fundamental_features=fundamental,
        regime_features=regime,
        symbols=symbols,
    )
    return EpisodeLoadResult(
        episode=episode,
        ohlcv_paths=ohlcv_paths,
        feature_sources=feature_sources,
        index_start=calendar[0],
        index_end=calendar[-1],
    )
