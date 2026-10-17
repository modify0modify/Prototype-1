"""Orchestrate look-ahead-safe feature construction from cleaned OHLCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings, FundamentalValuationSettings
from rl_trader.features.engineering import FeatureSpec
from rl_trader.features.fundamental_features import (
    FundamentalEvent,
    compute_fundamental_features,
    fundamental_feature_specs,
)
from rl_trader.features.manifest import FeatureManifest, build_manifest, write_manifest
from rl_trader.features.portfolio_state import (
    PortfolioStateSeries,
    compute_portfolio_state_features,
    portfolio_feature_specs,
)
from rl_trader.features.quality import (
    FeatureQualityReport,
    build_feature_quality_report,
    write_feature_quality_report,
)
from rl_trader.features.regime import compute_regime_features, regime_feature_specs
from rl_trader.features.technical import compute_technical_features, technical_feature_specs


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """Outputs of a single-symbol feature build."""

    symbol: str
    features: pd.DataFrame
    feature_path: Path
    manifest_path: Path
    quality_path: Path
    manifest: FeatureManifest
    quality: FeatureQualityReport
    specs: tuple[FeatureSpec, ...]


def collect_feature_specs(settings: FeatureSettings) -> list[FeatureSpec]:
    """All feature specs for the configured blocks."""
    specs = technical_feature_specs(settings)
    if settings.include_regime:
        specs.extend(regime_feature_specs(settings))
    if settings.include_portfolio_state:
        specs.extend(portfolio_feature_specs(settings))
    if settings.include_fundamentals:
        specs.extend(fundamental_feature_specs(settings))
    return specs


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    settings: FeatureSettings,
    *,
    symbol: str,
    benchmark_ohlcv: pd.DataFrame | None = None,
    portfolio_state: PortfolioStateSeries | None = None,
    fundamental_events: list[FundamentalEvent] | None = None,
    valuation: FundamentalValuationSettings | None = None,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Build the full feature matrix (same row count as ``ohlcv``).

    Row policy: never drop rows. Warm-up windows and missing fundamentals remain
    explicit NaNs and are summarized in the quality report.
    """
    if ohlcv.empty:
        raise ValueError("ohlcv frame is empty — refusing to build features")
    if "close" not in ohlcv.columns and "close_adj" not in ohlcv.columns:
        raise ValueError("ohlcv must include close or close_adj")

    close_col = "close_adj" if "close_adj" in ohlcv.columns else "close"
    price = ohlcv[close_col].astype(float)

    bench_close = None
    if benchmark_ohlcv is not None:
        bcol = "close_adj" if "close_adj" in benchmark_ohlcv.columns else "close"
        bench_close = benchmark_ohlcv[bcol].astype(float)

    parts = [compute_technical_features(ohlcv, settings, benchmark_close=bench_close)]

    if settings.include_regime:
        if bench_close is None:
            # Without a benchmark, emit NaN regime columns (still keep rows)
            regime = pd.DataFrame(index=ohlcv.index)
            for spec in regime_feature_specs(settings):
                regime[spec.name] = float("nan")
        else:
            regime = compute_regime_features(bench_close, settings, index=ohlcv.index)
        parts.append(regime)

    if settings.include_portfolio_state:
        parts.append(
            compute_portfolio_state_features(ohlcv.index, settings, portfolio_state)
        )

    if settings.include_fundamentals:
        parts.append(
            compute_fundamental_features(
                pd.DatetimeIndex(ohlcv.index),
                settings,
                events=fundamental_events,
                price=price,
                valuation=valuation,
            )
        )

    features = pd.concat(parts, axis=1)
    if len(features) != len(ohlcv):
        raise RuntimeError("feature pipeline changed row count — silent drops are forbidden")

    if settings.normalize:
        # Causal expanding z-score would still be look-ahead safe with shift;
        # we use trailing rolling z-score with lookback_bars (causal).
        window = settings.lookback_bars
        normed = features.copy()
        for col in features.columns:
            mean = features[col].rolling(window=window, min_periods=window).mean()
            std = features[col].rolling(window=window, min_periods=window).std(ddof=1)
            normed[col] = (features[col] - mean) / std.replace(0.0, np.nan)
        features = normed

    specs = collect_feature_specs(settings)
    return features, specs


def run_feature_pipeline(
    ohlcv: pd.DataFrame,
    settings: FeatureSettings,
    *,
    symbol: str,
    benchmark_ohlcv: pd.DataFrame | None = None,
    portfolio_state: PortfolioStateSeries | None = None,
    fundamental_events: list[FundamentalEvent] | None = None,
    valuation: FundamentalValuationSettings | None = None,
) -> FeatureBuildResult:
    """Build features, write matrix + manifest + quality report."""
    features, specs = build_feature_matrix(
        ohlcv,
        settings,
        symbol=symbol,
        benchmark_ohlcv=benchmark_ohlcv,
        portfolio_state=portfolio_state,
        fundamental_events=fundamental_events,
        valuation=valuation,
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = settings.output_dir / f"{symbol.upper()}_features.parquet"
    features.to_parquet(feature_path)

    manifest = build_manifest(
        symbol=symbol,
        settings=settings,
        specs=specs,
        n_rows=int(len(features)),
        n_feature_columns=int(features.shape[1]),
    )
    manifest_path = write_manifest(manifest, settings.manifest_dir)

    quality = build_feature_quality_report(features, symbol=symbol)
    quality_path = write_feature_quality_report(quality, settings.quality_report_dir)

    return FeatureBuildResult(
        symbol=symbol.upper(),
        features=features,
        feature_path=feature_path,
        manifest_path=manifest_path,
        quality_path=quality_path,
        manifest=manifest,
        quality=quality,
        specs=tuple(specs),
    )
