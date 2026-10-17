"""Feature-quality report: missing values + distributions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureQualityReport:
    """Quality summary for a feature matrix."""

    symbol: str
    n_rows: int
    n_columns: int
    missing_by_column: dict[str, int]
    missing_fraction_by_column: dict[str, float]
    distribution_by_column: dict[str, dict[str, float]]
    row_policy: str
    ok: bool
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return asdict(self)


def build_feature_quality_report(
    features: pd.DataFrame,
    *,
    symbol: str,
) -> FeatureQualityReport:
    """Compute missingness and simple distribution stats (no row drops)."""
    warnings: list[str] = []
    missing_by_column: dict[str, int] = {}
    missing_fraction: dict[str, float] = {}
    distributions: dict[str, dict[str, float]] = {}

    n_rows = int(len(features))
    for col in features.columns:
        series = pd.to_numeric(features[col], errors="coerce")
        n_miss = int(series.isna().sum())
        missing_by_column[col] = n_miss
        missing_fraction[col] = float(n_miss / n_rows) if n_rows else 1.0
        valid = series.dropna()
        if valid.empty:
            distributions[col] = {
                "count": 0.0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p25": float("nan"),
                "p50": float("nan"),
                "p75": float("nan"),
                "max": float("nan"),
            }
            warnings.append(f"{col}: all values missing")
            continue
        distributions[col] = {
            "count": float(len(valid)),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "min": float(valid.min()),
            "p25": float(np.nanpercentile(valid, 25)),
            "p50": float(np.nanpercentile(valid, 50)),
            "p75": float(np.nanpercentile(valid, 75)),
            "max": float(valid.max()),
        }

    if n_rows == 0:
        warnings.append("feature matrix has zero rows")

    return FeatureQualityReport(
        symbol=symbol.upper(),
        n_rows=n_rows,
        n_columns=int(features.shape[1]),
        missing_by_column=missing_by_column,
        missing_fraction_by_column=missing_fraction,
        distribution_by_column=distributions,
        row_policy="keep_all_rows_with_explicit_nans",
        ok=n_rows > 0,
        warnings=warnings,
    )


def write_feature_quality_report(report: FeatureQualityReport, directory: Path) -> Path:
    """Write ``{symbol}_feature_quality.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.symbol}_feature_quality.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
