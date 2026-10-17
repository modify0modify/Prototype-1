"""Market-regime bucketing for walk-forward fold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_trader.environment.episode import EpisodeData


@dataclass(frozen=True, slots=True)
class RegimeBucketizer:
    """Quantile buckets fit on a training window's regime feature only."""

    feature_index: int
    edges: tuple[float, ...]
    labels: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Label each value; edges are increasing quantile cut points."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.empty(arr.shape[0], dtype=object)
        edges = list(self.edges)
        for i, v in enumerate(arr):
            bucket = 0
            for e in edges:
                if v > e:
                    bucket += 1
                else:
                    break
            bucket = min(bucket, len(self.labels) - 1)
            out[i] = self.labels[bucket]
        return out


def fit_regime_bucketizer(
    train_episode: EpisodeData,
    *,
    feature_index: int = 0,
    n_buckets: int = 3,
) -> RegimeBucketizer:
    """Fit quantile edges on train regime column (never test)."""
    if train_episode.n_regime < 1:
        return RegimeBucketizer(
            feature_index=0,
            edges=(),
            labels=("no_regime",),
        )
    idx = min(feature_index, train_episode.n_regime - 1)
    col = train_episode.regime_features[:, idx]
    finite = col[np.isfinite(col)]
    if finite.size == 0 or n_buckets < 2:
        return RegimeBucketizer(feature_index=idx, edges=(), labels=("flat",))
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges = tuple(float(x) for x in np.quantile(finite, qs))
    labels = tuple(f"regime_{i}" for i in range(n_buckets))
    return RegimeBucketizer(feature_index=idx, edges=edges, labels=labels)


def label_episode_regimes(
    episode: EpisodeData, bucketizer: RegimeBucketizer
) -> np.ndarray:
    """Label each bar of ``episode`` (length T); returns may use T-1 labels."""
    if episode.n_regime < 1 or not bucketizer.labels:
        return np.array(["no_regime"] * episode.n_bars, dtype=object)
    idx = min(bucketizer.feature_index, episode.n_regime - 1)
    return bucketizer.transform(episode.regime_features[:, idx])


def regime_performance(
    daily_returns: np.ndarray, regime_labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Mean / std / count of daily returns by regime label."""
    rets = np.asarray(daily_returns, dtype=np.float64)
    labels = np.asarray(regime_labels, dtype=object)
    n = min(rets.size, labels.size)
    out: dict[str, dict[str, float]] = {}
    for lab in sorted(set(str(x) for x in labels[:n])):
        mask = np.array([str(x) == lab for x in labels[:n]], dtype=bool)
        subset = rets[:n][mask]
        out[lab] = {
            "mean_return": float(np.mean(subset)) if subset.size else 0.0,
            "std_return": float(np.std(subset, ddof=0)) if subset.size else 0.0,
            "n_days": float(subset.size),
            "cumulative_return": (
                float(np.prod(1.0 + subset) - 1.0) if subset.size else 0.0
            ),
        }
    return out


def merge_regime_tables(
    per_fold: list[dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Aggregate per-fold regime stats (means of means + total days)."""
    keys: set[str] = set()
    for table in per_fold:
        keys.update(table.keys())
    merged: dict[str, Any] = {}
    for key in sorted(keys):
        means = [t[key]["mean_return"] for t in per_fold if key in t]
        days = [t[key]["n_days"] for t in per_fold if key in t]
        merged[key] = {
            "mean_of_fold_mean_return": float(np.mean(means)) if means else 0.0,
            "std_of_fold_mean_return": float(np.std(means, ddof=0)) if means else 0.0,
            "total_days": float(np.sum(days)) if days else 0.0,
            "n_folds": float(len(means)),
        }
    return merged
