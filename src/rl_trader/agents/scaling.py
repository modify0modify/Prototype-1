"""Train-only feature scalers for exogenous episode feature blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rl_trader.environment.episode import EpisodeData

SCALER_FILENAME = "feature_scaler.json"


@dataclass
class FeatureScaler:
    """Column-wise z-score scaler for market / fundamental / regime features.

    ``fit`` must be called on the **training** episode only. ``transform`` never
    updates statistics. Portfolio-state observation components are not scaled
    here (they are constructed inside the env at runtime).
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    n_market: int = 0
    n_fundamental: int = 0
    n_regime: int = 0
    fitted_on: Literal["train", ""] = ""
    eps: float = 1e-8

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called."""
        return self.mean_ is not None and self.scale_ is not None

    def _stack(self, episode: EpisodeData) -> np.ndarray:
        return np.concatenate(
            [
                episode.market_features,
                episode.fundamental_features,
                episode.regime_features,
            ],
            axis=1,
        )

    def fit(self, episode: EpisodeData, *, split: Literal["train"] = "train") -> FeatureScaler:
        """Fit mean/std on ``episode`` (must be the training split)."""
        if split != "train":
            raise ValueError(
                "FeatureScaler.fit may only be called with split='train' "
                f"(got {split!r}) — never fit on validation or test"
            )
        stacked = self._stack(episode)
        if stacked.size == 0:
            self.mean_ = np.zeros(0, dtype=np.float64)
            self.scale_ = np.ones(0, dtype=np.float64)
        else:
            self.mean_ = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0, ddof=0)
            self.scale_ = np.where(std < self.eps, 1.0, std)
            # Replace non-finite means with 0 for sparse columns.
            self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        self.n_market = episode.n_market
        self.n_fundamental = episode.n_fundamental
        self.n_regime = episode.n_regime
        self.fitted_on = "train"
        return self

    def transform(self, episode: EpisodeData) -> EpisodeData:
        """Apply fitted scaler without updating statistics."""
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        if (
            episode.n_market != self.n_market
            or episode.n_fundamental != self.n_fundamental
            or episode.n_regime != self.n_regime
        ):
            raise ValueError(
                "episode feature widths do not match fitted scaler: "
                f"got ({episode.n_market}, {episode.n_fundamental}, {episode.n_regime}) "
                f"expected ({self.n_market}, {self.n_fundamental}, {self.n_regime})"
            )
        stacked = self._stack(episode)
        if stacked.shape[1] == 0:
            scaled = stacked
        else:
            scaled = (stacked - self.mean_) / self.scale_
            scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        i0 = 0
        i1 = self.n_market
        i2 = i1 + self.n_fundamental
        i3 = i2 + self.n_regime
        return EpisodeData(
            timestamps=episode.timestamps,
            prices=episode.prices,
            market_features=scaled[:, i0:i1],
            fundamental_features=scaled[:, i1:i2],
            regime_features=scaled[:, i2:i3],
            symbols=episode.symbols,
        )

    def fit_transform(
        self, episode: EpisodeData, *, split: Literal["train"] = "train"
    ) -> EpisodeData:
        """Fit on training episode and transform it."""
        return self.fit(episode, split=split).transform(episode)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        if not self.is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("cannot serialize an unfitted scaler")
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "n_market": self.n_market,
            "n_fundamental": self.n_fundamental,
            "n_regime": self.n_regime,
            "fitted_on": self.fitted_on,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureScaler:
        """Restore a fitted scaler from ``to_dict`` output."""
        fitted_on = str(data.get("fitted_on", "train"))
        if fitted_on != "train":
            raise ValueError(f"refusing scaler not fitted on train (got {fitted_on!r})")
        return cls(
            mean_=np.asarray(data["mean"], dtype=np.float64),
            scale_=np.asarray(data["scale"], dtype=np.float64),
            n_market=int(data["n_market"]),
            n_fundamental=int(data["n_fundamental"]),
            n_regime=int(data["n_regime"]),
            fitted_on="train",
            eps=float(data.get("eps", 1e-8)),
        )

    def save(self, directory: Path) -> Path:
        """Write scaler JSON next to the model checkpoint."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / SCALER_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> FeatureScaler:
        """Load scaler from ``directory / feature_scaler.json``."""
        path = directory / SCALER_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"missing feature scaler: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("scaler root must be an object")
        return cls.from_dict(data)
