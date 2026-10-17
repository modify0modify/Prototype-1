"""Train / validation / test episode splits (no leakage across windows)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from rl_trader.config.models import TrainingSettings
from rl_trader.environment.episode import EpisodeData


@dataclass(frozen=True, slots=True)
class SplitEpisodes:
    """Three non-overlapping episode panels."""

    train: EpisodeData
    validation: EpisodeData
    test: EpisodeData
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[no-any-return]
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"cannot coerce timestamp to date: {value!r}")


def _mask_for_range(timestamps: np.ndarray, start: date, end: date) -> np.ndarray:
    dates = np.array([_as_date(ts) for ts in timestamps], dtype=object)
    return np.array([(start <= d <= end) for d in dates], dtype=bool)


def split_episode_by_dates(
    episode: EpisodeData,
    settings: TrainingSettings,
) -> SplitEpisodes:
    """Slice ``episode`` into train / val / test by inclusive calendar dates.

    Raises if any window has fewer than 2 bars or if index ranges overlap.
    """
    train_mask = _mask_for_range(
        episode.timestamps, settings.train_start, settings.train_end
    )
    val_mask = _mask_for_range(episode.timestamps, settings.val_start, settings.val_end)
    test_mask = _mask_for_range(
        episode.timestamps, settings.test_start, settings.test_end
    )
    if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(
        val_mask & test_mask
    ):
        raise ValueError("train / validation / test date masks overlap")

    train_idx = tuple(int(i) for i in np.flatnonzero(train_mask))
    val_idx = tuple(int(i) for i in np.flatnonzero(val_mask))
    test_idx = tuple(int(i) for i in np.flatnonzero(test_mask))
    if len(train_idx) < 2 or len(val_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            "each of train/val/test must contain at least 2 bars; "
            f"got train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    # Contiguous slices from first to last selected index (inclusive).
    def _slice(indices: tuple[int, ...]) -> EpisodeData:
        return episode.slice_bars(indices[0], indices[-1] + 1)

    return SplitEpisodes(
        train=_slice(train_idx),
        validation=_slice(val_idx),
        test=_slice(test_idx),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )


def split_episode_by_fractions(
    episode: EpisodeData,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> SplitEpisodes:
    """Contiguous fractional split for synthetic fixtures (test = remainder)."""
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0):
        raise ValueError("fractions must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    n = episode.n_bars
    train_end = max(2, int(n * train_frac))
    val_end = max(train_end + 2, int(n * (train_frac + val_frac)))
    if n - val_end < 2:
        raise ValueError(f"episode too short for fractional split: n_bars={n}")
    train_idx = tuple(range(0, train_end))
    val_idx = tuple(range(train_end, val_end))
    test_idx = tuple(range(val_end, n))
    return SplitEpisodes(
        train=episode.slice_bars(0, train_end),
        validation=episode.slice_bars(train_end, val_end),
        test=episode.slice_bars(val_end, n),
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
