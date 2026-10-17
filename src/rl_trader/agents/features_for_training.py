"""Derive checkpoint feature specs from an episode panel + feature settings."""

from __future__ import annotations

from rl_trader.config.models import FeatureSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.features.engineering import FeatureSpec


def feature_specs_from_episode(
    episode: EpisodeData,
    settings: FeatureSettings | None = None,
) -> tuple[FeatureSpec, ...]:
    """Build ordered feature specs matching exogenous observation blocks."""
    version = settings.feature_version if settings is not None else "1"
    specs: list[FeatureSpec] = []
    for i in range(episode.n_market):
        specs.append(
            FeatureSpec(name=f"market_{i}", lookback=1, version=version, causal_only=True)
        )
    for i in range(episode.n_fundamental):
        specs.append(
            FeatureSpec(
                name=f"fundamental_{i}", lookback=1, version=version, causal_only=True
            )
        )
    for i in range(episode.n_regime):
        specs.append(
            FeatureSpec(name=f"regime_{i}", lookback=1, version=version, causal_only=True)
        )
    return tuple(specs)
