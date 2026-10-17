"""Feature-definition matching for checkpoint loads."""

from __future__ import annotations

from collections.abc import Sequence

from rl_trader.features.engineering import FeatureSpec


class FeatureMismatchError(ValueError):
    """Raised when checkpoint features do not match the required set."""


def _spec_key(spec: FeatureSpec) -> tuple[str, int, str, bool]:
    return (spec.name, spec.lookback, spec.version, spec.causal_only)


def assert_features_match(
    checkpoint_features: Sequence[FeatureSpec],
    required_features: Sequence[FeatureSpec],
) -> None:
    """Refuse load when required feature definitions do not match.

    Comparison is order-sensitive and includes name, lookback, version, and
    ``causal_only``.
    """
    left = [_spec_key(s) for s in checkpoint_features]
    right = [_spec_key(s) for s in required_features]
    if left != right:
        raise FeatureMismatchError(
            "checkpoint feature definitions do not match required features: "
            f"checkpoint={left!r} required={right!r}"
        )
