"""Look-ahead-safe feature engineering (research surface — no live imports)."""

from rl_trader.features.builder import CausalFeatureBuilder, StubFeatureBuilder
from rl_trader.features.dcf_bridge import (
    dcf_result_to_event,
    events_from_statements,
    intrinsic_value_asof,
)
from rl_trader.features.engineering import FeatureSpec
from rl_trader.features.fundamental_features import FundamentalEvent
from rl_trader.features.manifest import FeatureManifest
from rl_trader.features.pipeline import FeatureBuildResult, run_feature_pipeline
from rl_trader.features.portfolio_state import PortfolioStateSeries
from rl_trader.features.quality import FeatureQualityReport

__all__ = [
    "CausalFeatureBuilder",
    "FeatureBuildResult",
    "FeatureManifest",
    "FeatureQualityReport",
    "FeatureSpec",
    "FundamentalEvent",
    "PortfolioStateSeries",
    "StubFeatureBuilder",
    "dcf_result_to_event",
    "events_from_statements",
    "intrinsic_value_asof",
    "run_feature_pipeline",
]
