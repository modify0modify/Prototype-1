"""Evaluation evidence consumed by approval gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ApprovalEvidence:
    """Structured metrics / flags for paper and live approval gates.

    Values are supplied by offline evaluation / paper-trading telemetry — the
    gate engine compares them to **configurable** thresholds.
    """

    # Paper-tier evidence
    unresolved_critical_audit_findings: int = 0
    oos_return: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    transaction_costs_included: bool = False
    evaluated_market_periods: int = 0
    regimes_evaluated: int = 0
    feature_manifest_mismatch: bool = True
    model_metadata_missing: bool = True
    regression_tests_failed: bool = True
    excess_return_vs_benchmark: float = 0.0
    benchmark_comparison_present: bool = False

    # Live-tier evidence (ignored for paper-only approval)
    paper_duration_days: float = 0.0
    paper_trades: int = 0
    paper_sharpe: float = 0.0
    paper_max_drawdown: float = 0.0
    slippage_difference_bps: float = 0.0
    unresolved_reconciliation_errors: int = 0
    kill_switch_events_requiring_investigation: int = 0
    manual_approval_record: bool = False
    explicit_live_capital_limit: bool = False
    broker_sandbox_test_passed: bool = False

    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalEvidence:
        """Restore from a mapping (unknown keys → extras)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for key, value in data.items():
            if key in known and key != "extras":
                kwargs[key] = value
            elif key == "extras" and isinstance(value, dict):
                extras.update(value)
            else:
                extras[key] = value
        kwargs["extras"] = extras
        return cls(**kwargs)


def example_passing_paper_evidence() -> ApprovalEvidence:
    """Evidence that satisfies the **example** paper defaults (for demos/tests)."""
    return ApprovalEvidence(
        unresolved_critical_audit_findings=0,
        oos_return=0.02,
        max_drawdown=0.10,
        turnover=0.8,
        transaction_costs_included=True,
        evaluated_market_periods=5,
        regimes_evaluated=3,
        feature_manifest_mismatch=False,
        model_metadata_missing=False,
        regression_tests_failed=False,
        excess_return_vs_benchmark=0.01,
        benchmark_comparison_present=True,
    )


def example_passing_live_evidence() -> ApprovalEvidence:
    """Evidence that satisfies example paper **and** live defaults."""
    base = example_passing_paper_evidence()
    base.paper_duration_days = 45.0
    base.paper_trades = 80
    base.paper_sharpe = 0.9
    base.paper_max_drawdown = 0.12
    base.slippage_difference_bps = 1.0
    base.unresolved_reconciliation_errors = 0
    base.kill_switch_events_requiring_investigation = 0
    base.manual_approval_record = True
    base.explicit_live_capital_limit = True
    base.broker_sandbox_test_passed = True
    return base
