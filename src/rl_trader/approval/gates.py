"""Configurable paper / live model-approval gates.

Thresholds come from :class:`~rl_trader.config.models.ApprovalSettings` —
nothing financial is hard-coded here beyond comparisons against those values.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_trader.approval.evidence import ApprovalEvidence
from rl_trader.config.models import (
    ApprovalSettings,
    LiveApprovalGateSettings,
    PaperApprovalGateSettings,
)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Single gate outcome."""

    name: str
    passed: bool
    detail: str
    tier: str  # "paper" | "live"


@dataclass(frozen=True, slots=True)
class ApprovalGateReport:
    """Aggregate gate evaluation."""

    tier: str
    allowed: bool
    results: tuple[GateResult, ...]

    @property
    def failed(self) -> tuple[GateResult, ...]:
        """Gates that did not pass."""
        return tuple(r for r in self.results if not r.passed)


def _paper_gates(
    evidence: ApprovalEvidence,
    cfg: PaperApprovalGateSettings,
) -> list[GateResult]:
    results: list[GateResult] = []

    if cfg.require_no_critical_audit_findings:
        ok = evidence.unresolved_critical_audit_findings == 0
        results.append(
            GateResult(
                name="no_unresolved_critical_audit_findings",
                passed=ok,
                detail=(
                    f"unresolved_critical={evidence.unresolved_critical_audit_findings}"
                ),
                tier="paper",
            )
        )

    ok = evidence.oos_return >= cfg.min_oos_return
    results.append(
        GateResult(
            name="positive_out_of_sample_return",
            passed=ok,
            detail=(
                f"oos_return={evidence.oos_return} min={cfg.min_oos_return} (configurable)"
            ),
            tier="paper",
        )
    )

    ok = evidence.max_drawdown <= cfg.max_drawdown
    results.append(
        GateResult(
            name="acceptable_max_drawdown",
            passed=ok,
            detail=(
                f"max_drawdown={evidence.max_drawdown} max_allowed={cfg.max_drawdown} "
                "(configurable)"
            ),
            tier="paper",
        )
    )

    ok = evidence.turnover <= cfg.max_turnover
    results.append(
        GateResult(
            name="acceptable_turnover",
            passed=ok,
            detail=(
                f"turnover={evidence.turnover} max_allowed={cfg.max_turnover} (configurable)"
            ),
            tier="paper",
        )
    )

    if cfg.require_transaction_costs_included:
        results.append(
            GateResult(
                name="transaction_costs_included",
                passed=evidence.transaction_costs_included is True,
                detail=f"transaction_costs_included={evidence.transaction_costs_included}",
                tier="paper",
            )
        )

    ok = evidence.evaluated_market_periods >= cfg.min_evaluated_market_periods
    results.append(
        GateResult(
            name="minimum_evaluated_market_periods",
            passed=ok,
            detail=(
                f"periods={evidence.evaluated_market_periods} "
                f"min={cfg.min_evaluated_market_periods} (configurable)"
            ),
            tier="paper",
        )
    )

    ok = evidence.regimes_evaluated >= cfg.min_regimes_evaluated
    results.append(
        GateResult(
            name="performance_across_multiple_regimes",
            passed=ok,
            detail=(
                f"regimes={evidence.regimes_evaluated} "
                f"min={cfg.min_regimes_evaluated} (configurable)"
            ),
            tier="paper",
        )
    )

    if cfg.require_feature_manifest_match:
        results.append(
            GateResult(
                name="no_feature_manifest_mismatch",
                passed=evidence.feature_manifest_mismatch is False,
                detail=f"feature_manifest_mismatch={evidence.feature_manifest_mismatch}",
                tier="paper",
            )
        )

    if cfg.require_model_metadata:
        results.append(
            GateResult(
                name="no_missing_model_metadata",
                passed=evidence.model_metadata_missing is False,
                detail=f"model_metadata_missing={evidence.model_metadata_missing}",
                tier="paper",
            )
        )

    if cfg.require_regression_tests_passed:
        results.append(
            GateResult(
                name="no_failed_regression_tests",
                passed=evidence.regression_tests_failed is False,
                detail=f"regression_tests_failed={evidence.regression_tests_failed}",
                tier="paper",
            )
        )

    if cfg.require_benchmark_comparison:
        ok = (
            evidence.benchmark_comparison_present
            and evidence.excess_return_vs_benchmark >= cfg.min_excess_return_vs_benchmark
        )
        results.append(
            GateResult(
                name="performance_compared_against_required_benchmarks",
                passed=ok,
                detail=(
                    f"present={evidence.benchmark_comparison_present} "
                    f"excess={evidence.excess_return_vs_benchmark} "
                    f"min_excess={cfg.min_excess_return_vs_benchmark} (configurable)"
                ),
                tier="paper",
            )
        )

    return results


def _live_gates(
    evidence: ApprovalEvidence,
    cfg: LiveApprovalGateSettings,
) -> list[GateResult]:
    results: list[GateResult] = []

    ok = evidence.paper_duration_days >= cfg.min_paper_duration_days
    results.append(
        GateResult(
            name="min_prospective_paper_duration",
            passed=ok,
            detail=(
                f"paper_days={evidence.paper_duration_days} "
                f"min={cfg.min_paper_duration_days} (configurable)"
            ),
            tier="live",
        )
    )

    ok = evidence.paper_trades >= cfg.min_paper_trades
    results.append(
        GateResult(
            name="min_paper_trades",
            passed=ok,
            detail=(
                f"paper_trades={evidence.paper_trades} "
                f"min={cfg.min_paper_trades} (configurable)"
            ),
            tier="live",
        )
    )

    ok = evidence.paper_sharpe >= cfg.min_paper_sharpe
    results.append(
        GateResult(
            name="acceptable_paper_sharpe",
            passed=ok,
            detail=(
                f"paper_sharpe={evidence.paper_sharpe} "
                f"min={cfg.min_paper_sharpe} (configurable)"
            ),
            tier="live",
        )
    )

    ok = evidence.paper_max_drawdown <= cfg.max_paper_drawdown
    results.append(
        GateResult(
            name="acceptable_paper_max_dd",
            passed=ok,
            detail=(
                f"paper_max_dd={evidence.paper_max_drawdown} "
                f"max_allowed={cfg.max_paper_drawdown} (configurable)"
            ),
            tier="live",
        )
    )

    ok = evidence.slippage_difference_bps <= cfg.max_slippage_difference_bps
    results.append(
        GateResult(
            name="acceptable_slippage_difference",
            passed=ok,
            detail=(
                f"slippage_diff_bps={evidence.slippage_difference_bps} "
                f"max={cfg.max_slippage_difference_bps} (configurable)"
            ),
            tier="live",
        )
    )

    if cfg.require_no_unresolved_reconciliation_errors:
        results.append(
            GateResult(
                name="no_unresolved_reconciliation_errors",
                passed=evidence.unresolved_reconciliation_errors == 0,
                detail=(
                    f"unresolved_reconciliation_errors="
                    f"{evidence.unresolved_reconciliation_errors}"
                ),
                tier="live",
            )
        )

    if cfg.require_no_kill_switch_investigation:
        results.append(
            GateResult(
                name="no_kill_switch_events_requiring_investigation",
                passed=evidence.kill_switch_events_requiring_investigation == 0,
                detail=(
                    "kill_switch_events_requiring_investigation="
                    f"{evidence.kill_switch_events_requiring_investigation}"
                ),
                tier="live",
            )
        )

    if cfg.require_manual_approval_record:
        results.append(
            GateResult(
                name="manual_approval_record",
                passed=evidence.manual_approval_record is True,
                detail=f"manual_approval_record={evidence.manual_approval_record}",
                tier="live",
            )
        )

    if cfg.require_explicit_live_capital_limit:
        results.append(
            GateResult(
                name="explicit_live_capital_limit",
                passed=evidence.explicit_live_capital_limit is True,
                detail=f"explicit_live_capital_limit={evidence.explicit_live_capital_limit}",
                tier="live",
            )
        )

    if cfg.require_successful_broker_sandbox_test:
        results.append(
            GateResult(
                name="successful_broker_sandbox_test",
                passed=evidence.broker_sandbox_test_passed is True,
                detail=f"broker_sandbox_test_passed={evidence.broker_sandbox_test_passed}",
                tier="live",
            )
        )

    return results


def evaluate_paper_gates(
    evidence: ApprovalEvidence,
    settings: ApprovalSettings | PaperApprovalGateSettings,
) -> ApprovalGateReport:
    """Evaluate paper-trading approval gates."""
    cfg = settings.paper if isinstance(settings, ApprovalSettings) else settings
    results = tuple(_paper_gates(evidence, cfg))
    return ApprovalGateReport(
        tier="paper",
        allowed=all(r.passed for r in results),
        results=results,
    )


def evaluate_live_gates(
    evidence: ApprovalEvidence,
    settings: ApprovalSettings,
) -> ApprovalGateReport:
    """Evaluate paper gates then additional live-tier gates."""
    paper = _paper_gates(evidence, settings.paper)
    live = _live_gates(evidence, settings.live)
    results = tuple(paper + live)
    return ApprovalGateReport(
        tier="live",
        allowed=all(r.passed for r in results),
        results=results,
    )
