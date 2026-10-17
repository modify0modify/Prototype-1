"""Unit tests for formal model-approval gates, hashing, and verification."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rl_trader.approval.evidence import (
    ApprovalEvidence,
    example_passing_live_evidence,
    example_passing_paper_evidence,
)
from rl_trader.approval.gates import (
    evaluate_live_gates,
    evaluate_paper_gates,
)
from rl_trader.approval.manifest import (
    REQUIRED_MANIFEST_FIELDS,
    ManifestError,
    Period,
    build_manifest_payload,
    compute_content_hash,
    seal_manifest,
    verify_manifest,
    write_manifest_file,
)
from rl_trader.approval.service import (
    ApprovalRejectedError,
    verify_live_approval,
    verify_paper_approval,
)
from rl_trader.approval.store import persist_approval
from rl_trader.config.models import (
    ApprovalSettings,
    LiveApprovalGateSettings,
    PaperApprovalGateSettings,
)
from rl_trader.storage import (
    ModelApprovalRow,
    create_db_engine,
    session_scope,
    upgrade_head,
)


def _base_payload(**overrides: object) -> dict:
    evidence = example_passing_paper_evidence().to_dict()
    payload = build_manifest_payload(
        model_id="gate-test",
        model_version="1",
        algorithm="stub",
        hyperparameters={},
        train_period=Period("2023-01-03", "2023-06-30"),
        val_period=Period("2023-07-01", "2023-09-30"),
        test_period=Period("2023-10-01", "2023-12-29"),
        feature_manifest={
            "feature_specs": [
                {"name": "returns_1d", "lookback": 1, "version": "1", "causal_only": True}
            ]
        },
        benchmark_results={"buy_hold": {"total_return": 0.01}},
        transaction_cost_assumptions={"commission_bps": 1.0},
        approval_status="paper",
        approver="tester",
        code_commit_id="abc123",
        checkpoint_dir="/tmp/ckpt",
        evaluation_evidence=evidence,
        n_assets=2,
    )
    payload.update(overrides)
    return payload


def test_example_defaults_labelled_as_examples() -> None:
    """Config field defaults must remain EXAMPLE illustrations (docstrings)."""
    paper_doc = PaperApprovalGateSettings.__doc__ or ""
    live_doc = LiveApprovalGateSettings.__doc__ or ""
    assert "EXAMPLE" in paper_doc
    assert "EXAMPLE" in live_doc
    settings = ApprovalSettings()
    # Sanity: example defaults are conservative (not investment advice).
    assert settings.paper.min_oos_return == 0.0
    assert settings.paper.max_drawdown == 0.25
    assert settings.live.min_paper_duration_days == 30.0
    assert settings.require_manifest_signature is False


# ---------------------------------------------------------------------------
# Manifest hash / signature / incompleteness
# ---------------------------------------------------------------------------


def test_seal_and_verify_content_hash() -> None:
    sealed = seal_manifest(_base_payload())
    assert len(sealed["content_hash"]) == 64
    assert verify_manifest(sealed) == sealed["content_hash"]
    assert compute_content_hash(sealed) == sealed["content_hash"]


def test_tampered_manifest_rejected() -> None:
    sealed = seal_manifest(_base_payload())
    sealed["approver"] = "attacker"
    with pytest.raises(ManifestError, match="content_hash mismatch"):
        verify_manifest(sealed)


def test_incomplete_manifest_rejected() -> None:
    payload = _base_payload()
    del payload["approver"]
    with pytest.raises(ManifestError, match="incomplete manifest"):
        seal_manifest(payload)
    for field in REQUIRED_MANIFEST_FIELDS:
        broken = _base_payload()
        del broken[field]
        with pytest.raises(ManifestError, match="incomplete"):
            seal_manifest(broken)


def test_missing_content_hash_rejected() -> None:
    sealed = seal_manifest(_base_payload())
    sealed.pop("content_hash")
    with pytest.raises(ManifestError, match="missing content_hash"):
        verify_manifest(sealed)


def test_hmac_signature_required_and_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    key = b"unit-test-signing-key"
    monkeypatch.setenv("RL_TRADER_MANIFEST_SIGNING_KEY", key.decode())
    sealed = seal_manifest(_base_payload(), sign=True)
    assert "signature" in sealed
    assert verify_manifest(sealed, require_signature=True) == sealed["content_hash"]
    sealed["signature"] = "0" * 64
    with pytest.raises(ManifestError, match="signature invalid"):
        verify_manifest(sealed, require_signature=True)
    sealed2 = seal_manifest(_base_payload(), sign=False)
    with pytest.raises(ManifestError, match="signature required"):
        verify_manifest(sealed2, require_signature=True)


# ---------------------------------------------------------------------------
# Paper gates — one failure case each
# ---------------------------------------------------------------------------


PAPER_GATE_FAILURES: list[tuple[str, dict]] = [
    (
        "no_unresolved_critical_audit_findings",
        {"unresolved_critical_audit_findings": 1},
    ),
    ("positive_out_of_sample_return", {"oos_return": -0.01}),
    ("acceptable_max_drawdown", {"max_drawdown": 0.50}),
    ("acceptable_turnover", {"turnover": 5.0}),
    ("transaction_costs_included", {"transaction_costs_included": False}),
    ("minimum_evaluated_market_periods", {"evaluated_market_periods": 1}),
    ("performance_across_multiple_regimes", {"regimes_evaluated": 1}),
    ("no_feature_manifest_mismatch", {"feature_manifest_mismatch": True}),
    ("no_missing_model_metadata", {"model_metadata_missing": True}),
    ("no_failed_regression_tests", {"regression_tests_failed": True}),
    (
        "performance_compared_against_required_benchmarks",
        {"benchmark_comparison_present": False},
    ),
]


@pytest.mark.parametrize(("gate_name", "overrides"), PAPER_GATE_FAILURES)
def test_paper_gate_fails_individually(gate_name: str, overrides: dict) -> None:
    evidence = example_passing_paper_evidence()
    for key, value in overrides.items():
        setattr(evidence, key, value)
    report = evaluate_paper_gates(evidence, ApprovalSettings())
    assert report.allowed is False
    failed_names = {r.name for r in report.failed}
    assert gate_name in failed_names


def test_paper_gates_pass_with_example_evidence() -> None:
    report = evaluate_paper_gates(example_passing_paper_evidence(), ApprovalSettings())
    assert report.allowed is True
    assert all(r.passed for r in report.results)
    assert len(report.results) == len(PAPER_GATE_FAILURES)


def test_paper_thresholds_are_configurable() -> None:
    """Raising min_oos_return above evidence must fail; lowering must pass."""
    evidence = example_passing_paper_evidence()
    settings = ApprovalSettings(
        paper=PaperApprovalGateSettings(min_oos_return=evidence.oos_return + 0.01)
    )
    assert evaluate_paper_gates(evidence, settings).allowed is False
    settings_ok = ApprovalSettings(
        paper=PaperApprovalGateSettings(min_oos_return=evidence.oos_return - 0.01)
    )
    assert evaluate_paper_gates(evidence, settings_ok).allowed is True


# ---------------------------------------------------------------------------
# Live gates — one failure case each (paper evidence still passes)
# ---------------------------------------------------------------------------


LIVE_GATE_FAILURES: list[tuple[str, dict]] = [
    ("min_prospective_paper_duration", {"paper_duration_days": 5.0}),
    ("min_paper_trades", {"paper_trades": 1}),
    ("acceptable_paper_sharpe", {"paper_sharpe": 0.1}),
    ("acceptable_paper_max_dd", {"paper_max_drawdown": 0.50}),
    ("acceptable_slippage_difference", {"slippage_difference_bps": 20.0}),
    ("no_unresolved_reconciliation_errors", {"unresolved_reconciliation_errors": 2}),
    (
        "no_kill_switch_events_requiring_investigation",
        {"kill_switch_events_requiring_investigation": 1},
    ),
    ("manual_approval_record", {"manual_approval_record": False}),
    ("explicit_live_capital_limit", {"explicit_live_capital_limit": False}),
    ("successful_broker_sandbox_test", {"broker_sandbox_test_passed": False}),
]


@pytest.mark.parametrize(("gate_name", "overrides"), LIVE_GATE_FAILURES)
def test_live_gate_fails_individually(gate_name: str, overrides: dict) -> None:
    evidence = example_passing_live_evidence()
    for key, value in overrides.items():
        setattr(evidence, key, value)
    report = evaluate_live_gates(evidence, ApprovalSettings())
    assert report.allowed is False
    failed_names = {r.name for r in report.failed}
    assert gate_name in failed_names


def test_live_gates_pass_with_example_evidence() -> None:
    report = evaluate_live_gates(example_passing_live_evidence(), ApprovalSettings())
    assert report.allowed is True
    # Live evaluation includes paper + live gates.
    assert len(report.results) == len(PAPER_GATE_FAILURES) + len(LIVE_GATE_FAILURES)


def test_live_requires_paper_gates_too() -> None:
    evidence = example_passing_live_evidence()
    evidence.oos_return = -1.0
    report = evaluate_live_gates(evidence, ApprovalSettings())
    assert report.allowed is False
    assert "positive_out_of_sample_return" in {r.name for r in report.failed}


# ---------------------------------------------------------------------------
# Service verification (file + status + gates)
# ---------------------------------------------------------------------------


def _write_sealed(
    tmp_path: Path,
    *,
    status: str = "paper",
    evidence: ApprovalEvidence | None = None,
    tamper: bool = False,
) -> Path:
    ev = evidence or (
        example_passing_live_evidence()
        if status == "live"
        else example_passing_paper_evidence()
    )
    payload = _base_payload(
        approval_status=status,
        evaluation_evidence=ev.to_dict(),
    )
    sealed = seal_manifest(payload)
    if tamper:
        sealed = copy.deepcopy(sealed)
        sealed["model_id"] = "tampered"
        # leave stale content_hash
    path = tmp_path / f"manifest-{status}.json"
    write_manifest_file(path, sealed)
    return path


def test_verify_paper_approval_accepts_paper_and_live(tmp_path: Path) -> None:
    settings = ApprovalSettings()
    paper = verify_paper_approval(_write_sealed(tmp_path / "p", status="paper"), settings)
    assert paper.tier == "paper"
    live = verify_paper_approval(_write_sealed(tmp_path / "l", status="live"), settings)
    assert live.content_hash


def test_verify_paper_rejects_unapproved_status(tmp_path: Path) -> None:
    path = _write_sealed(tmp_path, status="rejected")
    with pytest.raises(ApprovalRejectedError, match="not paper/live"):
        verify_paper_approval(path, ApprovalSettings())


def test_verify_live_requires_live_status(tmp_path: Path) -> None:
    paper_path = _write_sealed(tmp_path / "p", status="paper")
    with pytest.raises(ApprovalRejectedError, match="approval_status='live'"):
        verify_live_approval(paper_path, ApprovalSettings())
    live_path = _write_sealed(tmp_path / "l", status="live")
    verified = verify_live_approval(live_path, ApprovalSettings())
    assert verified.tier == "live"


def test_verify_rejects_tampered_file(tmp_path: Path) -> None:
    path = _write_sealed(tmp_path, status="paper", tamper=True)
    with pytest.raises(ApprovalRejectedError, match="content_hash mismatch"):
        verify_paper_approval(path, ApprovalSettings())


def test_verify_rejects_gate_failures(tmp_path: Path) -> None:
    bad = example_passing_paper_evidence()
    bad.oos_return = -0.5
    path = _write_sealed(tmp_path, status="paper", evidence=bad)
    with pytest.raises(ApprovalRejectedError, match="paper approval gates failed"):
        verify_paper_approval(path, ApprovalSettings())


def test_persist_approval_row(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'a.db'}"
    upgrade_head(db_url)
    engine = create_db_engine(db_url, create_schema=False)
    row_id = persist_approval(
        engine,
        model_id="persist-me",
        approval_tier="paper",
        approved=True,
        approved_by="tester",
        content_hash="deadbeef",
        code_commit_id="cafebabe",
        manifest_path="/tmp/m.json",
        evidence=example_passing_paper_evidence().to_dict(),
    )
    with session_scope(engine) as session:
        row = session.get(ModelApprovalRow, row_id)
        assert row is not None
        assert row.approval_tier == "paper"
        assert row.content_hash == "deadbeef"
        assert row.code_commit_id == "cafebabe"
        assert "oos_return" in row.evidence_json
