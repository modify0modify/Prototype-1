"""Test helpers for sealed approval manifests (not for production use)."""

from __future__ import annotations

from pathlib import Path

from rl_trader.paper_trading.demo_model import provision_demo_approved_model


def write_sealed_approved_manifest(
    root: Path,
    *,
    tier: str = "paper",
    model_id: str = "unit-test-model",
    n_assets: int = 2,
) -> Path:
    """Write a hash-sealed manifest that passes example gate defaults for ``tier``."""
    if tier not in {"paper", "live"}:
        raise ValueError("tier must be 'paper' or 'live'")
    return provision_demo_approved_model(
        root,
        model_id=model_id,
        n_assets=n_assets,
        approval_status=tier,
    )


def write_sealed_unapproved_manifest(root: Path, *, model_id: str = "denied-model") -> Path:
    """Write a sealed manifest with approval_status=rejected (fails paper/live entry)."""
    from rl_trader.approval.evidence import example_passing_paper_evidence
    from rl_trader.approval.manifest import (
        Period,
        build_manifest_payload,
        seal_manifest,
        write_manifest_file,
    )
    from rl_trader.agents.checkpoint import (
        CheckpointMetadata,
        TrainingPeriod,
        save_checkpoint_metadata,
    )
    from rl_trader.features.engineering import FeatureSpec
    from datetime import date

    root = Path(root)
    ckpt = root / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    specs = (FeatureSpec(name="returns_1d", lookback=1),)
    save_checkpoint_metadata(
        ckpt,
        CheckpointMetadata(
            model_id=model_id,
            algorithm="stub",
            feature_specs=specs,
            training_period=TrainingPeriod(start=date(2023, 1, 3), end=date(2023, 6, 30)),
            seed=0,
            package_version="0",
        ),
    )
    fm = {
        "feature_specs": [
            {"name": "returns_1d", "lookback": 1, "version": "1", "causal_only": True}
        ]
    }
    payload = build_manifest_payload(
        model_id=model_id,
        model_version="0",
        algorithm="stub",
        hyperparameters={},
        train_period=Period("2023-01-03", "2023-06-30"),
        val_period=Period("2023-07-01", "2023-09-30"),
        test_period=Period("2023-10-01", "2023-12-29"),
        feature_manifest=fm,
        benchmark_results={},
        transaction_cost_assumptions={"commission_bps": 1.0},
        approval_status="rejected",
        approver="nobody",
        code_commit_id="none",
        checkpoint_dir=ckpt,
        evaluation_evidence=example_passing_paper_evidence().to_dict(),
        n_assets=2,
    )
    return write_manifest_file(root / "manifest.json", seal_manifest(payload))
