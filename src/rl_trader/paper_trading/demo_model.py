"""Provision a local stub approved model for ``--demo`` / fixture smoke runs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    save_checkpoint_metadata,
)
from rl_trader.approval.evidence import example_passing_paper_evidence
from rl_trader.approval.manifest import (
    Period,
    build_manifest_payload,
    seal_manifest,
    write_manifest_file,
)
from rl_trader.features.engineering import FeatureSpec


def provision_demo_approved_model(
    root: Path,
    *,
    model_id: str = "paper-demo-stub",
    n_assets: int = 2,
    evaluation_period_id: str = "paper-default",
    approval_status: str = "paper",
) -> Path:
    """Write checkpoint + sealed hashed manifest; return manifest path."""
    root = Path(root)
    ckpt = root / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    specs = (
        FeatureSpec(name="returns_1d", lookback=1, version="1", causal_only=True),
        FeatureSpec(name="vol_20d", lookback=20, version="1", causal_only=True),
    )
    meta = CheckpointMetadata(
        model_id=model_id,
        algorithm="stub",
        feature_specs=specs,
        training_period=TrainingPeriod(start=date(2023, 1, 3), end=date(2023, 6, 30)),
        seed=42,
        package_version="0.1.0-demo",
        notes="Auto-provisioned stub for paper --demo; not a production approval.",
    )
    save_checkpoint_metadata(ckpt, meta)
    feature_manifest = {
        "feature_specs": [
            {
                "name": s.name,
                "lookback": s.lookback,
                "version": s.version,
                "causal_only": s.causal_only,
            }
            for s in specs
        ]
    }
    (ckpt / "feature_manifest.json").write_text(
        __import__("json").dumps(feature_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evidence = example_passing_paper_evidence()
    if approval_status == "live":
        from rl_trader.approval.evidence import example_passing_live_evidence

        evidence = example_passing_live_evidence()

    payload = build_manifest_payload(
        model_id=model_id,
        model_version="0.1.0-demo",
        algorithm="stub",
        hyperparameters={"algorithm": "stub", "n_assets": n_assets},
        train_period=Period("2023-01-03", "2023-06-30"),
        val_period=Period("2023-07-01", "2023-09-30"),
        test_period=Period("2023-10-01", "2023-12-29"),
        feature_manifest=feature_manifest,
        benchmark_results={"buy_hold": {"total_return": 0.01}},
        transaction_cost_assumptions={
            "commission_bps": 1.0,
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
        },
        approval_status=approval_status,  # type: ignore[arg-type]
        approver="demo-provisioner",
        code_commit_id="demo-uncommitted",
        checkpoint_dir=ckpt,
        evaluation_evidence=evidence.to_dict(),
        n_assets=n_assets,
        notes=(
            "Demo stub approval with EXAMPLE evidence that satisfies example "
            "gate defaults — replace with a real sealed approval for production paper."
        ),
        extra={"evaluation_period_id": evaluation_period_id},
    )
    sealed = seal_manifest(payload, sign=False)
    return write_manifest_file(root / "manifest.json", sealed)
