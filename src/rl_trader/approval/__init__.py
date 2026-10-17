"""Formal model-approval system (hashed manifests + configurable gates)."""

from rl_trader.approval.evidence import (
    ApprovalEvidence,
    example_passing_live_evidence,
    example_passing_paper_evidence,
)
from rl_trader.approval.gates import (
    ApprovalGateReport,
    GateResult,
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
    VerifiedApproval,
    verify_live_approval,
    verify_paper_approval,
)

__all__ = [
    "REQUIRED_MANIFEST_FIELDS",
    "ApprovalEvidence",
    "ApprovalGateReport",
    "ApprovalRejectedError",
    "GateResult",
    "ManifestError",
    "Period",
    "VerifiedApproval",
    "build_manifest_payload",
    "compute_content_hash",
    "evaluate_live_gates",
    "evaluate_paper_gates",
    "example_passing_live_evidence",
    "example_passing_paper_evidence",
    "seal_manifest",
    "verify_live_approval",
    "verify_manifest",
    "verify_paper_approval",
    "write_manifest_file",
]
