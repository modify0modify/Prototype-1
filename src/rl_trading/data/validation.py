"""Market-data validation stubs.

Look-ahead bias note
--------------------
Validation must not peek at future labels or adjusted prices that revise after
the decision point. Calendar alignment should use only information available
as-of each timestamp.

Missing-data policy
-------------------
Missing values, gaps, and timezone inconsistencies are hard failures unless an
explicit, logged remediation step is recorded. No silent fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ValidationReport:
    """Outcome of a dataset validation pass."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_ohlcv_frame(frame: Any) -> ValidationReport:
    """Validate an OHLCV-like frame (stub).

    Real implementation will check schema, monotonic timestamps, no duplicate
    indices, missing bars, and corporate-action consistency — without
    silent imputation.
    """
    if frame is None:
        return ValidationReport(ok=False, errors=["frame is None"])
    return ValidationReport(
        ok=False,
        errors=["validate_ohlcv_frame is a stub; implement schema checks"],
    )
