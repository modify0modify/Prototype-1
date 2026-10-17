"""Load annual financials from JSON fixtures (research path).

Point-in-time access must go through :mod:`rl_trader.fundamentals.asof`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rl_trader.fundamentals.asof import latest_statement_asof
from rl_trader.fundamentals.statements import AnnualFinancials, DataProvenance


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    """Point-in-time fundamental fields for a symbol."""

    symbol: str
    as_of: date
    fields: dict[str, Any]


def load_annual_financials_json(path: Path) -> list[AnnualFinancials]:
    """Load a list of annual statements from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "statements" not in payload:
        raise ValueError(f"expected object with 'statements' key in {path}")
    rows = payload["statements"]
    if not isinstance(rows, list):
        raise TypeError("statements must be a list")
    out: list[AnnualFinancials] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each statement must be an object")
        prov_raw = row.get("field_provenance") or {}
        provenance = {
            str(k): DataProvenance(str(v)) for k, v in dict(prov_raw).items()
        }
        data = {k: v for k, v in row.items() if k != "field_provenance"}
        out.append(AnnualFinancials(**data, field_provenance=provenance))
    return out


def load_fundamentals(symbol: str, as_of: date, *, path: Path | None = None) -> FundamentalSnapshot:
    """Load the latest known statement fields as of ``as_of``.

    If ``path`` is omitted, raises ``FileNotFoundError`` — callers should pass
    an explicit fixture/dataset path. Never returns future publications.
    """
    if path is None:
        raise FileNotFoundError(
            "load_fundamentals requires an explicit JSON path "
            "(e.g. tests/fixtures/fundamentals/sample_annual.json)"
        )
    statements = [s for s in load_annual_financials_json(path) if s.symbol == symbol.upper()]
    latest = latest_statement_asof(statements, as_of)
    if latest is None:
        return FundamentalSnapshot(symbol=symbol.upper(), as_of=as_of, fields={})
    fields = latest.model_dump(mode="json")
    return FundamentalSnapshot(symbol=symbol.upper(), as_of=as_of, fields=fields)
