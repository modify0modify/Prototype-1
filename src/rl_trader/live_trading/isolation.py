"""Agent isolation — RL agents must never modify protected live controls.

Architecture
------------
- Protected controls are loaded from human config into
  :class:`ProtectedLiveControls` (frozen).
- Live paths ignore any agent-supplied control overrides via
  :func:`ignore_agent_control_attempt`.
- Static scan :func:`assert_agents_cannot_write_controls` fails if agent
  modules assign to forbidden attribute names.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AGENT_FORBIDDEN_CONTROL_ATTRS: frozenset[str] = frozenset(
    {
        "risk_limits",
        "max_live_capital",
        "max_order_value",
        "daily_loss_limit",
        "max_drawdown_limit",
        "approval_status",
        "live_api_key",
        "live_api_secret",
        "paper_api_key",
        "paper_api_secret",
        "app_mode",
        "live_trading_enabled",
        "kill_switch",
        "manual_activated",
        "rollout_stage",
        "broker_environment",
        "acknowledgement",
    }
)


class AgentControlIsolationError(RuntimeError):
    """Raised when agent code attempts to write protected controls."""


@dataclass(frozen=True, slots=True)
class ProtectedLiveControls:
    """Frozen controls visible to the live path (agent cannot mutate)."""

    app_mode: str
    live_trading_enabled: bool
    broker_environment: str
    max_live_capital: float | None
    max_order_value: float | None
    daily_loss_limit: float | None
    max_drawdown_limit: float
    approval_status_required: str
    kill_switch_active: bool
    manual_activated: bool
    rollout_stage: int
    account_id: str


def ignore_agent_control_attempt(
    controls: ProtectedLiveControls,
    agent_overrides: dict[str, Any] | None,
) -> ProtectedLiveControls:
    """Return ``controls`` unchanged — agent overrides are discarded."""
    _ = agent_overrides  # explicitly ignored
    return controls


def _assignment_targets(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Attribute):
        names.append(node.attr)
    elif isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Tuple | ast.List):
        for elt in node.elts:
            names.extend(_assignment_targets(elt))
    return names


def scan_agent_file_for_control_writes(path: Path) -> list[str]:
    """Return forbidden control attribute names written in an agent module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            targets.append(node.target)
        for target in targets:
            for name in _assignment_targets(target):
                if name in AGENT_FORBIDDEN_CONTROL_ATTRS:
                    hits.append(name)
    return sorted(set(hits))


def assert_agents_cannot_write_controls(package_root: Path | None = None) -> None:
    """Fail if any ``rl_trader.agents`` module writes protected control attrs."""
    root = package_root or Path(__file__).resolve().parents[1]
    agents_dir = root / "agents"
    if not agents_dir.is_dir():
        return
    violations: list[str] = []
    for path in sorted(agents_dir.rglob("*.py")):
        for hit in scan_agent_file_for_control_writes(path):
            violations.append(f"{path.relative_to(root)} writes {hit}")
    if violations:
        raise AgentControlIsolationError(
            "agent modules must not write protected live controls: "
            + "; ".join(violations)
        )
