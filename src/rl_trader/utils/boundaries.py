"""Import-boundary helpers for research vs live execution.

Hard rule
---------
Modules under the *research surface* must never import:

- ``rl_trader.brokers.live``
- ``rl_trader.execution.live``

Research surface packages:
``config``, ``data``, ``features``, ``fundamentals``, ``environment``,
``agents``, ``backtesting``, ``validation``, ``portfolio``, ``risk``,
``monitoring``, ``storage``, ``utils``.

Broker adapters live behind :mod:`rl_trader.brokers.base`. Paper adapters are
allowed from orchestration scripts; live adapters are only for live runners.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_LIVE_MODULES: frozenset[str] = frozenset(
    {
        "rl_trader.brokers.live",
        "rl_trader.execution.live",
    }
)

RESEARCH_PACKAGES: frozenset[str] = frozenset(
    {
        "config",
        "data",
        "features",
        "fundamentals",
        "environment",
        "agents",
        "backtesting",
        "validation",
        "portfolio",
        "risk",
        "monitoring",
        "storage",
        "utils",
    }
)


class LiveImportBoundaryError(ImportError):
    """Raised when research code attempts a live-module import."""


def _module_from_import(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        names.append(node.module)
    return names


def scan_file_for_forbidden_imports(path: Path) -> list[str]:
    """Return forbidden live module names imported by ``path``."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        for name in _module_from_import(node):
            if name in FORBIDDEN_LIVE_MODULES or any(
                name.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_LIVE_MODULES
            ):
                hits.append(name)
    return sorted(set(hits))


def iter_research_python_files(package_root: Path) -> list[Path]:
    """List ``.py`` files under research-surface packages."""
    files: list[Path] = []
    for package in sorted(RESEARCH_PACKAGES):
        directory = package_root / package
        if not directory.is_dir():
            continue
        files.extend(sorted(directory.rglob("*.py")))
    return files


def assert_research_has_no_live_imports(package_root: Path | None = None) -> None:
    """Fail if any research-surface module imports live brokers/execution."""
    root = package_root or Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in iter_research_python_files(root):
        hits = scan_file_for_forbidden_imports(path)
        for hit in hits:
            violations.append(f"{path.relative_to(root)} -> {hit}")
    if violations:
        joined = "; ".join(violations)
        raise LiveImportBoundaryError(
            f"research modules must not import live execution/brokers: {joined}"
        )


def guard_live_import(caller_module: str) -> None:
    """Runtime guard for live modules — reject research callers."""
    parts = caller_module.split(".")
    if len(parts) >= 2 and parts[0] == "rl_trader" and parts[1] in RESEARCH_PACKAGES:
        raise LiveImportBoundaryError(
            f"research module {caller_module!r} cannot import live execution/brokers"
        )
