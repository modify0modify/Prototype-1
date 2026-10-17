#!/usr/bin/env python3
"""Paper-service health check (heartbeat freshness + paper-mode guards).

Exit 0 when healthy, 1 when unhealthy. Suitable for Docker HEALTHCHECK.
Never enables live trading.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heartbeat",
        type=Path,
        default=Path("reports/paper_trading/heartbeat.json"),
        help="Path to heartbeat.json",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=300.0,
        help="Fail if heartbeat older than this (default 300)",
    )
    parser.add_argument(
        "--allow-starting",
        action="store_true",
        help="Treat status=starting as healthy (boot window)",
    )
    parser.add_argument(
        "--allow-stopped",
        action="store_true",
        help="Treat status=stopped as healthy (one-shot cycles)",
    )
    args = parser.parse_args(argv)

    path: Path = args.heartbeat
    if not path.is_file():
        print(f"UNHEALTHY: heartbeat missing: {path}", file=sys.stderr)
        return 1
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"UNHEALTHY: heartbeat unreadable: {exc}", file=sys.stderr)
        return 1
    if not isinstance(doc, dict):
        print("UNHEALTHY: heartbeat root must be object", file=sys.stderr)
        return 1

    status = str(doc.get("status") or "")
    ts_raw = doc.get("timestamp")
    if not ts_raw:
        print("UNHEALTHY: heartbeat missing timestamp", file=sys.stderr)
        return 1
    try:
        ts = datetime.fromisoformat(str(ts_raw))
    except ValueError:
        print(f"UNHEALTHY: bad timestamp {ts_raw!r}", file=sys.stderr)
        return 1
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - ts.astimezone(UTC)).total_seconds()
    if age > float(args.max_age_seconds):
        print(
            f"UNHEALTHY: heartbeat stale age={age:.1f}s max={args.max_age_seconds}",
            file=sys.stderr,
        )
        return 1

    unhealthy_statuses = {"shutdown_stale_data", "error"}
    if status in unhealthy_statuses:
        print(f"UNHEALTHY: status={status}", file=sys.stderr)
        return 1
    # Looping deploys should keep writing cycle_running / starting beats.
    # One-shot jobs end in stopped — pass --allow-stopped for those.
    if status == "stopped" and not args.allow_stopped:
        print("UNHEALTHY: status=stopped (service not looping)", file=sys.stderr)
        return 1
    if status == "starting" and not args.allow_starting:
        print("UNHEALTHY: still starting", file=sys.stderr)
        return 1

    print(f"OK status={status} age_s={age:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
