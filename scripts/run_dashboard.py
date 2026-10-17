#!/usr/bin/env python3
"""Launch the Streamlit monitoring dashboard (read-only).

Does not enable live trading. Binds to an uncommon port by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8505,
        help="HTTP port (default 8505; avoid common 8501 collisions)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open a browser window (default: headless / no browser)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    app = root / "src" / "rl_trader" / "monitoring" / "dashboard" / "app.py"
    if not app.is_file():
        print(f"dashboard app missing: {app}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    if not args.browser:
        cmd.extend(["--server.headless", "true"])

    print(
        "Starting RL Trader monitoring dashboard (SIMULATED / read-only; "
        "live activation disabled)"
    )
    print(f"URL: http://{args.host}:{args.port}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
