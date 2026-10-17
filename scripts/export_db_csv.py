#!/usr/bin/env python3
"""Export trading database tables to CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl_trader.storage import create_db_engine, export_all_csv, upgrade_head


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "sqlite:///./reports/rl_trader.db"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/db_exports"),
    )
    parser.add_argument(
        "--skip-migrate",
        action="store_true",
        help="Do not run alembic upgrade before export",
    )
    args = parser.parse_args(argv)
    if not args.skip_migrate:
        upgrade_head(args.database_url)
    engine = create_db_engine(args.database_url, create_schema=False)
    counts = export_all_csv(engine, args.output_dir)
    for name, n in sorted(counts.items()):
        print(f"{name}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
