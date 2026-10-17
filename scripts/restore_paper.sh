#!/usr/bin/env bash
# Restore paper-trading artifacts from a backup directory created by backup_paper.sh.
# Usage: scripts/restore_paper.sh BACKUP_DIR
# Does NOT restore .env secrets. Does NOT enable live trading.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "usage: $0 BACKUP_DIR" >&2
  exit 2
fi
if [[ ! -f "$SRC/BACKUP_TIMESTAMP" ]]; then
  echo "refusing: $SRC does not look like a paper backup (missing BACKUP_TIMESTAMP)" >&2
  exit 2
fi

echo "Restoring from $SRC into $ROOT"
mkdir -p "$ROOT/reports" "$ROOT/data/interim" "$ROOT/models" "$ROOT/configs"

if [[ -d "$SRC/configs" ]]; then
  cp -a "$SRC/configs/." "$ROOT/configs/"
fi
if [[ -d "$SRC/reports/paper_trading" ]]; then
  rm -rf "$ROOT/reports/paper_trading"
  cp -a "$SRC/reports/paper_trading" "$ROOT/reports/"
fi
if [[ -f "$SRC/reports/rl_trader.db" ]]; then
  cp -a "$SRC/reports/rl_trader.db" "$ROOT/reports/"
fi
if [[ -d "$SRC/data/interim/paper_broker" ]]; then
  rm -rf "$ROOT/data/interim/paper_broker"
  mkdir -p "$ROOT/data/interim"
  cp -a "$SRC/data/interim/paper_broker" "$ROOT/data/interim/"
fi
if [[ -d "$SRC/models/approved" ]]; then
  rm -rf "$ROOT/models/approved"
  cp -a "$SRC/models/approved" "$ROOT/models/"
fi
if [[ -d "$SRC/models/checkpoints" ]]; then
  rm -rf "$ROOT/models/checkpoints"
  cp -a "$SRC/models/checkpoints" "$ROOT/models/"
fi

echo "OK restore complete from $(cat "$SRC/BACKUP_TIMESTAMP")"
echo "Remember: re-create .env from .env.example (secrets are never in backups)."
