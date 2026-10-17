#!/usr/bin/env bash
# Backup paper-trading state, SQLite DB, configs, and approved model manifests.
# Usage: scripts/backup_paper.sh [BACKUP_ROOT]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_ROOT="${1:-$ROOT/backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$BACKUP_ROOT/paper_$STAMP"
mkdir -p "$DEST"

echo "Backing up to $DEST"

# Configs (no secrets — .env is NOT copied)
mkdir -p "$DEST/configs"
cp -a "$ROOT/configs/." "$DEST/configs/" 2>/dev/null || true

# Paper state / ledgers / reports
mkdir -p "$DEST/reports" "$DEST/data"
if [[ -d "$ROOT/reports/paper_trading" ]]; then
  cp -a "$ROOT/reports/paper_trading" "$DEST/reports/"
fi
if [[ -f "$ROOT/reports/rl_trader.db" ]]; then
  cp -a "$ROOT/reports/rl_trader.db" "$DEST/reports/"
fi
if [[ -d "$ROOT/data/interim/paper_broker" ]]; then
  mkdir -p "$DEST/data/interim"
  cp -a "$ROOT/data/interim/paper_broker" "$DEST/data/interim/"
fi
if [[ -d "$ROOT/models/approved" ]]; then
  mkdir -p "$DEST/models"
  cp -a "$ROOT/models/approved" "$DEST/models/"
fi
if [[ -d "$ROOT/models/checkpoints" ]]; then
  mkdir -p "$DEST/models"
  cp -a "$ROOT/models/checkpoints" "$DEST/models/" || true
fi

# Manifest of what was copied (for restore verification)
find "$DEST" -type f | sort > "$DEST/MANIFEST.txt"
echo "$STAMP" > "$DEST/BACKUP_TIMESTAMP"
echo "OK backup=$DEST"
echo "$DEST"
