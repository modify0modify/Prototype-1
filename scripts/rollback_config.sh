#!/usr/bin/env bash
# Restore a YAML config from a backup configs/ tree (configuration rollback).
# Usage: scripts/rollback_config.sh BACKUP_DIR/configs/paper.yaml [DEST]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
DEST="${2:-$ROOT/configs/paper.yaml}"
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
  echo "usage: $0 BACKUP_CONFIG.yaml [DEST.yaml]" >&2
  exit 2
fi

# Refuse accidental live enablement in the restored file.
if grep -Eiq '^\s*live_trading_enabled:\s*true' "$SRC"; then
  echo "refusing: source config has live_trading_enabled: true" >&2
  exit 3
fi
if grep -Eiq '^\s*app_mode:\s*live' "$SRC"; then
  echo "refusing: source config has app_mode: live (paper deploy only)" >&2
  exit 3
fi

cp -a "$DEST" "${DEST}.bak.$(date -u +%Y%m%dT%H%M%SZ)" 2>/dev/null || true
cp -a "$SRC" "$DEST"
echo "OK config rollback: $SRC -> $DEST (previous copied to .bak.* if present)"
