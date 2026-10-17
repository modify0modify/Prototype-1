#!/usr/bin/env bash
# Point paper trading at a previously approved sealed manifest (model rollback).
# Usage: scripts/rollback_model.sh PATH/TO/manifest.json [configs/paper.yaml]
# Does not enable live trading. Manifest must already be hash-sealed + paper-approved.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${1:-}"
CONFIG="${2:-$ROOT/configs/paper.yaml}"
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "usage: $0 MANIFEST.json [CONFIG.yaml]" >&2
  exit 2
fi
ABS="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"

# Update approved_model_manifest path under live: (paper service reads this field).
python3 - <<PY
from pathlib import Path
import yaml
cfg_path = Path("$CONFIG")
doc = yaml.safe_load(cfg_path.read_text()) or {}
live = dict(doc.get("live") or {})
live["approved_model_manifest"] = "$ABS"
doc["live"] = live
# Keep paper-only safety.
doc["app_mode"] = "paper"
doc["live_trading_enabled"] = False
broker = dict(doc.get("broker") or {})
broker["environment"] = "paper"
doc["broker"] = broker
cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False))
print(f"updated {cfg_path} -> approved_model_manifest={live['approved_model_manifest']}")
PY

echo "OK model rollback configured. Restart the paper service to load the new manifest."
echo "Tip: keep prior manifests under models/approved/archive/ for future rollbacks."
