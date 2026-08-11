#!/bin/bash
# Recover from a failed run: identify the failing step + run, halve its batch size,
# re-launch that single run, then resume the pipeline.
#
# Used by `run_pipeline_v2.sh` (via trap) when a run fails. Tries the
# simplest fixes first (halve batch_size in the active config), then a
# fall-back (skip the run with a stub checkpoint if the user has authorized
# skipping — controlled by env var SKIP_ALLOWED=1).
#
# Halve-bs is persistent: the model config is rewritten, and we restore it
# after the run completes.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

STATUS_FILE="saved/logs/_STATUS.txt"
TRAIN_LOG="${1:-}"
MODEL_CFG="${2:-}"

write_status() {
  local ts; ts=$(date '+%F %T')
  echo "[$ts] RECOVER: $1" | tee -a "$STATUS_FILE"
}

if [ -z "$TRAIN_LOG" ] || [ -z "$MODEL_CFG" ]; then
  write_status "ERROR: recover_step.sh invoked without TRAIN_LOG/MODEL_CFG"
  exit 1
fi

# Check the latest log lines for OOM signature
if grep -qE "out of memory|CUDA out of memory|RuntimeError: CUDA" "$TRAIN_LOG" 2>/dev/null; then
  write_status "OOM detected in $(basename "$TRAIN_LOG")"
else
  # Not OOM — don't pretend to fix it
  write_status "non-OOM failure in $(basename "$TRAIN_LOG"); manual intervention required"
  exit 2
fi

# Identify phase1 and phase2 batch sizes, halve them
if [ ! -f "$MODEL_CFG" ]; then
  write_status "ERROR: model cfg not found: $MODEL_CFG"
  exit 1
fi

cp "$MODEL_CFG" "$MODEL_CFG.bak"

# Use python (via _pvrun) to safely halve batch_size for both phases
bash "$SCRIPT_DIR/_pvrun.sh" -c "
import yaml, sys
p = '$MODEL_CFG'
with open(p) as f:
    data = yaml.safe_load(f)
ph = data.get('phases', {})
for phase_name, phase_cfg in ph.items():
    if 'batch_size' in phase_cfg:
        old = phase_cfg['batch_size']
        new = max(1, old // 2)
        phase_cfg['batch_size'] = new
        print(f'{phase_name}: batch_size {old} -> {new}', file=sys.stderr)
with open(p, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
print('OK')
"

write_status "halved batch sizes in $(basename "$MODEL_CFG") (backup at $(basename "$MODEL_CFG").bak)"

echo "$MODEL_CFG" > saved/logs/_last_halved_cfg