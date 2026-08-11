#!/bin/bash
# Master pipeline: Step C (seg ablation, 6 runs) → Step B (3-channel baseline, 12 runs).
# Step D (per-threshold calibration diagnostic) requires manual launch.
#
# Writes status updates to saved/logs/_STATUS.txt after each step boundary
# so you can `tail -f saved/logs/_STATUS.txt` from another terminal.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

STATUS_FILE="saved/logs/_STATUS.txt"
mkdir -p saved/logs

write_status() {
  local msg="$1"
  local ts
  ts=$(date '+%F %T')
  echo "[$ts] $msg" | tee -a "$STATUS_FILE"
}

write_status "=== pipeline started ==="

# Step C: 6 seg-ablation runs on convnext_tiny × {soft,tversky,morph} × {aptos,messidor2}
write_status "starting Step C (seg ablation: 6 runs)"
if bash "$SCRIPT_DIR/run_step_c.sh" > saved/logs/step_c_master.log 2>&1; then
  write_status "Step C complete"
else
  rc=$?
  write_status "Step C FAILED rc=$rc — aborting pipeline"
  exit $rc
fi

# Step B: 12 baseline runs (3 archs × 4 folds)
write_status "starting Step B (3-channel baseline: 12 runs)"
if bash "$SCRIPT_DIR/run_step_b.sh" > saved/logs/step_b_master.log 2>&1; then
  write_status "Step B complete"
else
  rc=$?
  write_status "Step B FAILED rc=$rc — pipeline halted, see saved/logs/step_b_master.log"
  exit $rc
fi

write_status "=== pipeline complete (Steps B+C). Await command for Step D. ==="