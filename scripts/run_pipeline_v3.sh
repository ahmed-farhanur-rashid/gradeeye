#!/bin/bash
# Master pipeline v3: Steps C → B → F → E.
#  - Step C: 6 seg-ablation runs on convnext_tiny × {soft,tversky,morph} × {aptos,messidor2}
#  - Step B: 12 baseline runs (3 archs × 4 folds, balanced)
#  - Step F: 6 seg-ablation runs completing eyepacs+ddr for the 3 variants (4-fold matrix)
#  - Step E: 4 unbalanced-baseline runs on convnext_tiny × 4 folds (for §6 class-balancing ablation)
# Step D (per-threshold calibration) still requires manual launch.
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

write_status "=== pipeline_v3 started ==="

run_step() {
  local name="$1"
  local script="$2"
  local runs="$3"
  write_status "starting $name ($runs)"
  if bash "$SCRIPT_DIR/$script" > "saved/logs/${name,,}_master.log" 2>&1; then
    write_status "$name complete"
    return 0
  fi
  local rc=$?
  write_status "$name FAILED rc=$rc — pipeline halted, see saved/logs/${name,,}_master.log"
  return $rc
}

# Order:
#  Step C → Step B (must happen first; establishes 3-channel baseline reference)
#  Step F (4ch on eyepacs+ddr; enables full 4-fold seg ablation matrix)
#  Step E (unbalanced baseline; needed for class-balancing ablation §6)
run_step "Step C" run_step_c.sh "seg ablation: 6 runs" || exit $?
run_step "Step B" run_step_b.sh "3-channel baseline: 12 runs" || exit $?
run_step "Step F" run_step_f.sh "4ch ablation eyepacs+ddr: 6 runs" || exit $?
run_step "Step E" run_step_e.sh "unbalanced baseline: 4 runs" || exit $?

write_status "=== pipeline_v3 complete (Steps B/C/E/F). Await command for Step D. ==="
