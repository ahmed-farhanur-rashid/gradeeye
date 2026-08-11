#!/bin/bash
# Step D: per-threshold calibration diagnostic on the 3-channel ConvNeXt-Tiny checkpoints.
# Runs `python -m src.eval.run_calibration_study` with appropriate flags for our
# baseline_3ch/<fold>/convnext_tiny/<run>_best.pt checkpoint layout.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYRUN="$SCRIPT_DIR/_pvrun.sh"

# Symlink the convnext_tiny baseline checkpoints into the layout expected by the
# calibration runner (`saved/checkpoints/<fold>/<arch>/lodo_<fold>_<arch>_best.pt`).
# This avoids modifying the runner or copying checkpoints.
TMP_ROOT="saved/checkpoints/calibration_layout"
mkdir -p "$TMP_ROOT"

for fold in eyepacs aptos messidor2 ddr ; do
  mkdir -p "$TMP_ROOT/$fold/convnext_tiny"
  if [ -f "saved/checkpoints/baseline_3ch/$fold/convnext_tiny/lodo_${fold}_convnext_tiny_best.pt" ]; then
    ln -sf "$(pwd)/saved/checkpoints/baseline_3ch/$fold/convnext_tiny/lodo_${fold}_convnext_tiny_best.pt" \
           "$TMP_ROOT/$fold/convnext_tiny/lodo_${fold}_convnext_tiny_best.pt"
  fi
done

mkdir -p saved/logs/calibration

echo "Starting calibration study on convnext_tiny × 4 folds"
bash "$PYRUN" -u -m src.eval.run_calibration_study \
    --folds eyepacs aptos messidor2 ddr \
    --split-root data/splits/lodo_unbalanced \
    --checkpoint-root "$TMP_ROOT" \
    --output-dir saved/logs/calibration \
    --batch-size 64

echo "Calibration study complete. Results in saved/logs/calibration/."
