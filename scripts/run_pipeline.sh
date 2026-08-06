#!/bin/bash
# Run the data pipeline end-to-end: manifests → preprocess → LODO splits.
# Designed for one-shot execution after a fresh data setup.
set -e

cd "$(dirname "$0")/.."

VENV_PYTHON="/home/farhan/my-projects/.venv/bin/python"

echo "=== Step 1: Build manifests from raw data ==="
$VENV_PYTHON -m src.data.cli manifests --dataset all

echo ""
echo "=== Step 2: Preprocess (border crop, CLAHE, resize to 384) ==="
$VENV_PYTHON -m src.data.cli preprocess --dataset all

echo ""
echo "=== Step 3: Build LODO splits (3 folds × 4 CSVs each) ==="
$VENV_PYTHON -m src.data.cli splits --holdout all

echo ""
echo "=== Step 4: (Optional) Pre-compute segmentation masks ==="
echo "  Skipping by default. Run manually if you want Option A segmentation:"
echo "  $VENV_PYTHON -m src.data.cli segmentation --datasets aptos messidor2 eyepacs"

echo ""
echo "=== Pipeline complete ==="
echo "Generate the LODO configs (5 model yamls + 3 LODO yamls):"
echo "  $VENV_PYTHON -m src.data.lodo_configs"
echo "Train one LODO fold:"
echo "  $VENV_PYTHON -m src.train --model-config configs/models/convnext_tiny.yaml \\"
echo "                                     --lodo-config configs/lodo/eyepacs.yaml"