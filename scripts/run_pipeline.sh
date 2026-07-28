#!/bin/bash
# Run the preprocessing -> splits pipeline end-to-end.
# Designed for one-shot execution after a fresh data setup.
set -e

cd "$(dirname "$0")/.."

VENV_PYTHON="/home/farhan/my-projects/.venv/bin/python"

echo "=== Step 1: Build manifests from raw data ==="
$VENV_PYTHON scripts/build_manifests.py --dataset all

echo ""
echo "=== Step 2: Preprocess (border crop, CLAHE, resize to 384) ==="
$VENV_PYTHON scripts/preprocess_all.py

echo ""
echo "=== Step 3: Build train/val/test splits ==="
$VENV_PYTHON scripts/build_splits.py --dataset all

echo ""
echo "=== Step 4: (Optional) Pre-compute segmentation masks ==="
echo "  Skipping by default. Run manually if you want Option A segmentation:"
echo "  $VENV_PYTHON scripts/precompute_seg_masks.py --datasets aptos messidor2"

echo ""
echo "=== Pipeline complete ==="
echo "Train with:"
echo "  $VENV_PYTHON scripts/train.py --config configs/full_method_convnext.yaml"