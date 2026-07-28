#!/bin/bash
# Train a model and evaluate it on Messidor2 (zero-shot external val).
# Usage:
#   bash scripts/run_train_eval.sh configs/full_method_convnext.yaml
#   bash scripts/run_train_eval.sh configs/full_method_swint.yaml
set -e

CONFIG="${1:?usage: $0 <config.yaml>}"
cd "$(dirname "$0")/.."

VENV_PYTHON="/home/farhan/my-projects/.venv/bin/python"
RUN_NAME=$(basename "$CONFIG" .yaml)
CKPT="saved/checkpoints/${RUN_NAME}_best.pt"

echo "=== Training: $CONFIG ==="
$VENV_PYTHON -u scripts/train.py --config "$CONFIG" 2>&1 | tee "saved/logs/${RUN_NAME}_train.log"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: expected checkpoint at $CKPT"
  exit 1
fi

echo ""
echo "=== Zero-shot evaluation on Messidor2 ==="
# Use the model's img_size from config if present
IMG_SIZE=$(grep -E "^\s*img_size:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
TTA_FLAG=""
if [ "${2:-}" = "--tta" ]; then
  TTA_FLAG="--tta"
fi
EXTRA=""
if [ -n "$IMG_SIZE" ] && [ "$IMG_SIZE" != "384" ]; then
  EXTRA="--img-size $IMG_SIZE"
fi

$VENV_PYTHON scripts/evaluate.py \
  --checkpoint "$CKPT" \
  --manifest data/splits/messidor2_test.csv \
  $TTA_FLAG \
  $EXTRA \
  2>&1 | tee "saved/logs/${RUN_NAME}_messidor2_eval.log"

echo ""
echo "=== Done. Results saved to saved/logs/ ==="
