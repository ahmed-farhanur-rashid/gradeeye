#!/bin/bash
# Train one LODO fold and evaluate on the held-out source's matched test set.
#
# Usage (split form, preferred):
#   bash scripts/run_train_eval.sh configs/models/convnext_tiny.yaml configs/lodo/eyepacs.yaml
#   bash scripts/run_train_eval.sh configs/models/deit3_small_patch16_384.yaml configs/lodo/aptos.yaml --tta
#
# Usage (legacy monolithic form, back-compat):
#   bash scripts/run_train_eval.sh configs/lodo_eyepacs_convnext_tiny.yaml
set -e

cd "$(dirname "$0")/.."

# Detect form: 1 arg = legacy monolithic, 2 args = split.
if [ $# -ge 2 ] && [[ "$2" == configs/lodo/* ]]; then
  MODEL_CONFIG="$1"
  LODO_CONFIG="$2"
  RUN_NAME="$(basename "$MODEL_CONFIG" .yaml)_$(basename "$LODO_CONFIG" .yaml)"
  # The trainer derives run_name as f"lodo_{holdout}_{arch}", so build that here
  # for the checkpoint filename convention.
  HOLDOUT=$(basename "$LODO_CONFIG" .yaml)
  ARCH=$(basename "$MODEL_CONFIG" .yaml)
  RUN_NAME="lodo_${HOLDOUT}_${ARCH}"
  shift 2
elif [ $# -ge 1 ] && [[ "$1" == configs/*.yaml ]]; then
  CONFIG="$1"
  RUN_NAME=$(basename "$CONFIG" .yaml)
  HOLDOUT=$(grep -E "^holdout_source:" "$CONFIG" | awk '{print $2}')
  # Synthesize a "model config" that's actually the monolithic file — use --config flag.
  TRAIN_CMD=(python -u -m src.train --config "$CONFIG")
  shift
else
  echo "usage: $0 <model.yaml> <lodo.yaml>  OR  $0 <monolithic.yaml>"
  exit 1
fi

if [ -n "${MODEL_CONFIG:-}" ]; then
  TRAIN_CMD=(python -u -m src.train --model-config "$MODEL_CONFIG" --lodo-config "$LODO_CONFIG")
fi

VENV_PYTHON="/home/farhan/my-projects/.venv/bin/python"
CKPT="saved/checkpoints/${RUN_NAME}_best.pt"

# Derive the held-out source + its matched test manifest.
TEST_MANIFEST="data/splits/lodo_${HOLDOUT}_test_matched.csv"

echo "=== Training: $RUN_NAME ==="
"${TRAIN_CMD[@]}" 2>&1 | tee "saved/logs/${RUN_NAME}_train.log"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: expected checkpoint at $CKPT"
  exit 1
fi

echo ""
echo "=== Evaluation on held-out source: $HOLDOUT ($TEST_MANIFEST) ==="
TTA_FLAG=""
if [ "${1:-}" = "--tta" ]; then
  TTA_FLAG="--tta"
fi

$VENV_PYTHON -m src.eval.evaluate \
  --checkpoint "$CKPT" \
  --manifest "$TEST_MANIFEST" \
  $TTA_FLAG \
  2>&1 | tee "saved/logs/${RUN_NAME}_eval.log"

echo ""
echo "=== Done. Results saved to saved/logs/ ==="
