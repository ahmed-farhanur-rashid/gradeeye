#!/bin/bash
# Train one LODO fold and evaluate on the full held-out test set declared by
# the LODO config.
#
# Usage (split form, preferred):
#   bash scripts/run_train_eval.sh configs/models/convnext_tiny.yaml configs/lodo_balanced/eyepacs.yaml
#   bash scripts/run_train_eval.sh configs/models/deit3_small_patch16_384.yaml configs/lodo_balanced/aptos.yaml --tta
#
# Usage (legacy monolithic form, back-compat):
#   bash scripts/run_train_eval.sh configs/lodo_eyepacs_convnext_tiny.yaml
set -e

cd "$(dirname "$0")/.."

VENV_PYTHON="bash $(dirname "$0")/_pvrun.sh"

# Detect form: model + LODO config, or one legacy monolithic config.
if [ $# -ge 2 ] && [[ "$1" == configs/models/*.yaml ]] && [[ "$2" == configs/*/*.yaml ]]; then
  MODEL_CONFIG="$1"
  LODO_CONFIG="$2"
  HOLDOUT=$(awk '/^holdout_source:/ {print $2}' "$LODO_CONFIG")
  TEST_MANIFEST=$(awk '/^manifest_test:/ {print $2}' "$LODO_CONFIG")
  ARCH=$(basename "$MODEL_CONFIG" .yaml)
  EXPERIMENT=$(basename "$(dirname "$LODO_CONFIG")")
  RUN_NAME="lodo_${HOLDOUT}_${ARCH}"
  CKPT="saved/checkpoints/${EXPERIMENT}/${HOLDOUT}/${ARCH}/${RUN_NAME}_best.pt"
  TRAIN_CMD=("$VENV_PYTHON" -u -m src.train --model-config "$MODEL_CONFIG" --lodo-config "$LODO_CONFIG")
  shift 2
elif [ $# -ge 1 ] && [[ "$1" == configs/*.yaml ]]; then
  CONFIG="$1"
  RUN_NAME=$(basename "$CONFIG" .yaml)
  HOLDOUT=$(awk '/^holdout_source:/ {print $2}' "$CONFIG")
  TEST_MANIFEST=$(awk '/^manifest_test:/ {print $2}' "$CONFIG")
  CKPT="saved/checkpoints/${RUN_NAME}_best.pt"
  TRAIN_CMD=("$VENV_PYTHON" -u -m src.train --config "$CONFIG")
  shift
else
  echo "usage: $0 <model.yaml> <lodo.yaml> [--tta] OR $0 <monolithic.yaml> [--tta]"
  exit 1
fi

if [ -z "$TEST_MANIFEST" ]; then
  echo "ERROR: manifest_test is missing from the LODO config"
  exit 1
fi

mkdir -p "saved/logs"

echo "=== Training: $RUN_NAME ==="
"${TRAIN_CMD[@]}" 2>&1 | tee "saved/logs/${RUN_NAME}_train.log"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: expected checkpoint at $CKPT"
  exit 1
fi

echo ""
echo "=== Evaluation on full held-out source: $HOLDOUT ($TEST_MANIFEST) ==="
TTA_ARGS=()
if [ "${1:-}" = "--tta" ]; then
  TTA_ARGS=(--tta)
fi

"$VENV_PYTHON" -m src.eval.evaluate \
  --checkpoint "$CKPT" \
  --manifest "$TEST_MANIFEST" \
  "${TTA_ARGS[@]}" \
  2>&1 | tee "saved/logs/${RUN_NAME}_eval.log"

echo ""
echo "=== Done. Results saved to saved/logs/ ==="
