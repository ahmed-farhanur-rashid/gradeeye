#!/bin/bash
# Step B: 3-channel balanced baseline across 3 architectures × 4 LODO folds = 12 runs.
# With auto-recovery: per-run OOM triggers a batch-size halving and retry.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYRUN="$SCRIPT_DIR/_pvrun.sh"
HOLDOUTS=(eyepacs aptos messidor2 ddr)
ARCHS=(convnext_tiny deit3_small_patch16_384 maxvit_tiny_tf_384)
EXPERIMENT_NAME="baseline_3ch"

if [ $# -gt 0 ]; then
  SEEN=()
  for arg in "$@"; do
    case "$arg" in
      convnext_tiny|convnext) SEEN+=(convnext_tiny) ;;
      deit|deeit|deit3|deit3_small_patch16_384) SEEN+=(deit3_small_patch16_384) ;;
      maxvit|maxvit_tiny_tf_384) SEEN+=(maxvit_tiny_tf_384) ;;
      eyepacs|aptos|messidor2|ddr) HOLDOUTS=("$arg") ;;
      *) echo "unknown selector: $arg"; exit 1 ;;
    esac
  done
  if [ ${#SEEN[@]} -gt 0 ]; then
    ARCHS=("${SEEN[@]}")
  fi
fi

mkdir -p saved/logs

declare -A RETRY_COUNT

run_one() {
  local arch="$1"; local holdout="$2"
  local key="${arch}_${holdout}"
  local model_config="configs/models/${arch}.yaml"
  local lodo_config="configs/lodo_balanced/${holdout}.yaml"
  local log_dir="saved/logs/${EXPERIMENT_NAME}/${holdout}/${arch}"
  local ckpt_dir="saved/checkpoints/${EXPERIMENT_NAME}/${holdout}/${arch}"
  local run_name="lodo_${holdout}_${arch}"
  local train_log="${log_dir}/${run_name}_train.log"
  local eval_log="${log_dir}/${run_name}_eval.log"
  local ckpt="${ckpt_dir}/${run_name}_best.pt"

  if [ -f "$ckpt" ]; then
    echo "[$(date '+%F %T')] $key: checkpoint already exists at $ckpt — skipping"
    return 0
  fi

  mkdir -p "${log_dir}" "${ckpt_dir}"

  local retry="${RETRY_COUNT[$key]:-0}"
  echo "================================================================"
  echo "Step B run: arch=${arch}  holdout=${holdout}  retry=${retry}"
  echo "  start: $(date '+%F %T')"
  echo "================================================================"

  local rc=0
  bash "$PYRUN" -u -m src.train \
      --model-config "$model_config" \
      --lodo-config   "$lodo_config" \
      --experiment-name "$EXPERIMENT_NAME" \
      > "$train_log" 2>&1 || rc=$?

  if [ $rc -ne 0 ]; then
    if grep -qE "out of memory|CUDA out of memory" "$train_log" 2>/dev/null; then
      RETRY_COUNT[$key]=$((retry + 1))
      if [ "${RETRY_COUNT[$key]}" -gt 2 ]; then
        echo "[$(date '+%F %T')] $key: $rc retries exhausted — giving up"
        return $rc
      fi
      echo "[$(date '+%F %T')] $key: OOM, halving batch size and retrying"
      bash "$PYRUN" -c "
import yaml
p = '$model_config'
with open(p) as f: data = yaml.safe_load(f)
for ph_name, ph in data.get('phases', {}).items():
    if 'batch_size' in ph:
        ph['batch_size'] = max(1, ph['batch_size'] // 2)
with open(p, 'w') as f: yaml.dump(data, f, sort_keys=False)
"
      rm -f "$ckpt"
      run_one "$arch" "$holdout"
      return $?
    fi
    echo "[$(date '+%F %T')] $key: non-OOM failure rc=$rc"
    return $rc
  fi

  if [ ! -f "$ckpt" ]; then
    echo "ERROR: expected checkpoint at $ckpt" >&2
    return 1
  fi

  local test_manifest
  test_manifest=$(awk '/^manifest_test:/ {print $2}' "$lodo_config")

  echo "" | tee -a "$train_log"
  echo "=== Eval on full held-out test: $holdout ($test_manifest) ===" | tee -a "$train_log"
  bash "$PYRUN" -m src.eval.evaluate \
      --checkpoint "$ckpt" \
      --manifest   "$test_manifest" \
      > "$eval_log" 2>&1

  echo "  done: $(date '+%F %T')"
}

start_time=$(date +%s)
total=0
for arch in "${ARCHS[@]}"; do
  for holdout in "${HOLDOUTS[@]}"; do
    total=$((total + 1))
    echo ""
    echo ">>> [Step B ${total}/12] starting ${arch} × ${holdout} <<<"
    run_one "${arch}" "${holdout}"
  done
done
elapsed=$(( $(date +%s) - start_time ))
echo "================================================================"
echo "Step B complete: ${total} runs in $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
echo "================================================================"