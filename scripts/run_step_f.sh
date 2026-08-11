#!/bin/bash
# Step F: 4-channel seg ablation on the remaining 2 folds (eyepacs, ddr) for all 3 variants.
# 3 variants × 2 folds = 6 runs. Combined with Step C, this gives a full 3 variants × 4 folds = 12-run matrix.
# With auto-recovery: per-run OOM triggers a batch-size halving and retry.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYRUN="$SCRIPT_DIR/_pvrun.sh"
MODEL="configs/models/convnext_tiny.yaml"
HOLDOUTS=(eyepacs ddr)
VARIANTS=(four_ch_soft four_ch_tversky four_ch_morph)

if [ $# -gt 0 ]; then
  SEEN=()
  for arg in "$@"; do
    case "$arg" in
      soft) SEEN+=(four_ch_soft) ;;
      tversky) SEEN+=(four_ch_tversky) ;;
      morph) SEEN+=(four_ch_morph) ;;
      eyepacs|ddr) HOLDOUTS=("$arg") ;;
      *) echo "unknown selector: $arg (use soft|tversky|morph or holdout name)"; exit 1 ;;
    esac
  done
  if [ ${#SEEN[@]} -gt 0 ]; then
    VARIANTS=("${SEEN[@]}")
  fi
fi

mkdir -p saved/logs

# Track OOM retries per (variant, holdout) tuple
declare -A RETRY_COUNT

run_one() {
  local variant="$1"; local holdout="$2"
  local arch="convnext_tiny"
  local key="${variant}_${holdout}"
  local variant_config="configs/variants/${variant}.yaml"
  local lodo_config="configs/lodo_balanced/${holdout}.yaml"
  local log_dir="saved/logs/${variant}/${holdout}"
  local run_name="lodo_${holdout}_${arch}"
  local train_log="${log_dir}/${run_name}_train.log"
  local eval_log="${log_dir}/${run_name}_eval.log"
  # Variant's checkpoint_dir wins in merge_configs; ckpt is flat under variant dir.
  local ckpt="saved/checkpoints/${variant}/${run_name}_best.pt"

  # If a usable ckpt from a previous run already exists, skip training
  if [ -f "$ckpt" ]; then
    echo "[$(date '+%F %T')] $key: checkpoint already exists at $ckpt — skipping"
    return 0
  fi

  mkdir -p "${log_dir}"

  local retry="${RETRY_COUNT[$key]:-0}"
  echo "================================================================"
  echo "Step F run: variant=${variant}  holdout=${holdout}  retry=${retry}"
  echo "  start: $(date '+%F %T')"
  echo "================================================================"

  local rc=0
  bash "$PYRUN" -u -m src.train \
      --model-config  "$MODEL" \
      --variant-config "$variant_config" \
      --lodo-config   "$lodo_config" \
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
p = '$MODEL'
with open(p) as f: data = yaml.safe_load(f)
for ph_name, ph in data.get('phases', {}).items():
    if 'batch_size' in ph:
        ph['batch_size'] = max(1, ph['batch_size'] // 2)
with open(p, 'w') as f: yaml.dump(data, f, sort_keys=False)
"
      rm -f "$ckpt"
      # Recurse: re-run this exact same (variant, holdout)
      run_one "$variant" "$holdout"
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
for variant in "${VARIANTS[@]}"; do
  for holdout in "${HOLDOUTS[@]}"; do
    total=$((total + 1))
    echo ""
    echo ">>> [Step F ${total}/6] starting ${variant} × ${holdout} <<<"
    run_one "${variant}" "${holdout}"
  done
done
elapsed=$(( $(date +%s) - start_time ))
echo "================================================================"
echo "Step F complete: ${total} runs in $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
echo "================================================================"
