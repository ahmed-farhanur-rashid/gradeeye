#!/bin/bash
# End-to-end smoke test: run a full train+eval cycle with 1 epoch each phase
# on the fastest config (convnext_tiny + 4ch_soft + messidor2).
# Restores configs after, regardless of outcome.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYRUN="$SCRIPT_DIR/_pvrun.sh"
MODEL_CFG="configs/models/convnext_tiny.yaml"

cp "$MODEL_CFG" "$MODEL_CFG.smoke_bak"

echo "[$(date '+%F %T')] smoke: truncate phase epochs to 1 for fast test"
bash "$PYRUN" -c "
import yaml
p = '$MODEL_CFG'
with open(p) as f: data = yaml.safe_load(f)
data['phases']['phase1_frozen']['num_epochs'] = 1
data['phases']['phase2_full_training']['num_epochs'] = 1
data['phases']['phase2_full_training']['min_epochs'] = 1
data['phases']['phase2_full_training']['overfitting_patience'] = 1
with open(p, 'w') as f: yaml.dump(data, f, sort_keys=False)
"

SMOKE_LOG="saved/logs/_smoke_test.log"
mkdir -p saved/logs

echo "[$(date '+%F %T')] smoke: launch train (1+1 epochs, convnext_tiny + 4ch_soft + messidor2)"
bash "$PYRUN" -u -m src.train \
    --model-config configs/models/convnext_tiny.yaml \
    --variant-config configs/variants/four_ch_soft.yaml \
    --lodo-config configs/lodo_balanced/messidor2.yaml \
    --experiment-name _smoke_test \
    > "$SMOKE_LOG" 2>&1

rc=$?
echo "[$(date '+%F %T')] smoke: train rc=$rc"

if [ $rc -ne 0 ]; then
  echo "smoke: train failed — tail of log:"
  tail -20 "$SMOKE_LOG"
  cp "$MODEL_CFG.smoke_bak" "$MODEL_CFG"
  rm "$MODEL_CFG.smoke_bak"
  exit $rc
fi

CKPT="saved/checkpoints/_smoke_test/messidor2/convnext_tiny/lodo_messidor2_convnext_tiny_best.pt"
if [ ! -f "$CKPT" ]; then
  echo "smoke: checkpoint missing at $CKPT"
  tail -20 "$SMOKE_LOG"
  cp "$MODEL_CFG.smoke_bak" "$MODEL_CFG"
  rm "$MODEL_CFG.smoke_bak"
  exit 1
fi

echo "[$(date '+%F %T')] smoke: launch eval on full test set"
TEST_MANIFEST=$(awk '/^manifest_test:/ {print $2}' configs/lodo_balanced/messidor2.yaml)
bash "$PYRUN" -m src.eval.evaluate \
    --checkpoint "$CKPT" \
    --manifest "$TEST_MANIFEST" \
    >> "$SMOKE_LOG" 2>&1

rc=$?
echo "[$(date '+%F %T')] smoke: eval rc=$rc"

cp "$MODEL_CFG.smoke_bak" "$MODEL_CFG"
rm "$MODEL_CFG.smoke_bak"

if [ $rc -ne 0 ]; then
  echo "smoke: eval failed — tail of log:"
  tail -30 "$SMOKE_LOG"
  exit $rc
fi

echo "[$(date '+%F %T')] smoke: PASS"
tail -10 "$SMOKE_LOG"
exit 0