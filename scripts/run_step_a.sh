#!/bin/bash
# Run Step A unbalanced baseline on all 4 LODO folds sequentially.
# Each fold trains convnext_tiny 3-channel unbalanced baseline.
# Saves to saved/checkpoints/lodo_unbalanced/<holdout>/convnext_tiny/.

set -e
cd "$(dirname "$0")/.."

declare -A FOLDS=(
    ["eyepacs"]="saved/checkpoints/lodo_unbalanced/eyepacs/convnext_tiny"
    ["aptos"]="saved/checkpoints/lodo_unbalanced/aptos/convnext_tiny"
    ["messidor2"]="saved/checkpoints/lodo_unbalanced/messidor2/convnext_tiny"
    ["ddr"]="saved/checkpoints/lodo_unbalanced/ddr/convnext_tiny"
)

for holdout in eyepacs aptos messidor2 ddr; do
    ckpt_dir="${FOLDS[$holdout]}"
    if [ -f "$ckpt_dir/lodo_${holdout}_convnext_tiny_best.pt" ]; then
        echo "SKIP $holdout: best.pt exists"
        continue
    fi
    echo "=== Running $holdout fold ==="
    PYTHONPATH=. .venv/bin/python -u -m src.train \
        --model-config configs/models/convnext_tiny.yaml \
        --lodo-config configs/lodo_unbalanced/${holdout}.yaml \
        2>&1 | tee saved/logs/lodo_unbalanced/${holdout}/convnext_tiny/run.log
done