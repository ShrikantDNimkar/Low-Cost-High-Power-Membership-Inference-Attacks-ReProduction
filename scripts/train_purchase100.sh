#!/bin/bash
# Train the Purchase-100 model set (PyTorch re-implementation).
# Mirrors research/2024_rmia/scripts/train_purchase100.sh of the official repo,
# but with a configurable number of models for CPU-only machines.
#
# Usage:  bash scripts/train_purchase100.sh [NUM_MODELS]
#         (default 16; the paper uses 256. Each model takes a few minutes on CPU.)
# Resume after an interruption:  START=5 bash scripts/train_purchase100.sh 16

NUM_MODELS=${1:-16}
EPOCHS=${EPOCHS:-50}
START=${START:-0}
LOGDIR=exp/purchase100
DATA_DIR=data/purchase100

if [ ! -f "$DATA_DIR/x_train.npy" ]; then
    echo "Dataset not found in $DATA_DIR."
    echo "Download the Purchase-100 .npy files first (see README.md, step 1)."
    exit 1
fi

mkdir -p logs/purchase100
FAILED=""
for i in $(seq "$START" $((NUM_MODELS - 1))); do
    echo "=== training model $i / $((NUM_MODELS - 1)) ==="
    if python -m src.train --dataset purchase100 --data-dir "$DATA_DIR" \
        --arch mlp --expid "$i" --num-experiments "$NUM_MODELS" \
        --logdir "$LOGDIR" --epochs "$EPOCHS" \
        &> "logs/purchase100/log_$i"; then
        tail -n 2 "logs/purchase100/log_$i"
    else
        echo "[train] WARNING: model $i FAILED - see logs/purchase100/log_$i (continuing)"
        FAILED="$FAILED $i"
    fi
done
echo "Done. Models in $LOGDIR/experiment-*/"
if [ -n "$FAILED" ]; then
    echo "FAILED models:$FAILED"
    echo "Retry them with:  START=$(echo $FAILED | awk '{print $1}') bash scripts/train_purchase100.sh $NUM_MODELS"
fi
