#!/bin/bash
# Train the CIFAR-100 model set (WideResNet-28-2, 100 classes).
# Usage: bash scripts/train_cifar100.sh [NUM_MODELS]   (default 32)
# Augmented setting:  AUGMENTS=18 bash scripts/train_cifar100.sh 32
# Resume after an interruption:
#   AUGMENTS=18 START=5 bash scripts/train_cifar100.sh 16
# A single failed model no longer stops the run: it is reported at the end.

NUM_MODELS=${1:-32}
EPOCHS=${EPOCHS:-100}
AUGMENTS=${AUGMENTS:-1}
START=${START:-0}
LOGDIR=exp/cifar100
DATA_DIR=data/cifar

mkdir -p logs/cifar100
FAILED=""
for i in $(seq "$START" $((NUM_MODELS - 1))); do
    echo "=== training model $i / $((NUM_MODELS - 1)) ==="
    if python -m src.train --dataset cifar100 --data-dir "$DATA_DIR" \
        --arch wrn28-2 --expid "$i" --num-experiments "$NUM_MODELS" \
        --logdir "$LOGDIR" --epochs "$EPOCHS" --device auto \
        --num-augmentations "$AUGMENTS" \
        &> "logs/cifar100/log_$i"; then
        tail -n 2 "logs/cifar100/log_$i"
    else
        echo "[train] WARNING: model $i FAILED - see logs/cifar100/log_$i (continuing)"
        FAILED="$FAILED $i"
    fi
done
echo "Done. Models in $LOGDIR/experiment-*/"
if [ -n "$FAILED" ]; then
    echo "FAILED models:$FAILED"
    echo "Retry them with:  AUGMENTS=$AUGMENTS START=$(echo $FAILED | awk '{print $1}') bash scripts/train_cifar100.sh $NUM_MODELS"
fi
