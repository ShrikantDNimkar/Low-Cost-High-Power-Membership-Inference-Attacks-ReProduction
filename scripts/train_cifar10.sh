#!/bin/bash
# Train the CIFAR-10 model set with the paper's architecture (WideResNet-28-2).
# Designed for Apple Silicon: uses the Metal GPU automatically (--device auto).
#
# Usage:  bash scripts/train_cifar10.sh [NUM_MODELS]
#   16 models  (~2-4 h on M1 Pro):  supports 1/2/4-ref-model experiments
#   32 models  (~4-8 h, overnight): solid curves
#   64 models  (~8-16 h, weekend):  near-paper quality
# The paper itself uses 256 models and 100 epochs on CIFAR-10.
#
# Augmented setting (paper's strongest attack, Figs. 4-6): also saves each
# model's logits on Q queries per point (clean + Q-1 random augmentations).
#   AUGMENTS=18 bash scripts/train_cifar10.sh 32
# Costs roughly Q/2 extra minutes per model on top of training.
# Then audit with:  bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10 18
# (or python main.py ... --num-augmentations 18)
#
# Resume after an interruption (models already on disk are NOT retrained):
#   AUGMENTS=18 START=5 bash scripts/train_cifar10.sh 16
# A single failed model no longer stops the run: it is reported at the end
# and can be retried the same way.

NUM_MODELS=${1:-32}
EPOCHS=${EPOCHS:-100}
AUGMENTS=${AUGMENTS:-1}
START=${START:-0}
LOGDIR=exp/cifar10
DATA_DIR=data/cifar

mkdir -p logs/cifar10
FAILED=""
for i in $(seq "$START" $((NUM_MODELS - 1))); do
    echo "=== training model $i / $((NUM_MODELS - 1)) ==="
    if python -m src.train --dataset cifar10 --data-dir "$DATA_DIR" \
        --arch wrn28-2 --expid "$i" --num-experiments "$NUM_MODELS" \
        --logdir "$LOGDIR" --epochs "$EPOCHS" --device auto \
        --num-augmentations "$AUGMENTS" \
        &> "logs/cifar10/log_$i"; then
        tail -n 2 "logs/cifar10/log_$i"
    else
        echo "[train] WARNING: model $i FAILED - see logs/cifar10/log_$i (continuing)"
        FAILED="$FAILED $i"
    fi
done
echo "Done. Models in $LOGDIR/experiment-*/"
if [ -n "$FAILED" ]; then
    echo "FAILED models:$FAILED"
    echo "Retry them with:  AUGMENTS=$AUGMENTS START=$(echo $FAILED | awk '{print $1}') bash scripts/train_cifar10.sh $NUM_MODELS"
fi
