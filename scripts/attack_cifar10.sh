#!/bin/bash
# Run all attacks on a trained CIFAR-10 (or CIFAR-100) model set.
# Usage: bash scripts/attack_cifar10.sh [LOGDIR] [OUTDIR] [NUM_AUGMENTATIONS]
#   bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10
#   bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10 18   # augmented attacks
# (the last argument is only usable if the models were trained with
#  AUGMENTS=18; it adds the paper's augmented RMIA/LiRA variants)

set -e
LOGDIR=${1:-exp/cifar10}
OUTDIR=${2:-results/cifar10}
NAUG=${3:-1}
WORKERS=${4:-0}   # 0 = auto (all cores, memory-capped); 1 = serial

python main.py --logdir "$LOGDIR" --target-idx all --num-ref-models 1 2 4 8 \
    --out "$OUTDIR" --num-augmentations "$NAUG" --workers "$WORKERS"
