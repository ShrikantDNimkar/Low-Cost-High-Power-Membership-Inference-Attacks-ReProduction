#!/bin/bash
# Run all attacks on the trained Purchase-100 model set.
# Usage: bash scripts/attack_purchase100.sh [LOGDIR] [OUTDIR]

set -e
LOGDIR=${1:-exp/purchase100}
OUTDIR=${2:-results/purchase100}
WORKERS=${3:-0}   # 0 = auto (all cores, memory-capped); 1 = serial

python main.py --logdir "$LOGDIR" --target-idx 0 --num-ref-models 1 2 4 8 \
    --out "$OUTDIR" --workers "$WORKERS"
