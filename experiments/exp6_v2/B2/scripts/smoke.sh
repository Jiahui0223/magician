#!/bin/bash
# B2: minimal inference smoke through the new entry + before / after equality (future depth, DistP1, ranking, prediction
# statistics vs the historical rows). Usage: smoke.sh [dev3|valsel] [GPU]
source $(dirname $0)/../../shared/env.sh; cd $R
S=${1:-dev3}; CUDA_VISIBLE_DEVICES=${2:-0} $PY $X2/shared/smoke_v2.py --arm B2 --set $S --out $X2/B2/results/smoke_$S.json
