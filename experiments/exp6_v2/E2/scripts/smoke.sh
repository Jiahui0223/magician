#!/bin/bash
# E2: minimal inference smoke through the new entry + before / after equality (future depth, DistP1, ranking, prediction
# statistics vs the historical rows). Usage: smoke.sh [dev3|valsel] [GPU] [LABEL]  (LABEL -> results/smoke_<set>_<LABEL>.json)
source $(dirname $0)/../../shared/env.sh; cd $R
S=${1:-dev3}; L=${3:+_$3}; CUDA_VISIBLE_DEVICES=${2:-0} $PY $X2/shared/smoke_v2.py --arm E2 --set $S --out $X2/E2/results/smoke_$S$L.json
