#!/bin/bash
# EXP6 V2 -- historical G1 / G2 fixed-state evaluation of B2 and E2 (one runner computes B0 / B2 / E2-S10 / E2-D100 together:
# e2_pilot/scripts/e2_fixed.py, then e2_analyze.py). Tracked form of work_dirs/.../e2/eval_node.sh + chain_g12.sh.
# It writes ONLY missing episode files into the E2 pilot root (work_dirs/exp5_future_view/task3_full_depth_p1/e2/fixed/), so
# with the historical results present it is a no-op; for a read-only check use smoke_v2.py.
# Usage: g12_fixed.sh STEP "<valsel parts>" "<dev3 B0 parts>" "<dev3 B2 parts>"   e.g. g12_fixed.sh 5000 "0/9 1/9" "0/4" ""
source $(dirname $0)/env.sh; cd $R
g=0
for p in $2; do CUDA_VISIBLE_DEVICES=$((g % 4)) $PY $E2P/e2_fixed.py --set valsel --step $1 --part $p & g=$((g + 1)); sleep 2; done
for p in $3; do CUDA_VISIBLE_DEVICES=$((g % 4)) $PY $E2P/e2_fixed.py --set dev3 --stratum B0 --step $1 --part $p & g=$((g + 1)); sleep 2; done
for p in $4; do CUDA_VISIBLE_DEVICES=$((g % 4)) $PY $E2P/e2_fixed.py --set dev3 --stratum B2 --step $1 --part $p & g=$((g + 1)); sleep 2; done
wait
CUDA_VISIBLE_DEVICES="" $PY $E2P/e2_analyze.py --step $1
