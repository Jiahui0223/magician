#!/bin/bash
# EXP Density A/B -- launch the 12-config fixed-state evaluation (FROZEN_PROTOCOL.md §3): 7 processes on the 4 GPUs of
# the interactive job (episodes split by --part; finished episodes are skipped, so this script can be re-run).
source /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp_density_ab/scripts/env.sh
cd $R
S=experiments/exp_density_ab/scripts/da_fixed.py; L=work_dirs/exp_density_ab/logs
CUDA_VISIBLE_DEVICES=0 $PY $S --set valsel --part 0/3 > $L/fixed_valsel_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S --set valsel --part 1/3 > $L/fixed_valsel_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $S --set valsel --part 2/3 > $L/fixed_valsel_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $S --set dev3 --stratum B0 --part 0/2 > $L/fixed_dev3_B0_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY $S --set dev3 --stratum B0 --part 1/2 > $L/fixed_dev3_B0_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S --set dev3 --stratum B2 --part 0/2 > $L/fixed_dev3_B2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $S --set dev3 --stratum B2 --part 1/2 > $L/fixed_dev3_B2_1.log 2>&1 &
wait
echo "FIXED_DONE $(date '+%F %T')" >> $L/fixed_done.txt
