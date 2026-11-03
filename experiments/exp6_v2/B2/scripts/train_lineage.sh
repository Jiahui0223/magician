#!/bin/bash
# B2 has no training of its own: its weights are V2-Mix EMA@17.5k. This script records (and with RUN=1 re-executes) the two
# historical stages exactly as they were run, writing to NEW output folders (never the historical ones):
#   stage 1  V2 Original  : EXP5 Task 1 t12_train.py --config train_main.json  (60k steps, random init, outdoor sparse P1)
#   stage 2  V2-Mix       : EXP5 Task 2 t22_train_mix.py --config train_mix_v1.json (20k steps from V2 Original EMA@60k)
#   selection             : t22_select.py --run <stage-2 dir> (frozen rule -> EMA@17.5k)
# Not run during the 2026-11-02 organisation (no retraining was requested).
source $(dirname $0)/../../shared/env.sh
O=${OUT:-$R/work_dirs/exp6_v2/B2/retrain}
C1="cd $EXP5/task1/scripts && CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 t12_train.py --config train_main.json --out $O/v2_original"
C2="cd $EXP5/task2/scripts && CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 t22_train_mix.py --config train_mix_v1.json --out $O/v2mix"
C3="cd $EXP5/task2/scripts && $PY t22_select.py --run $O/v2mix"
echo "$C1"; echo "$C2   # NOTE: train_mix_v1.json init_checkpoint = the historical V2 Original EMA@60k (sha a379c0fb...)"; echo "$C3"
[ "${RUN:-0}" = 1 ] && { eval "$C1" && eval "$C2" && eval "$C3"; }
