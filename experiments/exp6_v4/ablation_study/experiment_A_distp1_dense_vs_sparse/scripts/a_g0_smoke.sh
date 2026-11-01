#!/bin/bash
# Experiment A G0 closed-loop checks (magician env, 2 GPUs): (1) A0 = V4 planner in the default b2 mode, 2 moves on one
# MP3D start and one Macarons++ start -> must reproduce the stored V4 G3 episodes bitwise (poses, coverage, step scores);
# (2) A1 = same with V4_P1_MODE=full -> runs, same candidates, scores differ only through band_dil (checked by a_g0_check.py).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; SC=$R/experiments/exp6_v4/scripts
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
W=$R/work_dirs/exp6_v4/ablation_study/experiment_A/smoke; ck=$R/work_dirs/exp6_v4/checkpoints/final.pt
cd $R; rm -rf $W; mkdir -p $W/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
g1=${1:-0}; g2=${2:-1}
( for mode in b2 full; do
    V4_P1_MODE=$mode CUDA_VISIBLE_DEVICES=$g1 $MG $SC/sim_in_v4.py --v4-ckpt $ck --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 3 --out $W/indoor_$mode > $W/logs/in_$mode.log 2>&1; echo "in_$mode exit $?" >> $W/logs/status.log
  done ) &
( for mode in b2 full; do
    V4_P1_MODE=$mode CUDA_VISIBLE_DEVICES=$g2 $MG $SC/sim_out_v4.py --v4-ckpt $ck --group V2M --scene barts --start 0 --moves 3 --root $W/outdoor_$mode/V2M > $W/logs/out_$mode.log 2>&1; echo "out_$mode exit $?" >> $W/logs/status.log
  done ) &
wait; echo "A G0 SMOKES DONE $(date '+%F %T')" >> $W/logs/status.log
