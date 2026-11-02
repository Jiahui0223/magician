#!/bin/bash
# anim_run.sh [replay|gt] : data for the exploration animations (4 outdoor + 2 indoor V4 closed-loop episodes).
#   replay  seen2scene env, GPUs 0-3: anim_replay.py (planner imagination per step; asserted equal to the planner logs)
#   gt      magician env (outdoor, GPU) / CPU (indoor): anim_gt.py (GT surface points + first covered step; asserted equal
#           to the stored coverage curves)
# Episodes (rule): outdoor = the median-AUC start of V4 in colosseum, pantheon, alhambra, neuschwanstein; indoor = the
# two floors where V4 has the highest floor-mean AUC, the start with the higher AUC.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; SC=$R/experiments/exp6_v4/weekly_report/scripts; V=$R/work_dirs/exp6_v4/weekly_report_anim
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
OUT="colosseum:2 pantheon:3 alhambra:2 neuschwanstein:0"; IN="oLBMNvg9in8_L1:36 TbHJrupSAjP_L1:112"
cd $R; mkdir -p $V
if [ "$1" = replay ]; then
  export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  g=0
  for e in $OUT; do s=${e%%:*}; k=${e##*:}; f=$V/out_${s}_s${k}.npz; [ -f $f ] && continue
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/anim_replay.py --domain outdoor --scene $s --start $k --out $f > $V/out_${s}_s${k}.log 2>&1 &
    g=$(( (g + 1) % 4 )); done
  for e in $IN; do s=${e%%:*}; k=${e##*:}; f=$V/in_${s}_s${k}.npz; [ -f $f ] && continue
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/anim_replay.py --domain indoor --key $s --start $k --out $f > $V/in_${s}_s${k}.log 2>&1 &
    g=$(( (g + 1) % 4 )); done
  wait
  grep -h "saved\|Error\|assert" $V/out_*.log $V/in_*.log
elif [ "$1" = gt ]; then
  for e in $IN; do s=${e%%:*}; k=${e##*:}; $MG $SC/anim_gt.py --domain indoor --key $s --start $k --out $V/gt_in_${s}_s${k}.npz > $V/gt_in_${s}_s${k}.log 2>&1 & done
  g=0
  for e in $OUT; do s=${e%%:*}; k=${e##*:}
    CUDA_VISIBLE_DEVICES=$g $MG $SC/anim_gt.py --domain outdoor --scene $s --start $k --out $V/gt_out_${s}_s${k}.npz > $V/gt_out_${s}_s${k}.log 2>&1 &
    g=$(( (g + 1) % 4 )); done
  wait
  grep -h "saved\|Error\|assert" $V/gt_*.log
fi
