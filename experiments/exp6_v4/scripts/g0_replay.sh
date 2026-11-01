#!/bin/bash
# G0-11b baseline replay (magician env, 2 GPUs): 2-move closed-loop smokes of the FROZEN planners with the current code
# -- V3 path (T0 final.pt, B2-style planner) and V2 path (V2-Mix, Task 2 sparse planner) on both domains; the chosen
# poses are compared afterwards with the stored formal episodes (exp6_v4/scripts/g0_replay_check.py).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
W=$R/work_dirs/exp6_v4/eval/g0_replay; SC=$R/experiments/exp6_v3/shared; T2=$R/experiments/exp5_future_view/task2/scripts
T0=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt
cd $R; rm -rf $W; mkdir -p $W/logs $W/indoor_T0 $W/outdoor_T0/V2M $W/indoor_V2M $W/outdoor_V2M/V2M
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
g1=${1:-1}; g2=${2:-2}
( CUDA_VISIBLE_DEVICES=$g1 $MG $SC/sim_in_v3.py --v3-ckpt $T0 --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 2 --out $W/indoor_T0 > $W/logs/in_T0.log 2>&1; echo "in_T0 exit $?" >> $W/logs/status.log
  CUDA_VISIBLE_DEVICES=$g1 $MG $T2/t23_in_sim.py --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 2 --out $W/indoor_V2M > $W/logs/in_V2M.log 2>&1; echo "in_V2M exit $?" >> $W/logs/status.log ) &
( CUDA_VISIBLE_DEVICES=$g2 $MG $SC/sim_out_v3.py --v3-ckpt $T0 --group V2M --scene barts --start 0 --moves 2 --root $W/outdoor_T0/V2M > $W/logs/out_T0.log 2>&1; echo "out_T0 exit $?" >> $W/logs/status.log
  CUDA_VISIBLE_DEVICES=$g2 $MG $T2/t24_out_sim.py --group V2M --scene barts --start 0 --moves 2 --root $W/outdoor_V2M/V2M > $W/logs/out_V2M.log 2>&1; echo "out_V2M exit $?" >> $W/logs/status.log ) &
wait
echo "REPLAY SMOKES DONE $(date '+%F %T')" >> $W/logs/status.log
