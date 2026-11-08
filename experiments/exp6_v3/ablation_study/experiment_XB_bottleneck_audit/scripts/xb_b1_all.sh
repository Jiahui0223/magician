#!/bin/bash
# Experiment XB / B1 -- the 8 outdoor trajectories of the Task A pool, sequentially on one GPU. Usage: xb_b1_all.sh GPU
GPU=$1; R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R; X=$R/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R
for j in "explore g2_B0_barts_s0" "explore g2_B0_pantheon_s0" "explore e3_colosseum_s0" "confirm conf_eiffel_s0" "confirm conf_sestino_museum_s0" "confirm conf_bannerman_s0" "confirm conf_liberty_s0" "explore e3_pisa_s1"; do
  set -- $j; [ -f $X/results/B1_fidelity/$2.json ] && continue
  CUDA_VISIBLE_DEVICES=$GPU $S2E/bin/python $X/scripts/xb_b1_fidelity.py --set $1 --tag $2 > $X/logs/b1/$2.log 2>&1; echo "exit $? $2 $(date +%FT%T)" >> $X/logs/b1/queue.log
done
