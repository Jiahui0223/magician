#!/bin/bash
# Experiment F -- state-cache worker (seen2scene env). Trajectory-level claims (work_dirs/.../claims) balance the workers; explore first, then confirm.
# Usage: f_queue.sh GPU
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; host=$(hostname -s); mkdir -p /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/states
for set in explore confirm; do
  CUDA_VISIBLE_DEVICES=$g LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $S2E/bin/python /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/scripts/f_run.py --set $set >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/states/${host}_g${g}.log 2>&1
  echo "exit $? $set $host g$g $(date +%FT%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/state_queue.log
done
