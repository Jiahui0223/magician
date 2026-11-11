#!/bin/bash
# Usage: bash uav_start_retry.sh "GPU ..." N_PER_GPU JOBLIST -- plain queue workers over one job list (no chaining)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
for g in $1; do for w in $(seq 1 $2); do
  setsid nohup bash $R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_queue.sh $g $3 R > /dev/null 2>&1 < /dev/null &
  sleep 20
done; done
