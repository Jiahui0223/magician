#!/bin/bash
# Usage: bash uav_start_workers.sh "GPU GPU ..." N_PER_GPU   -- each worker: batch 1, then waits (<= 2 h) for batch 2 and runs it
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
U=$R/experiments/cvpr_results/exp7/scripts/uavscenes; J=$R/work_dirs/cvpr_results/exp7/jobs
for g in $1; do for w in $(seq 1 $2); do
  setsid nohup bash -c "bash $U/uav_queue.sh $g $J/uav_v0_batch1.txt A; for i in \$(seq 1 240); do [ -f $J/uav_v0_batch2.txt ] && break; sleep 30; done; [ -f $J/uav_v0_batch2.txt ] && bash $U/uav_queue.sh $g $J/uav_v0_batch2.txt A" > /dev/null 2>&1 < /dev/null &
  sleep 4
done; done
