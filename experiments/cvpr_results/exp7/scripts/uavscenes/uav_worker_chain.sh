#!/bin/bash
# One UAV worker on GPU $1: batch 1, batch 2, then waits (<= 3 h) for batch 3 and runs it. Claims prevent duplicate units.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
U=$R/experiments/cvpr_results/exp7/scripts/uavscenes; J=$R/work_dirs/cvpr_results/exp7/jobs
for b in 1 2 3; do
  for i in $(seq 1 360); do [ -f $J/uav_v0_batch$b.txt ] && break; sleep 30; done
  [ -f $J/uav_v0_batch$b.txt ] && bash $U/uav_queue.sh $1 $J/uav_v0_batch$b.txt A
done
