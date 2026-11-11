#!/bin/bash
# Wait until every unit of the UAV batches 1-3 has an episode, then start the coverage replay (2 workers per gpu114 GPU).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; LOG=$W7/logs/uav_after_all.log
while true; do
  miss=0
  for f in $J/uav_v0_batch1.txt $J/uav_v0_batch2.txt $J/uav_v0_batch3.txt; do
    while IFS=: read m s k; do
      case $m in OURS) e=$W7/episodes/uavscenes/raw/OURS/V2M/episodes/ep_V2M_${s}_s${k}_r0.json;; RANDOM) e=$W7/episodes/uavscenes/raw/RANDOM/random/episodes/ep_random_${s}_s${k}_r0.json;; esac
      [ -f $e ] || miss=$((miss+1))
    done < $f
  done
  [ $miss -eq 0 ] && break
  sleep 120
done
echo "$(date '+%F %T') all 120 closed-loop episodes present; starting coverage replay" >> $LOG
for g in 0 1 2 3; do for w in 1 2; do setsid nohup bash $R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_cov_queue.sh $g > /dev/null 2>&1 < /dev/null & sleep 5; done; done
