#!/bin/bash
# Wait for MAG-B v0 (60) and the v1 job list (60) to be complete, then the official coverage replay of every UAV episode (3 workers x 4 GPUs).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; EP=$W7/episodes/uavscenes/raw; LOG=$W7/logs/uav_after_v1.log
done_f() { case $1 in OURS) echo $EP/OURS/V2M/episodes/ep_V2M_$2_s$3_r0.json;; RANDOM) echo $EP/RANDOM/random/episodes/ep_random_$2_s$3_r0.json;; MAG-B) echo $EP/MAG-B/MAG-B/episodes/ep_MAG-B_$2_s$3_r0.json;; esac; }
while true; do
  miss=0
  for f in $J/uav_v0_mag.txt $J/uav_v1.txt; do
    [ -f $f ] || { miss=$((miss+1)); continue; }
    while IFS=: read m s k; do [ -f $(done_f $m $s $k) ] || miss=$((miss+1)); done < $f
  done
  [ $miss -eq 0 ] && break
  sleep 120
done
echo "$(date '+%F %T') all MAG v0 + v1 episodes present; coverage replay" >> $LOG
for g in 0 1 2 3; do for w in 1 2 3; do setsid nohup bash $R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_cov_queue.sh $g > /dev/null 2>&1 < /dev/null & sleep 3; done; done
