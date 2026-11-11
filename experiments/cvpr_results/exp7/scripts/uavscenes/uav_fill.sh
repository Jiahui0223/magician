#!/bin/bash
# EXP7 UAV coordinator: (a) when the HKisland_00 camera pre-selection has finished: D1 pose table + batch 3 job list;
# (b) every 60 s, start one more worker on any gpu114 GPU with < 22 GB used while unclaimed units remain (Ours ~10 GB each);
# exits when every unit of batches 1-3 is claimed.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; U=$R/experiments/cvpr_results/exp7/scripts/uavscenes; LOG=$W7/logs/uav_fill.log
echo "$(date '+%F %T') start" >> $LOG
while true; do
  if [ ! -f $J/uav_v0_batch3.txt ] && grep -q "rc=0" $W7/logs/uav_tile_gpu_HKisland_00.log 2>/dev/null; then
    MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $U/uav_build_pose_tables.py --scenes uav_HKisland_00 >> $LOG 2>&1
    if [ -f $W7/uav_pose_tables/uav_HKisland_00.npz ]; then
      for m in OURS RANDOM; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "${m}:uav_HKisland_00:${k}"; done; done > $J/uav_v0_batch3.tmp && mv $J/uav_v0_batch3.tmp $J/uav_v0_batch3.txt
      echo "$(date '+%T') batch3 written" >> $LOG
    fi
  fi
  un=0
  for f in $J/uav_v0_batch1.txt $J/uav_v0_batch2.txt $J/uav_v0_batch3.txt; do
    [ -f $f ] || continue
    while IFS=: read m s k; do [ -d $J/uav_claims/${m}_${s}_s${k} ] || un=$((un+1)); done < $f
  done
  if [ $un -eq 0 ] && [ -f $J/uav_v0_batch3.txt ]; then echo "$(date '+%T') all units claimed; exit" >> $LOG; break; fi
  if [ $un -gt 0 ]; then
    for g in 0 1 2 3; do
      mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      if [ "$mem" -lt 22000 ]; then
        setsid nohup bash $U/uav_worker_chain.sh $g > /dev/null 2>&1 < /dev/null &
        echo "$(date '+%T') +worker gpu $g (mem $mem MiB, unclaimed $un)" >> $LOG
        sleep 90; break
      fi
    done
  fi
  sleep 60
done
