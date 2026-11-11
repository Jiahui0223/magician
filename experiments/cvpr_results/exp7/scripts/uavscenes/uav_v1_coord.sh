#!/bin/bash
# EXP7 UAV-v1 coordinator: wait for the v1 tile GPU stage of both tiles -> D1 pose tables -> job list v1 (OURS / RANDOM / MAG-B x 2 tiles
# x 10 starts) -> 2 workers on GPU 3; when the MAG-B v0 workers on GPUs 0-2 have all ended -> 2 workers on each of GPUs 0-2 for v1.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; U=$R/experiments/cvpr_results/exp7/scripts/uavscenes; LOG=$W7/logs/uav_v1_coord.log
until grep -q "rc=" $W7/logs/uav_tile_gpu_AMvalley_01_v1.log 2>/dev/null && grep -q "rc=" $W7/logs/uav_tile_gpu_HKisland_00_v1.log 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') v1 tile stage: $(grep -ho 'rc=[0-9]*' $W7/logs/uav_tile_gpu_*_v1.log | tr '\n' ' ')" >> $LOG
MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp CUDA_VISIBLE_DEVICES=3 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $U/uav_build_pose_tables.py --scenes uav_AMvalley_01_v1 uav_HKisland_00_v1 >> $LOG 2>&1
for m in OURS MAG-B RANDOM; do for t in AMvalley_01_v1 HKisland_00_v1; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "${m}:uav_${t}:${k}"; done; done; done > $J/uav_v1.tmp && mv $J/uav_v1.tmp $J/uav_v1.txt
echo "$(date '+%F %T') job list v1: $(wc -l < $J/uav_v1.txt) units" >> $LOG
for w in 1 2; do setsid nohup bash $U/uav_queue.sh 3 $J/uav_v1.txt V > /dev/null 2>&1 < /dev/null & sleep 20; done
until [ $(pgrep -u $USER -f "uav_queue.sh [012] .*uav_v0_mag.txt" | wc -l) -eq 0 ]; do sleep 60; done
echo "$(date '+%F %T') MAG v0 workers ended; adding v1 workers on GPUs 0-2" >> $LOG
for g in 0 1 2; do for w in 1 2; do setsid nohup bash $U/uav_queue.sh $g $J/uav_v1.txt V > /dev/null 2>&1 < /dev/null & sleep 20; done; done
echo "$(date '+%F %T') done" >> $LOG
