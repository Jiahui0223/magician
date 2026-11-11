#!/bin/bash
# Wait for the HKisland_00 camera pre-selection, then its D1 pose table and the batch-3 job list (picked up by the
# uav_worker_chain.sh workers on GPU 0).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; U=$R/experiments/cvpr_results/exp7/scripts/uavscenes; LOG=$W7/logs/uav_batch3.log
until grep -q "rc=" $W7/logs/uav_tile_gpu_HKisland_00.log 2>/dev/null; do sleep 30; done
echo "$(date '+%T') tile stage $(grep -o 'rc=[0-9]*' $W7/logs/uav_tile_gpu_HKisland_00.log | tail -1)" >> $LOG
grep -q "rc=0" $W7/logs/uav_tile_gpu_HKisland_00.log || exit 1
MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $U/uav_build_pose_tables.py --scenes uav_HKisland_00 >> $LOG 2>&1
[ -f $W7/uav_pose_tables/uav_HKisland_00.npz ] || exit 1
for m in OURS RANDOM; do for k in 0 1 2 3 4 5 6 7 8 9; do echo "${m}:uav_HKisland_00:${k}"; done; done > $J/uav_v0_batch3.tmp && mv $J/uav_v0_batch3.tmp $J/uav_v0_batch3.txt
echo "$(date '+%T') batch3 written ($(wc -l < $J/uav_v0_batch3.txt) units)" >> $LOG
