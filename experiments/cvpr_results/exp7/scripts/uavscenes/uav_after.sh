#!/bin/bash
# EXP7 UAV: (1) when the camera pre-selection of the remaining tiles has finished: D1 pose tables + batch-2 job list;
# (2) when the MP3D-150 workers have ended: 3 UAV workers on each of gpu114 GPUs 0 and 3 (batch 1 then batch 2).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; J=$W7/jobs; U=$R/experiments/cvpr_results/exp7/scripts/uavscenes
LOG=$W7/logs/uav_after.log
TILES="HKisland_00 AMtown_11 HKairport_01"
until [ $(for t in $TILES; do grep -l "rc=" $W7/logs/uav_tile_gpu_$t.log 2>/dev/null; done | wc -l) -ge 3 ]; do sleep 30; done
echo "$(date '+%T') tile stage done: $(for t in $TILES; do echo -n "$t $(grep -o 'rc=[0-9]*' $W7/logs/uav_tile_gpu_$t.log | tail -1) "; done)" >> $LOG
ok=""; for t in $TILES; do grep -q "rc=0" $W7/logs/uav_tile_gpu_$t.log && ok="$ok uav_$t"; done
MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp CUDA_VISIBLE_DEVICES=1 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $U/uav_build_pose_tables.py --scenes $ok >> $LOG 2>&1
: > $J/uav_v0_batch2.tmp
for m in OURS RANDOM; do for t in $ok; do [ -f $W7/uav_pose_tables/$t.npz ] || continue; for k in 0 1 2 3 4 5 6 7 8 9; do echo "$m:$t:$k" >> $J/uav_v0_batch2.tmp; done; done; done
mv $J/uav_v0_batch2.tmp $J/uav_v0_batch2.txt
echo "$(date '+%T') batch2 written: $(wc -l < $J/uav_v0_batch2.txt) units for$ok" >> $LOG
until [ $(ps -u $USER -o args | grep -c "[m]p3d_queue.sh") -eq 0 ]; do sleep 30; done
echo "$(date '+%T') MP3D workers ended; starting 3 UAV workers on GPUs 0 and 3" >> $LOG
for g in 0 3; do for w in 1 2 3; do
  setsid nohup bash -c "bash $U/uav_queue.sh $g $J/uav_v0_batch1.txt A; bash $U/uav_queue.sh $g $J/uav_v0_batch2.txt A" > /dev/null 2>&1 < /dev/null &
  sleep 4
done; done
echo "$(date '+%T') started" >> $LOG
