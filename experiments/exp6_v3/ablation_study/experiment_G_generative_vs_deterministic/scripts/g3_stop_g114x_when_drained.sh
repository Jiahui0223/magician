#!/bin/bash
# Experiment G3 -- detached helper: once no expG3_* task is pending in the shared queue, touch STOP_g114x_gpu{0..3} so the four temporary
# gpu114 workers exit after their current task (instruction of 2026-11-15). Does nothing else. Log: work_dirs/exp6_v3/experiment_G/g3/logs/stop_g114x.log
Q=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/gpu_queue
L=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3/experiment_G/g3/logs/stop_g114x.log
echo "$(date '+%F %T') watcher pid $$ started" >> $L
while [ -n "$(ls $Q/pending | grep '^expG3_')" ]; do sleep 60; done
for g in 0 1 2 3; do touch $Q/STOP_g114x_gpu$g; done
echo "$(date '+%F %T') no expG3_* pending -> touched STOP_g114x_gpu0..3" >> $L
