#!/bin/bash
# EXP7 Pool A: MP3D-150 workers on the interactive allocation 3489889 (gpu114, 4 x L40S): per GPU one slot-A + one slot-B worker.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
Q=$R/experiments/cvpr_results/exp7/scripts/mp3d_queue.sh
J=$R/work_dirs/cvpr_results/exp7/jobs
for g in 0 1 2 3; do
  setsid nohup bash $Q $g $J/mp3d150_slotA.txt A > /dev/null 2>&1 < /dev/null & sleep 5
  setsid nohup bash $Q $g $J/mp3d150_slotB.txt A > /dev/null 2>&1 < /dev/null & sleep 5
done
