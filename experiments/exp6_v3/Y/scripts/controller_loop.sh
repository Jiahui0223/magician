#!/bin/bash
# EXP6 V3 Y / D -- controller loop (low resource; on the compute node that owns the track): one controller.py pass every 60 s,
# single instance per track (flock). Usage: controller_loop.sh Y|D   (stop: touch <workdir>/STOP_CONTROLLER)
T=$1
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
W=$([ "$T" = Y ] && echo $R/work_dirs/exp6_v3/Y || echo $R/work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead)
mkdir -p $W/logs
exec 9>$W/controller.lock
flock -n 9 || { echo "controller $T already running" >> $W/logs/controller.log; exit 0; }
echo "controller $T start $(hostname -s) $(date '+%F %T')" >> $W/logs/controller.log
while [ ! -f $W/STOP_CONTROLLER ]; do
  timeout 600 /bsuscratch/jiahuizhang/envs/seen2scene/bin/python $R/experiments/exp6_v3/Y/scripts/controller.py --track $T >> $W/logs/controller.log 2>&1
  echo "pass rc=$? $(date '+%F %T')" >> $W/logs/controller.log
  sleep 60
done
echo "controller $T stop $(date '+%F %T')" >> $W/logs/controller.log
