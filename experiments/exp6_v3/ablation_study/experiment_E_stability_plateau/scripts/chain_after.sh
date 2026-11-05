#!/bin/bash
# EXP6 V3 / ablation E -- start WORKER_SCRIPT on GPUS for JOBFILE once no process matching PATTERN is alive on this node.
# Usage: chain_after.sh PATTERN WORKER_SCRIPT JOBFILE GPU [GPU ...]
pat=$1; ws=$2; jf=$3; shift 3
sleep 60; while pgrep -f "$pat" > /dev/null; do sleep 60; done
for g in "$@"; do setsid nohup bash $ws $g $jf > /dev/null 2>&1 < /dev/null & sleep 2; done
echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"event\": \"chain_started\", \"host\": \"$(hostname -s)\", \"jobfile\": \"$(basename $jf)\", \"gpus\": \"$*\"}" >> /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau/EVENTS.jsonl
