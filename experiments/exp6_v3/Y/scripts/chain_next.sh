#!/bin/bash
# EXP6 V3 overnight -- on this node: wait until no t1_queue worker of WAIT_JOBFILE is alive, then start WORKERS t1_queue
# workers per GPU on NEXT_JOBFILE and record the event. Usage: chain_next.sh WAIT_JOBFILE NEXT_JOBFILE WORKERS
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
WJ=$(basename $1); NJ=$2; NW=$3
sleep 120
while [ $(pgrep -f "[t]1_queue.sh [0-9] .*$WJ" | wc -l) -gt 0 ]; do sleep 60; done
for g in 0 1 2 3; do for w in $(seq $NW); do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $NJ > /dev/null 2>&1 < /dev/null & sleep 1; done; done
echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"track\": \"YD\", \"event\": \"chain_started\", \"host\": \"$(hostname -s)\", \"after\": \"$WJ\", \"jobfile\": \"$(basename $NJ)\", \"workers_per_gpu\": $NW}" >> work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead/EVENTS.jsonl
