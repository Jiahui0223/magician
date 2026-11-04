#!/bin/bash
# EXP6 V3 overnight -- sequential chain on this node: for each "JOBFILE:WORKERS" stage, wait until no t1_queue worker of the
# previous stage's job file is alive, then start the stage. Usage: chain_seq.sh FIRST_WAIT_JOBFILE JOBFILE:N [JOBFILE:N ...]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
prev=$(basename $1); shift
for st in "$@"; do
  jf=${st%%:*}; n=${st##*:}
  sleep 90
  while [ $(pgrep -f "[t]1_queue.sh [0-9] .*$prev" | wc -l) -gt 0 ]; do sleep 60; done
  for g in 0 1 2 3; do for w in $(seq $n); do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $jf > /dev/null 2>&1 < /dev/null & sleep 1; done; done
  echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"track\": \"YD\", \"event\": \"chain_started\", \"host\": \"$(hostname -s)\", \"after\": \"$prev\", \"jobfile\": \"$(basename $jf)\", \"workers_per_gpu\": $n}" >> work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead/EVENTS.jsonl
  prev=$(basename $jf)
done
