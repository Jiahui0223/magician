#!/bin/bash
# EXP6 V3 / ablation E -- GT-oracle worker (evaluation side, MAGICIAN env). Job line: ID EPISODE PLANNER_LOG(or -) TWO_STEP(0/1) OUT
# noclobber claims, one try per job. Usage: e_oracle_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
E=$R/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/scripts
W=$R/work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p $W/jobs/oclaims $W/logs/oracle
while read -r id ep log two out; do
  [ -z "$id" ] && continue; [[ $id == \#* ]] && continue
  [ -f ${out%.npz}.json ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/jobs/oclaims/$id.claim ) 2>/dev/null || continue
  mkdir -p $(dirname $out); t0=$(date +%s)
  args="--episode $ep --out $out"; [ "$log" != "-" ] && args="$args --planner-log $log"; [ "$two" = "1" ] && args="$args --two-step"
  CUDA_VISIBLE_DEVICES=$g $MG $E/e_oracle.py $args >> $W/logs/oracle/$id.log 2>&1
  echo "exit $? $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $W/logs/oracle_queue.log
done < $jf
