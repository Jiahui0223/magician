#!/bin/bash
# EXP6 V3 / ablation E -- prediction-replay worker (seen2scene env). Job line: ID EPISODE CKPT ORACLE_JSON PLANNER_LOG(or -) OUT
# waits for the oracle of the same episode; noclobber claims. Usage: e_predict_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; cd $R
E=$R/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/scripts
W=$R/work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p $W/jobs/pclaims $W/logs/predict
for pass in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  left=0
  while read -r id ep ck oj log out; do
    [ -z "$id" ] && continue; [ -f $out ] && continue
    [ -f $W/jobs/pclaims/$id.claim ] && continue
    if [ ! -f $oj ]; then left=1; continue; fi
    ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/jobs/pclaims/$id.claim ) 2>/dev/null || continue
    t0=$(date +%s); args="--episode $ep --ckpt $ck --oracle $oj --out $out"; [ "$log" != "-" ] && args="$args --planner-log $log"
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $E/e_predict.py $args >> $W/logs/predict/$id.log 2>&1
    echo "exit $? $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $W/logs/predict_queue.log
  done < $jf
  [ $left -eq 0 ] && break; sleep 120
done
