#!/bin/bash
# Experiment XB / Task B3 -- worker: one GPU, job lines "SET TAG" (SET = indoor | explore | confirm); every job runs all 21 cores (3 models x interventions)
# through xb_b3.py on the trajectory TAG. noclobber claims; output = states/xb_b3/<TAG>.jsonl + <TAG>.a1.json. Usage: xb_b3_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=$R/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; W=$R/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
CORES="T1@none,XA_SR@none,XA_MI@none"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r set tag; do
  [ -z "$set" ] && continue; [[ $set == \#* ]] && continue
  [ -f $W/states/xb_b3/$tag.jsonl ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/claims/xb_b3/$tag.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $X/scripts/xb_b3.py --set $set --group xb_b3 --ckpts $CORES --only $tag >> $X/logs/b3/$tag.log 2>&1; rc=$?
  echo "exit $rc $set $tag $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $X/logs/b3/queue.log
  [ $rc -ne 0 ] && mv $W/claims/xb_b3/$tag.claim $W/claims/xb_b3/$tag.claim.failed.$(date +%s)
done < $jf
