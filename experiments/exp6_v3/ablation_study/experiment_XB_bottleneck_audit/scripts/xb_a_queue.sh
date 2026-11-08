#!/bin/bash
# Experiment XB / Task A -- worker: one GPU, job lines "SET TAG" (SET = indoor | explore | confirm); every job runs all 21 cores (3 models x interventions)
# through xb_fixed.py on the trajectory TAG. noclobber claims; output = states/xb_a/<TAG>.jsonl + <TAG>.a1.json. Usage: xb_a_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=$R/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; W=$R/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
CORES="T1@none,T1@zero_wo,T1@shuffle_tok,T1@shuffle_pe,T1@drop_band,T1@drop_free,XA_SR@none,XA_SR@zero_wo,XA_SR@shuffle_tok,XA_SR@shuffle_pe,XA_SR@drop_band,XA_SR@drop_free,XA_MI@none,XA_MI@zero_wo,XA_MI@shuffle_tok,XA_MI@shuffle_pe,XA_MI@drop_band,XA_MI@drop_free,XA_MI@drop_f16,XA_MI@drop_f8,XA_MI@drop_local"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r set tag; do
  [ -z "$set" ] && continue; [[ $set == \#* ]] && continue
  [ -f $W/states/xb_a/$tag.jsonl ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/claims/xb_a/$tag.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $X/scripts/xb_fixed.py --set $set --group xb_a --ckpts $CORES --only $tag >> $X/logs/a/$tag.log 2>&1; rc=$?
  echo "exit $rc $set $tag $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $X/logs/a/queue.log
  [ $rc -ne 0 ] && mv $W/claims/xb_a/$tag.claim $W/claims/xb_a/$tag.claim.failed.$(date +%s)
done < $jf
