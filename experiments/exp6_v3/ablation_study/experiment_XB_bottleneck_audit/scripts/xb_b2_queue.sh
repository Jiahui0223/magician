#!/bin/bash
# Experiment XB / B2 -- counterfactual worker (one GPU): job lines "SET TAG CKPT" through xb_cf.py; claims; output states/xb_cf/<TAG>__<CKPT>.jsonl. Usage: xb_b2_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R; X=$R/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; W=$R/work_dirs/exp6_v3/ablation_study/experiment_XB_bottleneck_audit; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r set tag ck; do
  [ -z "$set" ] && continue; [[ $set == \#* ]] && continue
  od=$W/states/xb_cf/$ck; mkdir -p $od
  [ -f $od/$tag.jsonl ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/claims/xb_cf/${tag}__$ck.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $X/scripts/xb_cf.py --set $set --group xb_cf/$ck --ckpt $ck --only $tag --out $od >> $X/logs/b2/${tag}__$ck.log 2>&1; rc=$?
  echo "exit $rc $set $tag $ck $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $X/logs/b2/queue.log
  [ $rc -ne 0 ] && mv $W/claims/xb_cf/${tag}__$ck.claim $W/claims/xb_cf/${tag}__$ck.claim.failed.$(date +%s)
done < $jf
