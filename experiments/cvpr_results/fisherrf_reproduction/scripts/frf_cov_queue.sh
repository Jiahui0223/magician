#!/bin/bash
# FisherRF reproduction -- official coverage replay worker for all finished Macarons++ episodes (mkdir claims). Usage: bash frf_cov_queue.sh GPU
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
F=$R/work_dirs/cvpr_results/fisherrf_reproduction; CL=$F/jobs/cov_claims; mkdir -p $CL $F/logs/cov
for root in $F/outdoor/closed_loop/*_s[0-9]; do
  u=$(basename $root); s=${u%_s*}; k=${u##*_s}
  [ -f $root/episodes/ep_FisherRF_${u}_r0.json ] || continue
  [ -f $root/metrics/FisherRF/FisherRF_${u}_r0.json ] && continue
  mkdir $CL/$u 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=$1 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $R/experiments/cvpr_results/fisherrf_reproduction/scripts/fisherrf_out_cov.py --root $root --out $root/metrics --scene $s --start $k > $F/logs/cov/$u.log 2>&1
  echo "$(date '+%T') $u rc=$?" >> $F/logs/cov/worker_g$1.log
done
