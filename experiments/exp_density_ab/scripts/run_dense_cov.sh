#!/bin/bash
# EXP Density A/B -- dense-coverage sensitivity check: 2 workers per GPU over configs/dense_cov_jobs.txt (cfg:scene:start);
# noclobber claims; finished episodes (output json) are skipped. Usage: bash run_dense_cov.sh
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
D=work_dirs/exp_density_ab/dense_cov; mkdir -p $D/claims $D/logs
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
worker() {
  local g=$1
  while IFS=: read -r cfg s k; do
    [ -f $D/$cfg/V2M_${s}_s${k}_r0.json ] && continue
    ( set -o noclobber; echo "$g:$$" > $D/claims/${cfg}_${s}_${k}.claim ) 2>/dev/null || continue
    CUDA_VISIBLE_DEVICES=$g $MG experiments/exp_density_ab/scripts/da_dense_cov.py --cfg $cfg --scene $s --start $k >> $D/logs/worker_${g}_$$.log 2>&1
    echo "exit $? $cfg $s $k $(date '+%T')" >> $D/logs/queue.log
  done < experiments/exp_density_ab/configs/dense_cov_jobs.txt
}
for g in 0 1 2 3; do worker $g & sleep 3; worker $g & sleep 3; done
wait
echo "DENSE_DONE $(date '+%F %T') $(find $D -name 'V2M_*.json' | wc -l)" >> $D/logs/queue.log
