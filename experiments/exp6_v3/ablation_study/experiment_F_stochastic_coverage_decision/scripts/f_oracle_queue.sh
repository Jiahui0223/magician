#!/bin/bash
# Experiment F -- confirmation-layer GT counterfactuals with experiment E's frozen e_oracle.py (MAGICIAN env; no two-step). Outputs only into F's work dir.
# Job line: ID EPISODE OUT. Usage: f_oracle_queue.sh GPU JOBFILE
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/claims /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/oracle
while read -r id ep out; do
  [ -z "$id" ] && continue; [[ $id == \#* ]] && continue
  [ -f ${out%.npz}.json ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/claims/oracle_$id.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g $MG /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/scripts/e_oracle.py --episode $ep --out $out >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/oracle/$id.log 2>&1
  echo "exit $? $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/logs/oracle_queue.log
done < $jf
