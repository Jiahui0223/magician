#!/bin/bash
# Experiment CT -- counterfactual worker (seen2scene env; trajectory claims). Usage: ct_cf_queue.sh GPU GROUP CKPT
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; grp=$2; ck=$3; host=$(hostname -s); mkdir -p /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed
for s in indoor explore confirm; do
  CUDA_VISIBLE_DEVICES=$g LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $S2E/bin/python /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts/ct_cf.py --set $s --group $grp --ckpt $ck >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed/${grp}_${host}_g${g}.log 2>&1
  echo "exit $? $grp $s $host g$g $(date +%FT%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed_queue.log
done
