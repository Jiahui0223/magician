#!/bin/bash
# Experiment CT -- fixed-state worker (seen2scene env; trajectory claims balance the workers). Usage: ct_fixed_queue.sh GPU GROUP CKPTS [SETS]
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
g=$1; grp=$2; ck=$3; sets=${4:-"indoor explore confirm"}; host=$(hostname -s); mkdir -p /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed
for s in $sets; do
  CUDA_VISIBLE_DEVICES=$g LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $S2E/bin/python /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts/ct_fixed.py --set $s --group $grp --ckpts $ck >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed/${grp}_${host}_g${g}.log 2>&1
  echo "exit $? $grp $s $host g$g $(date +%FT%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training/logs/fixed_queue.log
done
