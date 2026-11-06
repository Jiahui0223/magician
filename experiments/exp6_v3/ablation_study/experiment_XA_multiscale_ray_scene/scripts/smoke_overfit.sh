#!/bin/bash
# Experiment XA -- engineering smoke: the trainer's tiny-overfit mode (fixed batch, 200 steps, 1 GPU) for the given variants, one after another. Usage: smoke_overfit.sh GPU ID [ID ...]
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; g=$1; shift
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
for id in "$@"; do
  mkdir -p /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/smoke/overfit_$id
  CUDA_VISIBLE_DEVICES=$g $S2E/bin/python /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/scripts/train_xa.py --id $id --out /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/smoke/overfit_$id --smoke overfit > /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/smoke/overfit_$id/train.log 2>&1
  echo "exit $? overfit $id $(hostname -s) g$g $(date +%FT%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/logs/smoke.log
done
