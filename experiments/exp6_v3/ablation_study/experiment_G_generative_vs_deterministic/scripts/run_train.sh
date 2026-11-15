#!/bin/bash
# Experiment G -- 4-GPU launcher of train_g.py with the environment of experiments/exp6_v3/shared/train_arm.sh (the V3-T0 launcher):
# same python env, LD_PRELOAD / LD_LIBRARY_PATH / PYTHONPATH / PYTORCH_CUDA_ALLOC_CONF / OMP_NUM_THREADS=6, torchrun --standalone, 4 ranks.
# Usage: run_train.sh HEAD RUN_DIR LOG [trainer args, e.g. --override steps=50]      (run detached with setsid nohup)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
G=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
head=$1; out=$2; log=$3; shift 3
cd $R
mkdir -p $out $(dirname $log)
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
echo "start head=$head $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-interactive} git $(git rev-parse --short HEAD) pid $$ $(date '+%F %T') args: $*" >> $log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 $G/scripts/train_g.py --head $head --out $out "$@" >> $log 2>&1
echo "EXIT $? head=$head $(hostname -s) $(date '+%F %T')" >> $log
