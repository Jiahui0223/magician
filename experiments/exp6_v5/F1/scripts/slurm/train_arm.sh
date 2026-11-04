#!/bin/bash
# EXP6 V5 / F1 -- formal run of one arm (A | B) on the 4 GPUs of the node it runs on (torchrun, 4 ranks), inside an existing
# Slurm allocation (QOS 8-GPU limit, PROTOCOL §6). Resumable: re-run the same command (resumes from runs/<arm>/ckpt_last.pt).
# Log work_dirs/exp6_v5/F1/logs/train_<arm>.log
arm=$1
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=$R/work_dirs/exp6_v5/F1
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
mkdir -p $W/runs/$arm $W/logs
echo "start $arm $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD) dirty $(git status --porcelain | wc -l) $(date '+%F %T')" >> $W/logs/train_$arm.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v5/F1/scripts/train_f1.py --arm $arm --out $W/runs/$arm >> $W/logs/train_$arm.log 2>&1
echo "EXIT $? $arm $(hostname -s) $(date '+%F %T')" >> $W/logs/train_$arm.log
