#!/bin/bash
# train_v31.sh : formal V3.1 Outdoor-only run on the 4 GPUs of the node it runs on (torchrun, 4 ranks); resumable
# (re-run the same command: the trainer resumes from runs/main/ckpt_last.pt). Tracked copy of the launcher.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp6_v3_1
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
mkdir -p $W/runs/main $W/logs
echo "start v3.1 $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-ssh} git $(git rev-parse --short HEAD) $(date '+%F %T')" >> $W/logs/train.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v3.1/scripts/train.py --out $W/runs/main >> $W/logs/train.log 2>&1
echo "EXIT $? v3.1 $(hostname -s) $(date '+%F %T')" >> $W/logs/train.log
