#!/bin/bash
# train_arm.sh : the single formal V5.2 run on the 4 GPUs of this node (torchrun, 4 ranks; resumable).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=$R/work_dirs/exp6_v5_2
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
mkdir -p $W/runs/v52 $W/logs
echo "start v52 $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD) tracked-dirty $(git diff HEAD --stat | wc -l) $(date '+%F %T')" >> $W/logs/train_v52.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v5_2/scripts/train_v5_2.py --out $W/runs/v52 >> $W/logs/train_v52.log 2>&1
echo "EXIT $? v52 $(hostname -s) $(date '+%F %T')" >> $W/logs/train_v52.log
