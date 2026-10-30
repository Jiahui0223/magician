#!/bin/bash
# train_arm.sh t0|t1 : formal V3 run of one arm on the 4 GPUs of the node it runs on (torchrun, 4 ranks)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp6_v3
arm=$1; T=$([ $arm = t0 ] && echo task0_vae || echo task1_xattn)
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
echo "start $arm $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-interactive} $(date '+%F %T')" >> $W/$T/logs/train.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/$T/scripts/train.py --out $W/$T/runs/main >> $W/$T/logs/train.log 2>&1
echo "EXIT $? $arm $(hostname -s) $(date '+%F %T')" >> $W/$T/logs/train.log
