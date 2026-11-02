#!/bin/bash
#SBATCH -J v52m-cont1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:L40:4
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --no-requeue
#SBATCH -x gpu118
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v5_2_scratch70k/logs/cont1_%j.out
# Continuation of EXP6 V5.2-M Scratch70k on one 4-GPU L40S node: resumes from checkpoints/last.pt (global batch 64 kept,
# 4 ranks x 16 views) and runs the rest of the frozen chain (milestones, final G2, gate, G3, reports). If a node-B worker
# job (nodeB.sbatch) is alive, the driver expands to 8 ranks at a checkpoint.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R; echo "$(date '+%F %T') continuation on $(hostname -s) job $SLURM_JOB_ID"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
rm -f work_dirs/exp6_v5_2_scratch70k/ctl/STOP_DRIVER
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/exp6_v5_2_scratch70k/scripts/overnight_driver.py --head-job $SLURM_JOB_ID >> work_dirs/exp6_v5_2_scratch70k/logs/driver.log 2>&1
echo "$(date '+%F %T') driver exit $?"
