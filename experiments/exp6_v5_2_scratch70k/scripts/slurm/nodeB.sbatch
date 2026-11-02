#!/bin/bash
#SBATCH -J v52m-nodeB
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:L40:4
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --no-requeue
#SBATCH -x gpu118
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v5_2_scratch70k/logs/nodeB_%j.out
# EXP6 V5.2-M Scratch70k -- second 4 x L40 node. Together with the 4 GPUs of the interactive allocation on gpu115
# (job 3443137) this gives the 8-GPU per-user QOS (gres/gpu=8). The node runs a control loop
# (scripts/nodeB_loop.sh): it joins every 2-node training round announced by overnight_driver.py as torchrun
# node_rank 1 (8 DDP ranks in total), and after training it runs G3 evaluation workers. No interactive session needed.
echo "$(date '+%F %T') node $(hostname) job $SLURM_JOB_ID GPUs $CUDA_VISIBLE_DEVICES cpus $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
bash /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp6_v5_2_scratch70k/scripts/nodeB_loop.sh
