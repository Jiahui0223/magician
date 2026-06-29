#!/bin/bash
#SBATCH -J magician          # job name
#SBATCH -o jiahui/log/magician_%j.out  # stdout -> jiahui/log/ (在项目根目录下 sbatch；%j = job ID)
#SBATCH -e jiahui/log/magician_%j.err  # stderr 单独一个文件（报错/Traceback 会在这里）
#SBATCH -n 1                 # total tasks
#SBATCH -c 48                # CPU cores
#SBATCH -N 1                 # nodes
#SBATCH -p gpu-l40           # L40 partition (4x L40, 64 cores/node, max 7 days)
#SBATCH -t 3-00:00:00        # walltime: days-hh:mm:ss
#SBATCH --gres=gpu:L40:4     # request 4 L40 GPUs

# Activate the environment
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh
conda activate /bsuscratch/jiahuizhang/envs/magician

# Check GPU status
nvidia-smi

# Run
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
python -u test_magician_planning.py   # -u: 不缓冲，进度实时写进 .out
