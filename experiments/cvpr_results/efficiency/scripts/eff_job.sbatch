#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/efficiency/logs/sbatch_%x_%j.out
# EXP8: sequential replay runs on ONE dedicated GPU (nothing else runs on it). Usage: sbatch -J NAME eff_job.sbatch PLANFILE
# PLANFILE lines: METHOD SRC SCENE START TMAX SUBDIR   (SUBDIR = runs | warmup)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
W=$R/work_dirs/cvpr_results/efficiency
echo "job $SLURM_JOB_ID node $(hostname -s) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES) start $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
while read -r m src s k tmax sub; do
  [ -z "$m" ] && continue
  echo "=== $(date '+%T') $m $src $s $k $tmax $sub"
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $R/experiments/cvpr_results/efficiency/scripts/eff_replay.py --method $m --src $src --scene $s --start $k --tmax $tmax --out $W/$sub 2>&1 | grep -v Warning | tail -3
done < $1
echo "end $(date '+%F %T')"
