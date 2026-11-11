#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/efficiency/logs/sbatch_%x_%j.out
# EXP8-MP3D: sequential indoor replay runs on ONE dedicated GPU. Usage: sbatch -J NAME eff_in_job.sbatch PLANFILE
# PLANFILE lines: METHOD SCAN LEVEL START TMAX SUBDIR   (SUBDIR = runs_in | warmup_in)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
W=$R/work_dirs/cvpr_results/efficiency
echo "job $SLURM_JOB_ID node $(hostname -s) gpu $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES) start $(date '+%F %T')"
while read -r m sc lv st tmax sub; do
  [ -z "$m" ] && continue
  echo "=== $(date '+%T') $m $sc $lv $st $tmax $sub"
  env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $R/experiments/cvpr_results/efficiency/scripts/eff_in_replay.py --method $m --scan $sc --level $lv --start $st --tmax $tmax --out $W/$sub 2>&1 | grep -v Warning | tail -3
done < $1
echo "end $(date '+%F %T')"
