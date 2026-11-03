#!/bin/bash
#SBATCH -J da_g3
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:4
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH -t 12:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp_density_ab/logs/sbatch_%j.out
# EXP Density A/B -- extra G3 workers on a second 4-GPU node: 2 queue workers per allocated GPU over the same job files
# (claims in the shared output dir prevent duplicates). Usage: sbatch g3_node.sbatch JOBFILE [JOBFILE2]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
Q=$R/experiments/exp_density_ab/scripts/da_g3_queue.sh
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "node $(hostname -s) gpus ${GPUS[*]} $(date)"
for jf in "$@"; do
  for g in "${GPUS[@]}"; do bash $Q $g $jf & bash $Q $g $jf & done
  wait
done
echo "done $(date)"
