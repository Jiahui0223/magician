#!/bin/bash
#SBATCH -J exp7_uav
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:3
#SBATCH -c 36
#SBATCH --mem=240G
#SBATCH -t 8:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/exp7/logs/sbatch_uav_%j.out
# EXP7 UAVScenes UAV-v0 closed loops: per allocated GPU 3 queue workers; each worker walks batch1 then batch2 (waits up to 2 h for
# batch2 to appear: tiles whose camera pre-selection finishes later). mkdir claims prevent duplicates across nodes.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
Q=$R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_queue.sh
J=$R/work_dirs/cvpr_results/exp7/jobs
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "node $(hostname -s) gpus ${GPUS[*]} job $SLURM_JOB_ID $(date)"
for g in "${GPUS[@]}"; do
  for w in 1 2 3; do
    ( bash $Q $g $J/uav_v0_batch1.txt B; for i in $(seq 1 240); do [ -f $J/uav_v0_batch2.txt ] && break; sleep 30; done; [ -f $J/uav_v0_batch2.txt ] && bash $Q $g $J/uav_v0_batch2.txt B ) &
    sleep 5
  done
done
wait
echo "done $(date)"
