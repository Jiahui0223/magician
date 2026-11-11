#!/bin/bash
#SBATCH -J exp7_poolB
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:3
#SBATCH -c 36
#SBATCH --mem=240G
#SBATCH -t 14:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/exp7/logs/sbatch_%j.out
# EXP7 Pool B: extra MP3D-150 workers (UAVScenes has no GPU-ready unit yet). Per allocated GPU: one slot-A worker (V3 planners)
# and one slot-B worker (MAGICIAN / Random) over the shared job lists; mkdir claims prevent duplicates across pools.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
Q=$R/experiments/cvpr_results/exp7/scripts/mp3d_queue.sh
J=$R/work_dirs/cvpr_results/exp7/jobs
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "node $(hostname -s) gpus ${GPUS[*]} job $SLURM_JOB_ID $(date)"
for g in "${GPUS[@]}"; do
  bash $Q $g $J/mp3d150_slotA.txt B & sleep 5
  bash $Q $g $J/mp3d150_slotB.txt B & sleep 5
done
wait
echo "done $(date)"
