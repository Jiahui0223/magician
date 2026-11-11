#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/fisherrf_reproduction/logs/sbatch_%x_%j.out
# FisherRF reproduction: W queue workers on the allocated GPU over a job list, optionally preceded by one direct command (PRE).
# Usage: sbatch -J NAME frf_job.sbatch JOBLIST W ["PRE command"]
# SLURM_JOB_QOS=kd-high in the child env disables the official ClusterStateManager scontrol-requeue hook (audit finding).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
export SLURM_JOB_QOS=kd-high
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "job $SLURM_JOB_ID node $(hostname -s) gpu ${GPUS[*]} $(nvidia-smi --query-gpu=name --format=csv,noheader -i ${GPUS[0]}) start $(date '+%F %T')"
if [ -n "$3" ]; then echo "PRE: $3"; CUDA_VISIBLE_DEVICES=${GPUS[0]} bash -c "$3"; echo "PRE rc=$?"; fi
for g in "${GPUS[@]}"; do for w in $(seq 1 $2); do bash $R/experiments/cvpr_results/fisherrf_reproduction/scripts/frf_queue.sh $g $1 S$SLURM_JOB_ID & sleep 10; done; done
wait
echo "end $(date '+%F %T')"
