#!/bin/bash
#SBATCH -J v52m-cont2
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:L40:4
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --no-requeue
#SBATCH -x gpu118
#SBATCH -o /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v5_2_scratch70k/logs/cont2_%j.out
# Continuation of EXP6 V5.2-M Scratch70k on 2 x 4 L40S (8 GPU / 60 CPU; needs the whole per-user GPU QOS): node 0 runs the
# driver (rank 0-3, master on its ib0 address), node 1 runs nodeB_loop.sh (rank 4-7, evaluation tasks). Same chain.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; C=$R/work_dirs/exp6_v5_2_scratch70k/ctl
cd $R; echo "$(date '+%F %T') continuation on $SLURM_JOB_NODELIST job $SLURM_JOB_ID"
rm -f $C/SHUTDOWN $C/STOP_DRIVER $C/nodeB_status.json
srun --nodes=1 --ntasks=1 --relative=1 --exclusive bash experiments/exp6_v5_2_scratch70k/scripts/nodeB_loop.sh > work_dirs/exp6_v5_2_scratch70k/logs/nodeB_cont2_$SLURM_JOB_ID.out 2>&1 &
for i in $(seq 30); do [ -f $C/nodeB_status.json ] && break; sleep 10; done
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/exp6_v5_2_scratch70k/scripts/overnight_driver.py --head-job $SLURM_JOB_ID >> work_dirs/exp6_v5_2_scratch70k/logs/driver.log 2>&1
echo "$(date '+%F %T') driver exit $?"; touch $C/SHUTDOWN; wait
