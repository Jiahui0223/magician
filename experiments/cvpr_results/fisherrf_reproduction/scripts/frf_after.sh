#!/bin/bash
# Wait for all 70 Macarons++ FisherRF episodes, then run the official coverage replay: a 1-GPU sbatch job if it starts without
# preempting anyone (sbatch --test-only), else 2 workers on gpu114 GPU 3.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
F=$R/work_dirs/cvpr_results/fisherrf_reproduction; S=$R/experiments/cvpr_results/fisherrf_reproduction/scripts; LOG=$F/logs/frf_after.log
until [ $(ls $F/outdoor/closed_loop/*/episodes/ep_FisherRF_*.json 2>/dev/null | wc -l) -ge 70 ] || [ $(squeue -u $USER -h -o %j | grep -c '^frf_full') -eq 0 ]; do sleep 120; done
echo "$(date '+%F %T') outdoor episodes: $(ls $F/outdoor/closed_loop/*/episodes/ep_FisherRF_*.json | wc -l)" >> $LOG
cat > $F/jobs/cov.sbatch <<EOS
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:L40:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -x gpu118
#SBATCH --no-requeue
#SBATCH -o $F/logs/sbatch_frf_cov_%j.out
for w in 1 2 3 4; do bash $S/frf_cov_queue.sh \$CUDA_VISIBLE_DEVICES & sleep 5; done; wait
EOS
r=$(sbatch --test-only -J frf_cov $F/jobs/cov.sbatch 2>&1 | tail -1)
echo "$(date '+%T') test-only: $r" >> $LOG
if echo "$r" | grep -q Preempts; then
  for w in 1 2; do bash $S/frf_cov_queue.sh 3 & sleep 5; done; wait
  echo "$(date '+%T') replay done on gpu114 GPU 3" >> $LOG
else
  sbatch -J frf_cov $F/jobs/cov.sbatch >> $LOG 2>&1
fi
