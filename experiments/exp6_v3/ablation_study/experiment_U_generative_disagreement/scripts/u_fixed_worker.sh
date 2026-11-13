#!/bin/bash
# Experiment U Phase D fixed-state worker (queue task; GPU = CUDA_VISIBLE_DEVICES): u_fixed.py over the confirm then the explore set.
# Trajectories are claimed with O_EXCL inside u_fixed.py, so any number of workers share the pool. Usage: u_fixed_worker.sh WID [EXTRA ARGS]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
U=$R/experiments/exp6_v3/ablation_study/experiment_U_generative_disagreement
L=$R/work_dirs/exp6_v3/experiment_U/fixed/logs; mkdir -p $L
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
w=$1; shift; H=$(hostname -s)
for s in confirm explore; do
  echo "START $s $H gpu$CUDA_VISIBLE_DEVICES w$w $(date '+%F %T')" >> $L/worker_chain.log
  $S2E/bin/python $U/scripts/u_fixed.py --set $s "$@" >> $L/fixed_${H}_g${CUDA_VISIBLE_DEVICES}_w${w}.log 2>&1
  echo "END $s rc $? $H gpu$CUDA_VISIBLE_DEVICES w$w $(date '+%F %T')" >> $L/worker_chain.log
done
