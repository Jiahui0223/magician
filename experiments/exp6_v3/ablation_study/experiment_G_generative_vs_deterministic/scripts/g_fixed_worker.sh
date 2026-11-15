#!/bin/bash
# Experiment G fixed-state worker (queue task; GPU = CUDA_VISIBLE_DEVICES): g_fixed.py over the confirm then the explore set (protocol G-v1 s7).
# Trajectories are claimed with O_EXCL inside g_fixed.py, so any number of workers share the pool. Usage: g_fixed_worker.sh WID [EXTRA ARGS]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
L=$R/work_dirs/exp6_v3/experiment_G/fixed/logs; mkdir -p $L
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
w=$1; shift; H=$(hostname -s); rc_all=0
for s in confirm explore; do
  echo "START $s $H gpu$CUDA_VISIBLE_DEVICES w$w $(date '+%F %T')" >> $L/worker_chain.log
  $S2E/bin/python $GD/scripts/g_fixed.py --set $s "$@" >> $L/fixed_${H}_g${CUDA_VISIBLE_DEVICES}_w${w}.log 2>&1; rc=$?
  echo "END $s rc $rc $H gpu$CUDA_VISIBLE_DEVICES w$w $(date '+%F %T')" >> $L/worker_chain.log
  [ $rc = 0 ] || rc_all=$rc
done
exit $rc_all
