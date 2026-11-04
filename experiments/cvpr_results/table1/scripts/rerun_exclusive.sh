#!/bin/bash
# CVPR Table 1 -- exclusive phase for episodes that need a whole GPU (b2 / full P1 twin maps on the largest scene,
# dunnottar, OOM next to 2-3 other planners). Waits until the regular workers of BOTH nodes have ended, then runs the jobs
# of JOBFILE one at a time on GPU g (one exclusive worker per GPU). Same launch chain, coverage replay and import as
# t1_queue.sh (functions sourced from it); before a run, the old reservation / gave-up claim and any partial outputs are
# moved to g3/failed/<model>_<tag>_excl_pre (nothing deleted). Single attempt per job; results logged.
# Usage: rerun_exclusive.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
GPU_ARG=$1; JF=$(readlink -f $2)
source <(sed -n '/^EXP5=/,/^release()/p' experiments/cvpr_results/table1/scripts/t1_queue.sh | sed '$d')
g=$GPU_ARG; host=$(hostname -s)
T1W=$R/work_dirs/cvpr_results/table1/g3; L=$T1W/logs
until grep -q "^T1_RUN_DONE " $L/chain.log 2>/dev/null && grep -q "^T1_RUN_DONE_gpu114" $L/chain.log 2>/dev/null; do sleep 60; done
while IFS=: read -r m s k; do
  [ -z "$m" ] && continue
  [ -f $R/work_dirs/cvpr_results/table1/episodes/$m/${s}_s${k}.json ] && continue
  ( set -o noclobber; echo "$host:$g:$$" > $T1W/claims/excl_${m}_${s}_${k}.claim ) 2>/dev/null || continue
  G=$(group_of $m); tag=${G}_${s}_s${k}_r0; root=$T1W/closed_loop/outdoor/$m/$G; dst=$T1W/failed/${m}_${tag}_excl_pre; mkdir -p $dst
  mv $T1W/claims/${m}_${s}_${k}.claim $dst/ 2>/dev/null
  for sub in frames svc ckpt planner_logs episodes; do for f in $root/$sub/*${tag}*; do [ -e "$f" ] && mkdir -p $dst/$sub && mv "$f" $dst/$sub/; done; done
  [ -f $T1W/metrics/outdoor/$m/$G/$tag.json ] && mv $T1W/metrics/outdoor/$m/$G/$tag.json $dst/
  echo "$host:$g:$$ exclusive" > $T1W/claims/${m}_${s}_${k}.claim
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g run_sim $m $s $k $root > $L/sim_${m}_${s}_s${k}_excl.log 2>&1 < /dev/null; rc=$?
  [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g $MG $SC/t1_out_cov.py --model $m --group $G --scene $s --start $k > $L/cov_${m}_${s}_s${k}_excl.log 2>&1 < /dev/null; rc=$?; }
  [ $rc -eq 0 ] && { $PY $SC/import_episodes.py --new $m $s $k >> $L/import.log 2>&1; rc=$?; }
  echo "exit $rc EXCL $m:$s:$k $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $L/exclusive.log
done < $JF
echo "EXCL_WORKER_END $host gpu$g $(date '+%F %T')" >> $L/exclusive.log
