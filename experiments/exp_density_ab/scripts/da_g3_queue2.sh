#!/bin/bash
# EXP Density A/B -- G3 closed-loop worker v2 (same simulators / settings / outputs as da_g3_queue.sh; only scheduling
# differs). Changes after the 2026-11-03 OOM (two MP3D planners of ~18-21 GB on one 44.5 GB GPU):
#   * at most ONE MP3D (cl_in) episode per GPU: before claiming a cl_in job the worker takes a per-GPU flock, checks that no
#     da_in_sim.py process with CUDA_VISIBLE_DEVICES=<gpu> is running, claims, starts the episode, releases the lock;
#     otherwise it skips the job for now (no claim) and takes an outdoor job instead
#   * multi-pass over the job file until every job is done or claimed
#   * cov-only recovery: an outdoor episode whose episode file exists but whose coverage file is missing (its worker was
#     stopped) gets the coverage replay
# Usage: [MOVES=n] da_g3_queue2.sh GPU JOBFILE [OUT_ROOT]   (job-line format as da_g3_queue.sh)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp_density_ab/scripts
E2T=$R/work_dirs/exp5_future_view/task3_full_depth_p1/e2/training
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); OUT=$(readlink -m ${3:-$R/work_dirs/exp_density_ab/g3})
mkdir -p $OUT/claims $OUT/logs $OUT/locks; cd $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DA_G3_ROOT=$OUT
host=$(hostname -s)

indoor_busy() {   # any MP3D episode running on GPU $1 of this node?
  for p in $(pgrep -f "da_in_sim.py --p1"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0
  done
  return 1
}

while true; do
  pending=0; progress=0
  while read -r it; do
    [ -z "$it" ] && continue
    IFS=: read kind cfg b c d <<< "$it"
    IFS=_ read ck inf sco <<< "$cfg"
    case $ck in
      B2) ckenv="" ;;
      E2S) ckenv="T3_CKPT=$E2T/s10/ema_005000.pt" ;;
      E2D) ckenv="T3_CKPT=$E2T/d100/ema_005000.pt" ;;
      *) echo "unknown ckpt $ck"; exit 2 ;;
    esac
    mode=$(echo ${inf}${sco} | tr 'A-Z' 'a-z')
    claim=claims/$(echo $it | tr ':' '_').claim
    case $kind in
      cl_in)  done_f=closed_loop/indoor/$cfg/V2M/${b}_L${c}_S${d}/episode.json ;;
      cl_out) done_f=metrics/outdoor/$cfg/V2M/V2M_${b}_s${c}_r0.json
              ep_f=closed_loop/outdoor/$cfg/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
    esac
    [ -f $done_f ] && continue
    if [ -f $claim ]; then
      # cov-only recovery for an outdoor episode whose worker was stopped after the closed loop
      if [ $kind = cl_out ] && [ -f $ep_f ] && ! pgrep -f "da_out_sim.py --p1 $mode --group V2M --scene $b --start $c " > /dev/null; then
        ( set -o noclobber; echo "$host:$g:$$" > $claim.cov ) 2>/dev/null || continue
        t0=$(date +%s)
        CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_cov.py --cfg $cfg --scene $b --start $c > logs/cov_${cfg}_${b}_s${c}.log 2>&1; rc=$?
        echo "exit $rc cov-only $it $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> logs/queue2_${host}_$g.log
        progress=1
      fi
      continue
    fi
    if [ $kind = cl_in ]; then
      exec 9>locks/${host}_gpu$g.lock; flock 9
      if indoor_busy $g; then flock -u 9; exec 9>&-; pending=1; continue; fi
      ( set -o noclobber; echo "$host:$g:$$" > $claim ) 2>/dev/null || { flock -u 9; exec 9>&-; continue; }
      t0=$(date +%s)
      env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_in_sim.py --p1 $mode --scan $b --level $c --start $d --group V2M ${MOVES:+--moves $MOVES} --out $OUT/closed_loop/indoor/$cfg > logs/cl_in_${cfg}_${b}_L${c}_S${d}.log 2>&1 &
      pid=$!; sleep 15; flock -u 9; exec 9>&-
      wait $pid; rc=$?
    else
      ( set -o noclobber; echo "$host:$g:$$" > $claim ) 2>/dev/null || continue
      t0=$(date +%s)
      env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_sim.py --p1 $mode --group V2M --scene $b --start $c ${MOVES:+--moves $MOVES} --root $OUT/closed_loop/outdoor/$cfg/V2M > logs/cl_out_${cfg}_${b}_s${c}.log 2>&1; rc=$?
      if [ $rc -eq 0 ] && [ -z "$MOVES" ]; then CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_cov.py --cfg $cfg --scene $b --start $c > logs/cov_${cfg}_${b}_s${c}.log 2>&1; rc=$?; fi
    fi
    echo "exit $rc $it $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> logs/queue2_${host}_$g.log
    progress=1
  done < $jf
  [ $pending = 0 ] && [ $progress = 0 ] && break
  [ $progress = 0 ] && sleep 60
done
echo "worker end $host gpu$g $jf $(date '+%F %T')" >> logs/queue2_${host}_$g.log
