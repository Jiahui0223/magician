#!/bin/bash
# EXP Density A/B -- G3 closed-loop worker v3 (same simulators / settings / outputs as da_g3_queue.sh; only scheduling and
# failure handling differ; FROZEN_PROTOCOL §9). Fixes of v2:
#   * the coverage replay of an outdoor episode always takes the claim <job>.cov first (noclobber): the episode's own worker
#     and a recovering worker can no longer run it concurrently (v2 race on the replay-frame dir)
#   * recovery of a job whose worker is gone (owner pid of the claim dead, no sim / cov process for that episode running):
#     outdoor episode file complete -> coverage only; otherwise -> release
#   * release = move every partial output of the episode and its claims to g3/failed/<cfg>_<tag>_tryN (nothing deleted);
#     a worker releases its own failed job right away; at most 3 tries per job, then GAVE_UP (logs/gave_up.log)
#   * at most ONE MP3D episode per GPU (per-GPU flock + process check, as v2); multi-pass over the job file
# Usage: da_g3_queue3.sh GPU JOBFILE [OUT_ROOT]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp_density_ab/scripts
E2T=$R/work_dirs/exp5_future_view/task3_full_depth_p1/e2/training
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); OUT=$(readlink -m ${3:-$R/work_dirs/exp_density_ab/g3})
mkdir -p $OUT/claims $OUT/logs $OUT/locks $OUT/failed; cd $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DA_G3_ROOT=$OUT
host=$(hostname -s)
LOG=logs/queue3_${host}_$g.log

indoor_busy() {
  for p in $(pgrep -f "da_in_sim.py --p1"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0
  done
  return 1
}
owner_alive() {   # claim file -> 0 if its worker is alive on this host
  IFS=: read h gg pid < $1 2>/dev/null
  [ "$h" != "$host" ] && return 0          # other node: assume alive
  kill -0 $pid 2>/dev/null
}
release() {   # $1 job line, $2 reason
  local it=$1 kind cfg b c d tag dst n
  IFS=: read kind cfg b c d <<< "$it"
  if [ $kind = cl_in ]; then tag=${b}_L${c}_S${d}; else tag=V2M_${b}_s${c}_r0; fi
  n=$(ls -d failed/${cfg}_${tag}_try* 2>/dev/null | wc -l)
  if [ $n -ge 3 ]; then echo "GAVE_UP $it $2 $(date '+%F %T')" >> logs/gave_up.log; return; fi
  dst=failed/${cfg}_${tag}_try$((n+1)); mkdir -p $dst
  if [ $kind = cl_in ]; then
    [ -d closed_loop/indoor/$cfg/V2M/$tag ] && mv closed_loop/indoor/$cfg/V2M/$tag $dst/
  else
    for sub in frames svc ckpt planner_logs episodes; do
      for f in closed_loop/outdoor/$cfg/V2M/$sub/*${tag}*; do [ -e "$f" ] && mkdir -p $dst/$sub && mv "$f" $dst/$sub/; done
    done
    [ -f metrics/outdoor/$cfg/V2M/$tag.json ] && mv metrics/outdoor/$cfg/V2M/$tag.json $dst/
    [ -d cov_tmp/$cfg/eval/outdoor/replay_frames/V2M_$tag ] && mv cov_tmp/$cfg/eval/outdoor/replay_frames/V2M_$tag $dst/replay_frames
  fi
  local cl=claims/$(echo $it | tr ':' '_').claim
  for f in $cl $cl.cov; do [ -f $f ] && mv $f $dst/; done
  echo "released $it ($2) -> $dst $(date '+%F %T')" >> logs/requeue.log
}
ep_complete() {   # outdoor episode + planner log complete?
  /bsuscratch/jiahuizhang/envs/seen2scene/bin/python - "$1" "$2" <<'PYEOF' 2>/dev/null
import json, sys
try:
    e = json.load(open(sys.argv[1])); p = json.load(open(sys.argv[2]))
    sys.exit(0 if len(e["executed_keys"]) == 101 and len(p["steps"]) == 100 else 1)
except Exception:
    sys.exit(1)
PYEOF
}
run_cov() {   # $1 job line -> rc
  local it=$1 kind cfg b c d
  IFS=: read kind cfg b c d <<< "$it"
  local cl=claims/$(echo $it | tr ':' '_').claim
  ( set -o noclobber; echo "$host:$g:$$" > $cl.cov ) 2>/dev/null || return 99
  CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_cov.py --cfg $cfg --scene $b --start $c > logs/cov_${cfg}_${b}_s${c}.log 2>&1 < /dev/null
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
    if [ $kind = cl_in ]; then
      tag=${b}_L${c}_S${d}; done_f=closed_loop/indoor/$cfg/V2M/$tag/episode.json
      simpat="da_in_sim.py --p1 $mode --scan $b --level $c --start $d --group V2M --out $OUT/closed_loop/indoor/$cfg\$"
    else
      tag=V2M_${b}_s${c}_r0; done_f=metrics/outdoor/$cfg/V2M/$tag.json
      ep_f=closed_loop/outdoor/$cfg/V2M/episodes/ep_$tag.json; pl_f=closed_loop/outdoor/$cfg/V2M/planner_logs/pl_$tag.json
      simpat="da_out_sim.py --p1 $mode --group V2M --scene $b --start $c --root $OUT/closed_loop/outdoor/$cfg/V2M\$"
    fi
    [ -f $done_f ] && continue
    if [ -f $claim ]; then
      # recovery of a job whose worker is gone
      owner_alive $claim && continue
      pgrep -f "$simpat" > /dev/null && continue
      [ $kind = cl_out ] && pgrep -f "da_out_cov.py --cfg $cfg --scene $b --start $c\$" > /dev/null && continue
      if [ $kind = cl_out ] && [ ! -f $claim.cov ] && ep_complete $ep_f $pl_f; then
        t0=$(date +%s); run_cov "$it"; rc=$?
        [ $rc = 99 ] && continue
        echo "exit $rc cov-recovery $it $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $LOG
        [ $rc -ne 0 ] && release "$it" "cov-recovery rc=$rc"
      else
        release "$it" "stale claim, worker gone"
      fi
      progress=1; continue
    fi
    if [ $kind = cl_in ]; then
      exec 9>locks/${host}_gpu$g.lock; flock 9
      if indoor_busy $g; then flock -u 9; exec 9>&-; pending=1; continue; fi
      ( set -o noclobber; echo "$host:$g:$$" > $claim ) 2>/dev/null || { flock -u 9; exec 9>&-; continue; }
      t0=$(date +%s)
      env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_in_sim.py --p1 $mode --scan $b --level $c --start $d --group V2M --out $OUT/closed_loop/indoor/$cfg > logs/cl_in_${cfg}_${tag}.log 2>&1 < /dev/null &
      pid=$!; sleep 15; flock -u 9; exec 9>&-
      wait $pid; rc=$?
      [ $rc -eq 0 ] && [ ! -f $done_f ] && rc=98
    else
      ( set -o noclobber; echo "$host:$g:$$" > $claim ) 2>/dev/null || continue
      t0=$(date +%s)
      env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_sim.py --p1 $mode --group V2M --scene $b --start $c --root $OUT/closed_loop/outdoor/$cfg/V2M > logs/cl_out_${cfg}_${b}_s${c}.log 2>&1 < /dev/null; rc=$?
      if [ $rc -eq 0 ]; then ep_complete $ep_f $pl_f || rc=97; fi
      if [ $rc -eq 0 ]; then run_cov "$it"; rc=$?; [ $rc = 99 ] && rc=0; fi
    fi
    echo "exit $rc $it $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $LOG
    [ $rc -ne 0 ] && release "$it" "rc=$rc"
    progress=1
  done < $jf
  [ $pending = 0 ] && [ $progress = 0 ] && break
  [ $progress = 0 ] && sleep 60
done
echo "worker end $host gpu$g $jf $(date '+%F %T')" >> $LOG
