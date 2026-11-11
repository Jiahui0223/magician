#!/bin/bash
# FisherRF reproduction -- queue worker (one GPU slot). Job list lines:  OUT:<scene>:<start>[:<moves>]   or   IN:<scan>:<level>:<start>[:<moves>]
# Decisions D12 = --init-pose docstring, D13 handled in fisherrf_planner.py. FisherRF runs are not bitwise reproducible (R1), so a failed
# try is never resumed: its whole output folder is moved to _failed/ and the episode restarts from t = 0 (<= 2 tries).
# Outdoor: one closed-loop root per episode (the planner's official mapping PNGs go to <root>/planner_work).
# Usage: bash frf_queue.sh GPU JOBLIST [POOL]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
G=$1; JL=$2; POOL=${3:-A}
F=$R/work_dirs/cvpr_results/fisherrf_reproduction
S=$R/experiments/cvpr_results/fisherrf_reproduction/scripts
CL=$F/jobs/claims; mkdir -p $CL $F/logs $F/_failed
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
WLOG=$F/logs/worker_${POOL}_$(hostname -s)_g${G}_$(basename $JL .txt).log
echo "start $(date '+%F %T') host $(hostname -s) gpu $G joblist $JL pid $$" >> $WLOG
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind a b c d <<< "$it"
  if [ "$kind" = OUT ]; then
    s=$a; k=$b; mv_=${c:-100}; u=OUT_${s}_s${k}; root=$F/outdoor/closed_loop/${s}_s${k}; done_f=$root/episodes/ep_FisherRF_${s}_s${k}_r0.json
  else
    sc=$a; lv=$b; k=$c; mv_=${d:-150}; u=IN_${sc}_L${lv}_S${k}; root=$F/indoor/closed_loop/FisherRF/${sc}_L${lv}_S${k}; done_f=$root/episode.json
  fi
  [ -f $done_f ] && continue
  mkdir $CL/$u 2>/dev/null || continue
  echo "$(hostname -s):$G:$$:$POOL $(date '+%F %T')" > $CL/$u/owner
  for try in 1 2; do
    [ -d $root ] && [ ! -f $done_f ] && { mkdir -p $F/_failed/$kind; mv $root $F/_failed/$kind/${u}_try${try}_$(date +%s); }
    EL=$F/logs/episodes/${u}.log; mkdir -p $(dirname $EL)
    echo "=== try $try $(date '+%F %T') host $(hostname -s) gpu $G" >> $EL
    t0=$(date +%s)
    if [ "$kind" = OUT ]; then
      CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $S/fisherrf_sim_out.py --scene $s --start $k --moves $mv_ --root $root --init-pose docstring --verify-every 0 >> $EL 2>&1
    else
      CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $MG $S/fisherrf_sim_in.py --scan $sc --level $lv --start $k --moves $mv_ --out $F/indoor/closed_loop --init-pose docstring --verify-every 0 >> $EL 2>&1
    fi
    rc=$?
    echo "exit $rc $it try $try wall $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WLOG
    [ -f $done_f ] && { echo "done $(date '+%F %T')" >> $CL/$u/owner; break; }
  done
  [ -f $done_f ] || echo "FAILED_2 $it" >> $WLOG
done < $JL
echo "end $(date '+%F %T')" >> $WLOG
