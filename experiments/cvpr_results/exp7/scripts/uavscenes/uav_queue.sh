#!/bin/bash
# EXP7 UAVScenes queue worker (UAV-v0). One worker = one GPU slot; job list lines METHOD:SCENE:START (METHOD = OURS | RANDOM | MAG-B).
# mkdir claims (atomic across nodes), <= 3 tries, failed tries' partial outputs are left in place for inspection and the
# simulator's own resume logic (per-step checkpoints) continues them; a finished episode is never overwritten.
# Usage: bash uav_queue.sh GPU JOBLIST [POOL]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
G=$1; JL=$2; POOL=${3:-A}
W7=$R/work_dirs/cvpr_results/exp7
EP=$W7/episodes/uavscenes/raw
CL=$W7/jobs/uav_claims
mkdir -p $CL $W7/logs/uav_episodes
export MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
WLOG=$W7/logs/uav_worker_${POOL}_$(hostname -s)_g${G}_$(basename $JL .txt).log
echo "start $(date '+%F %T') host $(hostname -s) gpu $G joblist $JL pid $$" >> $WLOG
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read m s k <<< "$it"
  case $m in
    OURS)   root=$EP/OURS/V2M; done_f=$root/episodes/ep_V2M_${s}_s${k}_r0.json ;;
    RANDOM) root=$EP/RANDOM/random; done_f=$root/episodes/ep_random_${s}_s${k}_r0.json ;;
    MAG-B)  root=$EP/MAG-B/MAG-B; done_f=$root/episodes/ep_MAG-B_${s}_s${k}_r0.json ;;
    *) echo "unknown method $m" >> $WLOG; continue ;;
  esac
  [ -f $done_f ] && continue
  mkdir $CL/${m}_${s}_s${k} 2>/dev/null || continue
  echo "$(hostname -s):$G:$$:$POOL $(date '+%F %T')" > $CL/${m}_${s}_s${k}/owner
  for try in 1 2 3; do
    EL=$W7/logs/uav_episodes/${m}_${s}_s${k}.log
    echo "=== try $try $(date '+%F %T') host $(hostname -s) gpu $G" >> $EL
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$G /usr/bin/time -v $PY $R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_sim_out.py --method $m --scene $s --start $k --root $root >> $EL 2>&1
    rc=$?
    echo "exit $rc $it try $try wall $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WLOG
    [ -f $done_f ] && { echo "done $(date '+%F %T')" >> $CL/${m}_${s}_s${k}/owner; break; }
  done
  [ -f $done_f ] || echo "FAILED_3 $it" >> $WLOG
done < $JL
echo "end $(date '+%F %T')" >> $WLOG
