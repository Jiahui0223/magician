#!/bin/bash
# FisherRF-full (Stage F1c = v3 + MAGICIAN pose-space frontier selection (FRF_FRONTIER_MODE=magician); claims3, root closed_loop_f1c) -- queue worker for one GPU slot: closed loop (fisherrf_sim_out.py --planner fisherrf_full_planner.py,
# D12 docstring init) followed by the official coverage replay (fisherrf_out_cov.py = Table 1 t25 procedure) of the same episode.
# Job list lines: <scene>:<start>[:<moves>]. Output root: work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop/<scene>_s<k>
# (the D3a runs under .../outdoor/closed_loop are never touched). A failed try is moved to full/_failed/ and re-run from t = 0 (<= 2 tries).
# Usage: bash frf_full_queue.sh GPU JOBLIST [POOL]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
G=$1; JL=$2; POOL=${3:-A}
F=$R/work_dirs/cvpr_results/fisherrf_reproduction/full
S=$R/experiments/cvpr_results/fisherrf_reproduction/scripts
CL=$F/jobs/claims3; mkdir -p $CL $F/logs/episodes_f1c $F/_failed_f1c
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
WLOG=$F/logs/worker3_${POOL}_$(hostname -s)_g${G}.log
until [ -f $F/jobs/GO_f1c ]; do sleep 10; done          # F1c smoke passed
echo "start $(date '+%F %T') host $(hostname -s) gpu $G joblist $JL pid $$" >> $WLOG
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read s k mv_ <<< "$it"; mv_=${mv_:-100}
  u=${s}_s${k}; root=$F/outdoor/closed_loop_f1c/$u; ep=$root/episodes/ep_FisherRF_${u}_r0.json; met=$root/metrics/FisherRF/FisherRF_${u}_r0.json
  [ -f $met ] && continue
  mkdir $CL/$u 2>/dev/null || continue
  echo "$(hostname -s):$G:$$:$POOL $(date '+%F %T')" > $CL/$u/owner
  EL=$F/logs/episodes_f1c/$u.log
  for try in 1 2; do
    [ -f $ep ] && break
    [ -d $root ] && { mv $root $F/_failed_f1c/${u}_try${try}_$(date +%s); }
    echo "=== try $try $(date '+%F %T') host $(hostname -s) gpu $G" >> $EL
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$G SLURM_JOB_QOS=kd-high FRF_FRONTIER_MODE=magician /usr/bin/time -v $MG $S/fisherrf_sim_out.py --scene $s --start $k --moves $mv_ --root $root \
      --init-pose docstring --verify-every 0 --planner fisherrf_full_planner.py >> $EL 2>&1
    echo "loop exit $? $u try $try wall $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WLOG
  done
  if [ -f $ep ]; then
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$G $MG $S/fisherrf_out_cov.py --root $root --out $root/metrics --scene $s --start $k > $F/logs/episodes_f1c/${u}_cov.log 2>&1
    echo "cov exit $? $u wall $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WLOG
  else
    echo "FAILED_2 $it" >> $WLOG
  fi
  echo "done $(date '+%F %T')" >> $CL/$u/owner
done < $JL
echo "end $(date '+%F %T')" >> $WLOG
