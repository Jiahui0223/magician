#!/bin/bash
# Coverage replay worker for all finished UAV episodes (claims: mkdir). Usage: bash uav_cov_queue.sh GPU
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
W7=$R/work_dirs/cvpr_results/exp7; CL=$W7/jobs/uav_cov_claims; mkdir -p $CL $W7/logs/uav_cov
export MACARONS_PLUSPLUS_ROOT=$W7/uav_macpp
for f in $W7/episodes/uavscenes/raw/OURS/V2M/episodes/ep_V2M_*.json $W7/episodes/uavscenes/raw/RANDOM/random/episodes/ep_random_*.json $W7/episodes/uavscenes/raw/MAG-B/MAG-B/episodes/ep_MAG-B_*.json; do
  [ -f $f ] || continue
  b=$(basename $f .json); case $b in ep_V2M_*) m=OURS; t=${b#ep_V2M_};; ep_random_*) m=RANDOM; t=${b#ep_random_};; ep_MAG-B_*) m=MAG-B; t=${b#ep_MAG-B_};; esac
  t=${t%_r0}; s=${t%_s*}; k=${t##*_s}
  case $m in OURS) grp=V2M;; RANDOM) grp=random;; MAG-B) grp=MAG-B;; esac
  [ -f $W7/episodes/uavscenes/metrics/$m/$grp/${grp}_${s}_s${k}_r0.json ] && continue
  mkdir $CL/${m}_${s}_s${k} 2>/dev/null || continue
  CUDA_VISIBLE_DEVICES=$1 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $R/experiments/cvpr_results/exp7/scripts/uavscenes/uav_cov.py --method $m --scene $s --start $k > $W7/logs/uav_cov/${m}_${s}_s${k}.log 2>&1
  echo "$(date '+%T') $m $s $k rc=$?" >> $W7/logs/uav_cov/worker_g$1.log
done
