#!/bin/bash
# build_tsdf_cache_s10.sh : build the S10 4x partial-TSDF cache (V3-T1 S10 arm input) with one worker per GPU in
# CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 -> 4 workers). Workers claim trajectories (resumable: finished
# trajectories are skipped; delete stale work_dirs/exp_unified_outdoor/cache/tsdf4_s10/claims/*.claim of killed workers).
# Exit code: non-zero if any worker failed or fewer than 180 trajectories exist at the end.
set -u
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp_unified_outdoor
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "set CUDA_VISIBLE_DEVICES" >&2; exit 2; fi
cd $R || exit 2
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=6
mkdir -p $W/logs
IFS=',' read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
N=${#G[@]}
pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $S2E/bin/python experiments/exp_unified_outdoor/scripts/build_tsdf_cache_s10.py $i $N --device cuda \
    >> $W/logs/cache_s10_w$i.log 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
NC=$(find $W/cache/tsdf4_s10/outdoor -name '*.npz' | wc -l)
echo "S10 cache: $NC / 180 trajectories $(date '+%F %T')" | tee -a $W/logs/cache_s10.log
[ $rc -eq 0 ] && [ "$NC" -ge 180 ] || exit 1
exit 0
