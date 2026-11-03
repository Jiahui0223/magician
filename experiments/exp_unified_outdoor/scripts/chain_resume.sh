#!/bin/bash
# EXP Unified Outdoor -- RESUME chain after the 2026-11-03 14:06 pause (PAUSED_RESTART.md). Same order and settings as
# chain_stage2.sh, WITHOUT the one-off deliberate-interrupt test and without waiting for Stage 1:
#   V4-S10 (resumes from runs/v4_s10/ckpt_last.pt, step 38000, exact per-rank RNG) -> [V4-S10 fixed-state eval in parallel
#   with] V4C-12k-S10 -> [V4C eval in parallel with] V5-warm-12k-S10 -> S10 TSDF cache check -> V3-T1-S10 profile smoke ->
#   V3-T1-S10 formal. Every run_arm.sh call resumes automatically and exits 0 at once if the arm is already finished.
# Usage (4 GPUs): CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash experiments/exp_unified_outdoor/scripts/chain_resume.sh &
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=experiments/exp_unified_outdoor/scripts; W=work_dirs/exp_unified_outdoor; L=$W/logs; mkdir -p $L
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
log() { echo "$1 $(date '+%F %T')" >> $L/chain.log; }
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to 4 GPU ids}"
IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"
run_retry() {
  local arm=$1 i rc=1; shift
  for i in 1 2 3; do
    bash $X/run_arm.sh $arm "$@" > $L/run_${arm}_resume_try$i.out 2>&1; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && return 3
    log "$arm attempt $i failed rc=$rc"; sleep 30
  done
  return $rc
}
ev() {   # fixed-state G1/G2 of a V2Model-compatible checkpoint, one process per GPU
  ( export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
    E=$X/uo_fixed.py
    CUDA_VISIBLE_DEVICES=${G[0]} $S2E/bin/python $E --name $1 --ckpt $2 --set valsel > $L/eval_$1_valsel.log 2>&1 &
    CUDA_VISIBLE_DEVICES=${G[1]} $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B0 --part 0/2 > $L/eval_$1_dev3_B0_0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=${G[2]} $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B0 --part 1/2 > $L/eval_$1_dev3_B0_1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=${G[3]} $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B2 > $L/eval_$1_dev3_B2.log 2>&1 &
    wait; log "eval $1 fixed done: $(find $W/eval_fixed/$1 -name '*.jsonl' | wc -l) / 39 episodes" )
}
log "RESUME chain start (GPUs $CUDA_VISIBLE_DEVICES)"
log "V4-S10 resume start"; run_retry v4_s10; rc=$?; log "V4-S10 end rc=$rc"
if [ $rc -ne 0 ] || [ ! -f $W/runs/v4_s10/ema_060000.pt ]; then log "V4-S10 FAILED -> V4C / V5 / evals BLOCKED"; else
  ev V4S10_60k $R/$W/runs/v4_s10/ema_060000.pt & EV1=$!
  log "v4c12k_from48k_s10 start"; run_retry v4c12k_from48k_s10; rc=$?; log "v4c12k_from48k_s10 end rc=$rc"; wait $EV1
  [ $rc -eq 0 ] && { ev V4C12k_S10 $R/$W/runs/v4c12k_from48k_s10/ema_012000.pt & EV2=$!; }
  log "v5warm12k_from48k_s10 start"; run_retry v5warm12k_from48k_s10; log "v5warm12k_from48k_s10 end rc=$?"; [ -n "${EV2:-}" ] && wait $EV2
fi
log "S10 TSDF cache check"; bash $X/build_tsdf_cache_s10.sh > $L/cache_s10_resume.out 2>&1; rc=$?; log "S10 TSDF cache rc=$rc ($(find $W/cache/tsdf4_s10/outdoor -name '*.npz' | wc -l)/180)"
[ $rc -ne 0 ] && { log "V3-T1-S10 BLOCKED (cache)"; exit 1; }
UO_SMOKE_SCENE=dunnottar bash $X/run_arm.sh v3t1_s10 --smoke profile > $L/run_v3t1_s10_profile.out 2>&1; rc=$?; log "V3-T1-S10 profile smoke rc=$rc"
[ $rc -ne 0 ] && { log "V3-T1-S10 BLOCKED (profile smoke failed)"; exit 1; }
log "V3-T1-S10 formal start"; run_retry v3t1_s10; log "V3-T1-S10 end rc=$?"
