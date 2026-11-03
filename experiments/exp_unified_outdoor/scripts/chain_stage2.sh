#!/bin/bash
# EXP Unified Outdoor -- unattended Stage-2 chain on gpu115 (UNIFIED_PROTOCOL.md §4). Waits for the Stage-1 chain to end
# (S1_G3_DONE or BLOCKED), then: V4-S10 60k (with one deliberate interrupt + resume after the first full checkpoint) and,
# in parallel, the fixed-state G1/G2 of V2-S10 (reused V2 Original EMA@60k); then V4C-12k and V5-warm-12k from V4-S10@48k;
# then the S10 TSDF cache; then the V3-T1 profile smoke and the V3-T1-S10 formal run. A failed step marks its dependants
# BLOCKED (logged) and the chain moves on to independent steps only. Log: work_dirs/exp_unified_outdoor/logs/chain.log
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=experiments/exp_unified_outdoor/scripts; W=work_dirs/exp_unified_outdoor; L=$W/logs; mkdir -p $L
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
log() { echo "$1 $(date '+%F %T')" >> $L/chain.log; }
export CUDA_VISIBLE_DEVICES=0,1,2,3
run_retry() {   # run_arm.sh with up to 3 attempts (each attempt resumes from the last full checkpoint); exit 3 = parent missing
  local arm=$1 i rc=1; shift
  for i in 1 2 3; do
    bash $X/run_arm.sh $arm "$@" > $L/run_${arm}_try$i.out 2>&1; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && return 3
    log "$arm attempt $i failed rc=$rc"; sleep 30
  done
  return $rc
}
until grep -q "S1_G3_DONE\|S1 BLOCKED" work_dirs/exp_temporal_refresh/logs/chain.log 2>/dev/null; do sleep 60; done
log "S2 start (Stage 1 chain: $(tail -n 1 work_dirs/exp_temporal_refresh/logs/chain.log))"

# ---- V2-S10 (reused) fixed-state evaluation, 1 process per GPU, in parallel with V4 training
V2CK=$R/work_dirs/exp5_future_view/task1/training/main/ema_060000.pt
( export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
  E=$X/uo_fixed.py
  CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $E --name V2S10_60k --ckpt $V2CK --set valsel > $L/eval_V2S10_valsel.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $S2E/bin/python $E --name V2S10_60k --ckpt $V2CK --set dev3 --stratum B0 --part 0/2 > $L/eval_V2S10_dev3_B0_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $S2E/bin/python $E --name V2S10_60k --ckpt $V2CK --set dev3 --stratum B0 --part 1/2 > $L/eval_V2S10_dev3_B0_1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $S2E/bin/python $E --name V2S10_60k --ckpt $V2CK --set dev3 --stratum B2 > $L/eval_V2S10_dev3_B2.log 2>&1 &
  wait; log "eval V2S10_60k fixed done: $(find $W/eval_fixed/V2S10_60k -name '*.jsonl' | wc -l) / 39 episodes" ) &
EVAL_PID=$!

# ---- V4-S10 with a resume test after the first full checkpoint (step 1000)
log "V4-S10 start"
bash $X/run_arm.sh v4_s10 > $L/run_v4_s10_a.out 2>&1 &
RUN=$!
until [ $(ls $W/runs/v4_s10/rng_states/s0001000_r*.pt 2>/dev/null | wc -l) -ge 4 ] || ! kill -0 $RUN 2>/dev/null; do sleep 20; done
if kill -0 $RUN 2>/dev/null; then
  sleep 60
  pkill -TERM -f "train_unified.py --arm v4_s10" ; wait $RUN
  log "V4-S10 deliberate interrupt after step 1000 (rc $?) -> resume"
fi
run_retry v4_s10; rc=$?
log "V4-S10 end rc=$rc exact-resume: $(cat $W/runs/v4_s10/resume_log_r*.jsonl 2>/dev/null | grep -c '"exact": true')/$(cat $W/runs/v4_s10/resume_log_r*.jsonl 2>/dev/null | wc -l) rank records exact"
wait $EVAL_PID
if [ $rc -ne 0 ] || [ ! -f $W/runs/v4_s10/ema_060000.pt ]; then
  log "V4-S10 FAILED -> V4C / V5 BLOCKED"
else
  for arm in v4c12k_from48k_s10 v5warm12k_from48k_s10; do
    log "$arm start"; run_retry $arm; log "$arm end rc=$?"
  done
fi
# ---- V3-T1-S10: cache, profile smoke, formal run
log "S10 TSDF cache start"; bash $X/build_tsdf_cache_s10.sh > $L/cache_s10.out 2>&1; rc=$?; log "S10 TSDF cache end rc=$rc"
if [ $rc -ne 0 ]; then log "V3-T1-S10 BLOCKED (cache)"; exit 1; fi
UO_SMOKE_SCENE=dunnottar bash $X/run_arm.sh v3t1_s10 --smoke profile > $L/run_v3t1_s10_profile.out 2>&1; rc=$?
log "V3-T1-S10 profile smoke rc=$rc"
if [ $rc -ne 0 ]; then log "V3-T1-S10 BLOCKED (profile smoke failed; sampler / memory decision needed)"; exit 1; fi
log "V3-T1-S10 formal start"; run_retry v3t1_s10; log "V3-T1-S10 end rc=$?"
