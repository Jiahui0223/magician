#!/bin/bash
# EXP Unified Outdoor -- fixed-state G1/G2 (uo_fixed.py, UNIFIED_PROTOCOL §5) of each finished V2Model-compatible arm,
# started as soon as its final EMA exists (runs next to the following training arm; one eval process per GPU).
# V4-S10 (ema_060000) then V4C-12k-S10 (ema_012000). V5 / V3 need their own inference cores -> not covered here.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=work_dirs/exp_unified_outdoor; L=$W/logs
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
log() { echo "$1 $(date '+%F %T')" >> $L/chain.log; }
ev() {   # name ckpt
  local E=experiments/exp_unified_outdoor/scripts/uo_fixed.py
  CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $E --name $1 --ckpt $2 --set valsel > $L/eval_$1_valsel.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B0 --part 0/2 > $L/eval_$1_dev3_B0_0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B0 --part 1/2 > $L/eval_$1_dev3_B0_1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $S2E/bin/python $E --name $1 --ckpt $2 --set dev3 --stratum B2 > $L/eval_$1_dev3_B2.log 2>&1 &
  wait
  log "eval $1 fixed done: $(find $W/eval_fixed/$1 -name '*.jsonl' | wc -l) / 39 episodes"
}
until [ -f $W/runs/v4_s10/ema_060000.pt ] && grep -q "V4-S10 end" $L/chain.log; do sleep 120; done
grep -q "V4-S10 end rc=0" $L/chain.log && ev V4S10_60k $R/$W/runs/v4_s10/ema_060000.pt
until [ -f $W/runs/v4c12k_from48k_s10/ema_012000.pt ] && grep -q "v4c12k_from48k_s10 end" $L/chain.log || grep -q "V4C / V5 BLOCKED" $L/chain.log; do sleep 120; done
grep -q "v4c12k_from48k_s10 end rc=0" $L/chain.log && ev V4C12k_S10 $R/$W/runs/v4c12k_from48k_s10/ema_012000.pt
log "chain_eval end"
