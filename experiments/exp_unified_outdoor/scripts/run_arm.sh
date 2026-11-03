#!/bin/bash
# run_arm.sh <arm> [extra historical-trainer args, e.g. --smoke short]
#   Formal run of one unified outdoor-only arm on 4 GPUs (torchrun --standalone, 4 ranks) given by CUDA_VISIBLE_DEVICES.
#   Output: work_dirs/exp_unified_outdoor/runs/<arm>/ ; log: work_dirs/exp_unified_outdoor/logs/train_<arm>.log
#   Resume: re-run the same command; the trainer resumes from runs/<arm>/ckpt_last.pt and the per-rank RNG is restored
#   from runs/<arm>/rng_states/ (uo_common.ExactResume). A finished arm (run_none.json with "finished") exits 0.
#   Exit codes: 0 ok / already finished; 2 usage or environment error; 3 parent checkpoint missing (V4C / V5 arms);
#   4 trainer finished but the final EMA is missing; otherwise the torchrun exit code.
# arms: v2_s10 v2_d100 v4_s10 v4_d100 v3t1_s10 v3t1_d100
#       v4c12k_from48k_s10 v4c12k_from48k_d100 v5warm12k_from48k_s10 v5warm12k_from48k_d100
set -u
ARM=${1:-}
[ -n "$ARM" ] || { echo "usage: $0 <arm> [extra trainer args]" >&2; exit 2; }
shift
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp_unified_outdoor
CFG=$R/experiments/exp_unified_outdoor/configs/$ARM.json
OUT=$W/runs/$ARM
LOG=$W/logs/train_$ARM.log
[ -f "$CFG" ] || { echo "unknown arm $ARM (no $CFG)" >&2; exit 2; }
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "set CUDA_VISIBLE_DEVICES to 4 GPU ids (e.g. 0,1,2,3)" >&2; exit 2; fi
NG=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[ "$NG" -eq 4 ] || { echo "the unified recipe needs exactly 4 GPUs (16 views / rank); got $NG ($CUDA_VISIBLE_DEVICES)" >&2; exit 2; }
case $ARM in
  v4c12k_from48k_*|v5warm12k_from48k_*) FINAL=12000; PARENT=$W/runs/v4_${ARM##*_}/ema_048000.pt ;;
  *) FINAL=60000; PARENT="" ;;
esac
case $ARM in v3t1_s10)
  NC=$(find $W/cache/tsdf4_s10/outdoor -name '*.npz' 2>/dev/null | wc -l)
  [ "$NC" -ge 180 ] || { echo "S10 TSDF cache incomplete ($NC / 180): run scripts/build_tsdf_cache_s10.sh first" >&2; exit 2; } ;;
esac
SMOKE=0
for x in "$@"; do case $x in --smoke*) SMOKE=1 ;; esac; done
if [ $SMOKE -eq 0 ] && [ -f "$OUT/run_none.json" ] && grep -q '"finished"' "$OUT/run_none.json"; then
  echo "$ARM already finished ($OUT/run_none.json)"; exit 0
fi
if [ -n "$PARENT" ] && [ ! -f "$PARENT" ]; then
  echo "parent checkpoint missing: $PARENT (run v4_${ARM##*_} to >= 48000 steps first)" >&2; exit 3
fi
cd $R || exit 2
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
mkdir -p "$OUT" "$W/logs"
echo "start $ARM $(hostname -s) GPUs $CUDA_VISIBLE_DEVICES job ${SLURM_JOB_ID:-none} git $(git rev-parse --short HEAD) dirty $(git status --porcelain | wc -l) extra [$*] $(date '+%F %T')" >> "$LOG"
$S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp_unified_outdoor/scripts/train_unified.py --arm "$ARM" --out "$OUT" "$@" >> "$LOG" 2>&1
rc=$?
echo "EXIT $rc $ARM $(hostname -s) $(date '+%F %T')" >> "$LOG"
[ $rc -eq 0 ] || exit $rc
if [ $SMOKE -eq 0 ]; then
  F=$(printf "%s/ema_%06d.pt" "$OUT" "$FINAL")
  [ -f "$F" ] || { echo "trainer exited 0 but $F is missing" >&2; exit 4; }
fi
exit 0
