#!/bin/bash
# CVPR Table 1 -- regression replay (user Task 9): re-run one historical episode (barts start 0) per main model with the
# CURRENT code and the Table-1 launch chain, into work_dirs/cvpr_results/table1/g3/replay/<model>/<group>, then compare the
# executed keys and the official coverage curve with the historical episode (must be bitwise equal).
# Usage: replay_check.sh GPU MODEL [MODEL ...]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
GPU_ARG=$1; shift
source <(sed -n '/^EXP5=/,/^release()/p' experiments/cvpr_results/table1/scripts/t1_queue.sh | sed '$d')
g=$GPU_ARG
T1W=$R/work_dirs/cvpr_results/table1/g3; mkdir -p $T1W/replay $T1W/logs
for m in "$@"; do
  G=$(group_of $m); root=$T1W/replay/$m/$G
  [ -f $root/episodes/ep_${G}_barts_s0_r0.json ] || CUDA_VISIBLE_DEVICES=$g run_sim $m barts 0 $root > $T1W/logs/replay_${m}.log 2>&1 < /dev/null
  /bsuscratch/jiahuizhang/envs/seen2scene/bin/python - "$m" "$root/episodes/ep_${G}_barts_s0_r0.json" <<'PY' >> $T1W/logs/replay_check.log 2>&1
import json, sys
m, new = sys.argv[1], sys.argv[2]
aud = json.load(open("experiments/cvpr_results/table1/audit/historical_outdoor_episodes.json"))["episodes"]
h = next(e for e in aud if e["model"] == m and e["scene"] == "barts" and e["start"] == 0)
a, b = json.load(open(new)), json.load(open(h["episode"]))
print(m, "executed_keys_equal", a["executed_keys"] == b["executed_keys"], "official_coverage_equal", a["coverage"] == b["coverage"], "hist", h["episode"])
PY
done
echo "REPLAY_DONE $* $(date '+%F %T')" >> $T1W/logs/replay_check.log
