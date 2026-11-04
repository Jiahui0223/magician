#!/bin/bash
# EXP6 V3 / Y -- wait for the milestone EMA ema_<STEP>.pt, register it in the Table 1 registry (sha256, status READY), write the
# Dev-5 job file (H0 = the V3 launch chain) and start WORKERS t1_queue workers per GPU on gpu114.
# Usage: y_register_and_queue.sh STEP WORKERS [extra_condition: wait_stop]
set -u
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
STEP=$1; NW=$2; COND=${3:-}
E=$(printf "work_dirs/exp6_v3/Y/runs/main/ema_%06d.pt" $STEP); MID="V3-Y$((STEP/1000))K-H0"
EV=work_dirs/exp6_v3/Y/EVENTS.jsonl
until [ -f $E ]; do sleep 60; done
sleep 30                                                    # the trainer writes the EMA with torch.save (complete when the eval line follows)
if [ "$COND" = wait_stop ]; then until grep -q '"stopped_at_target"' work_dirs/exp6_v3/Y/STATUS.json 2>/dev/null; do sleep 60; done; fi
SHA=$(sha256sum $E | cut -d' ' -f1)
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python - "$MID" "$SHA" <<'PY'
import sys, re
mid, sha = sys.argv[1], sys.argv[2]
p = "experiments/cvpr_results/table1/MODEL_REGISTRY.yaml"; s = open(p).read()
i = s.index(f"  - id: {mid}\n"); j = s.index("  - id:", i + 5) if "  - id:" in s[i + 5:] else len(s)
blk = s[i:j].replace("status: PENDING_CHECKPOINT", "status: READY").replace("sha256: null", f"sha256: {sha}")
s = s[:i] + blk + s[j:]
open(p, "w").write(s)
import yaml; d = yaml.safe_load(open(p)); m = next(x for x in d["models"] if x["id"] == mid); print(mid, m["status"], m["sha256"][:12])
PY
J=experiments/cvpr_results/table1/configs/jobs_y$((STEP/1000))k_dev.txt
( for k in 0 1 2 3 4; do for s in barts colosseum eiffel pantheon sestino_museum; do echo "$MID:$s:$k"; done; done ) > $J
echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"track\": \"Y\", \"event\": \"milestone_registered\", \"model\": \"$MID\", \"ema\": \"$E\", \"sha256\": \"$SHA\", \"jobfile\": \"$J\", \"workers_per_gpu\": $NW}" >> $EV
timeout 60 ssh -o BatchMode=yes gpu114 "cd $R && for g in 0 1 2 3; do for w in \$(seq $NW); do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh \$g $J > /dev/null 2>&1 < /dev/null & sleep 1; done; done; echo started" 2>&1 | grep -v "WARN\|vulner\|upgrad"
echo "{\"ts\": \"$(date -u +%FT%TZ)\", \"track\": \"Y\", \"event\": \"eval_workers_started\", \"model\": \"$MID\", \"host\": \"gpu114\", \"workers\": $((4*NW))}" >> $EV
