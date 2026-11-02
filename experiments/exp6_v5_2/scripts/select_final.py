"""Publish the frozen V5.2 checkpoint (FROZEN_PROTOCOL: fixed EMA@6000) as work_dirs/exp6_v5_2/checkpoints/v52_final.pt
(+ v52_final.json: sha256, rule, S-c monitor, Stage-J collapse check). Refuses without 'EXIT 0' or the EMA file; never
reads an evaluation scene. Collapse check (protocol §1): S-c second+ surface recall at 6000 < 0.5 x step 0 -> exit 2."""
import hashlib
import json
import os
import shutil
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = f"{R}/work_dirs/exp6_v5_2"
cfg = json.load(open(f"{R}/experiments/exp6_v5_2/configs/train_v52.json"))
step = cfg["steps"]
run = f"{W}/runs/v52"; src = f"{run}/ema_{step:06d}.pt"
if not any(x.startswith("EXIT 0 ") for x in open(f"{W}/logs/train_v52.log").read().splitlines()):
    sys.exit("no 'EXIT 0' in train_v52.log; nothing published")
ev = [json.loads(x) for x in open(f"{run}/eval.jsonl")]
e0, eN = ev[0], [e for e in ev if e["step"] == step][0]
collapse = eN["recall_second_plus"] < 0.5 * e0["recall_second_plus"]
dst = f"{W}/checkpoints/v52_final.pt"
if os.path.exists(dst):
    sys.exit(f"{dst} exists -- frozen once published")
shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
info = {"rule": f"fixed EMA @ {step} (frozen before training)", "source": src, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
        "time": time.strftime("%F %T"), "sc_second_plus_recall": {"step0": e0["recall_second_plus"], f"step{step}": eN["recall_second_plus"]},
        "collapse": bool(collapse), "monitor_sc": ev}
json.dump(info, open(f"{W}/checkpoints/v52_final.json", "w"), indent=1, default=float)
print("published", dst, info["sha256"], "collapse", collapse)
sys.exit(2 if collapse else 0)
