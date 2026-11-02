"""Publish the frozen checkpoint of an arm (FROZEN_PROTOCOL v1.0: fixed EMA @ 12000) as work_dirs/exp6_v5/checkpoints/
<arm>_final.pt (+ <arm>_final.json: sha256, rule, S-c monitoring curve). Refuses unless train_<arm>.log has 'EXIT 0'
and the EMA file of the frozen step exists; never reads an evaluation scene. Usage: python select_final.py v5|v4c"""
import hashlib
import json
import os
import shutil
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
arm = sys.argv[1]
W = f"{R}/work_dirs/exp6_v5"
cfg = json.load(open(f"{R}/experiments/exp6_v5/configs/" + ("train_v5_outdoor.json" if arm == "v5" else "train_v4_continue.json")))
step = cfg["steps"]
assert cfg["checkpoint_policy"].startswith(f"fixed_ema_step_{step}"), cfg["checkpoint_policy"]
run = f"{W}/runs/{arm}"
src = f"{run}/ema_{step:06d}.pt"
tl = open(f"{W}/logs/train_{arm}.log").read().splitlines()
if not any(x.startswith("EXIT 0 ") for x in tl):
    sys.exit("train log has no 'EXIT 0' line; nothing published")
log = [json.loads(x) for x in open(f"{run}/log.jsonl")]
if log[-1]["step"] != step or not os.path.exists(src):
    sys.exit(f"run did not reach step {step} or {src} missing; nothing published")
ev = [json.loads(x) for x in open(f"{run}/eval.jsonl")] if os.path.exists(f"{run}/eval.jsonl") else []
dst = f"{W}/checkpoints/{arm}_final.pt"
if os.path.exists(dst):
    sys.exit(f"{dst} exists -- frozen once published")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
info = {"arm": arm, "rule": f"fixed EMA @ {step} (frozen before training; no evaluation scene involved)", "chosen_step": step, "source": src,
        "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(), "time": time.strftime("%F %T"), "evaluated_steps": [e["step"] for e in ev],
        "monitor_sc": {str(e["step"]): {k: v for k, v in e.items() if k != "step"} for e in ev}}
json.dump(info, open(f"{W}/checkpoints/{arm}_final.json", "w"), indent=1, default=float)
print("published", dst, info["sha256"])
