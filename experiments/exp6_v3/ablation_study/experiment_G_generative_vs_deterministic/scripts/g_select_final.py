"""Experiment G -- D0 checkpoint selection with the FROZEN V3 rule (seen2scene.v3.trainer.selection_score: min over evaluated steps >= 5000
of ES_in_val / ES_in_val(5k) + ES_out_sc / ES_out_sc(5k); ties -> later step), as V3-T0 (task0_vae/checkpoints/final.json) and
experiment_XB xb_select_final.py. Copies the chosen EMA to work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt (+ final.json).
Usage: python g_select_final.py [--run DIR]"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import torch

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
from seen2scene.v3 import trainer as T  # noqa: E402

W = os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default=os.path.join(W, "runs/main"))
a = ap.parse_args()
dst = os.path.join(W, "checkpoints/final.pt")
if os.path.exists(dst):
    sys.exit(f"exists {dst} (never overwritten)")
ev = [json.loads(x) for x in open(os.path.join(a.run, "eval.jsonl"))]
log = [json.loads(x) for x in open(os.path.join(a.run, "log.jsonl"))]
cfg = json.load(open(os.path.join(T.CFG_DIR, "train_v3.json")))
done = log[-1]["step"] == cfg["steps"] and any(e["step"] == cfg["steps"] for e in ev)
early = None
for f in sorted(os.listdir(a.run)):
    if f == "run_none.json":
        early = json.load(open(os.path.join(a.run, f))).get("early_stopped_at")
if not done and early is None:
    sys.exit(f"run not finished: last logged step {log[-1]['step']}, evaluated {[e['step'] for e in ev]}")
sc = T.selection_score(ev)
best = min(sc.values())
step = max(k for k, v in sc.items() if v == best)
src = os.path.join(a.run, f"ema_{step:06d}.pt")
ck = torch.load(src, map_location="cpu")
assert ck["model_cfg"].get("head") == "det" and ck["step"] == step, (ck["model_cfg"].get("head"), ck["step"])
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(src, dst + ".tmp")
os.replace(dst + ".tmp", dst)
info = {"arm": "D0", "protocol": "G-v1", "rule": "min over steps >= 5000 of ES_in/ES_in(5k) + ES_out/ES_out(5k) (VALSEL, S-c); ties -> later step",
        "scores": {str(k): v for k, v in sorted(sc.items())}, "chosen_step": step, "chosen_score": best, "evaluated_steps": [e["step"] for e in ev],
        "last_logged_step": log[-1]["step"], "early_stopped_at": early, "source": src, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
        "time": time.strftime("%F %T")}
json.dump(info, open(os.path.join(W, "checkpoints/final.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
