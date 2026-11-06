"""Experiment XA -- checkpoint selection of one finished run: the frozen V3 rule (seen2scene.v3.trainer.selection_score, protocol.md v1.0 §3 of EXP6 V3):
score(step) = ES_in(step) / ES_in(5k) + ES_out(step) / ES_out(5k) over the evaluated steps >= 5k on VALSEL and S-c only (no evaluation scene), minimum
wins, ties -> the later step. The chosen EMA is copied to work_dirs/.../checkpoints/<id>/final.pt (+ final.json with scores and sha256); never overwritten.
Usage: python select_final.py ID"""
import hashlib
import json
import os
import shutil
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
from seen2scene.v3 import trainer as T  # noqa: E402

vid = sys.argv[1]
W = f"{R}/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene"
run = f"{W}/runs/{vid}"
dst = f"{W}/checkpoints/{vid}/final.pt"
if os.path.exists(dst):
    print("exists", dst)
    sys.exit(0)
ev = [json.loads(x) for x in open(f"{run}/eval.jsonl")]
log = [json.loads(x) for x in open(f"{run}/log.jsonl")]
steps = json.load(open(f"{R}/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/configs/train_{vid}.json"))["steps"]
if log[-1]["step"] != steps or not any(e["step"] == steps for e in ev):
    sys.exit(f"run not finished: last logged step {log[-1]['step']}, evaluated {[e['step'] for e in ev]}")
sc = T.selection_score(ev)
best = min(sc.values())
step = max(k for k, v in sc.items() if v == best)
src = f"{run}/ema_{step:06d}.pt"
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy2(src, dst + ".tmp")
os.replace(dst + ".tmp", dst)
info = {"id": vid, "rule": "min over steps >= 5000 of ES_in/ES_in(5k) + ES_out/ES_out(5k) (VALSEL, S-c); ties -> later step", "scores": {str(k): v for k, v in sorted(sc.items())},
        "chosen_step": step, "chosen_score": best, "evaluated_steps": [e["step"] for e in ev], "source": src, "sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
        "time": time.strftime("%F %T")}
json.dump(info, open(f"{W}/checkpoints/{vid}/final.json", "w"), indent=1)
print("published", dst, "step", step, info["sha256"][:16])
