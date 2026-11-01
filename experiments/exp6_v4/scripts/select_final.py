"""Publish the frozen checkpoint of an arm (FROZEN_PROTOCOL v1.0: fixed EMA @ steps) as <work>/checkpoints/final.pt
(+ final.json with sha256, the S-c monitoring curve and the run facts). Refuses unless train.log has 'EXIT 0' and the
EMA file of the frozen step exists; never reads an evaluation scene. Usage: python select_final.py v4|v31 [--dry]"""
import hashlib, json, os, shutil, sys, time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
arm = sys.argv[1]; dry = "--dry" in sys.argv
W = f"{R}/work_dirs/exp6_v4" if arm == "v4" else f"{R}/work_dirs/exp6_v3_1"
cfg_p = f"{R}/experiments/exp6_v4/configs/outdoor_train.json" if arm == "v4" else f"{R}/experiments/exp6_v3.1/configs/outdoor_train.json"
cfg = json.load(open(cfg_p)); step = cfg["steps"]
assert cfg["checkpoint_policy"] == f"fixed_ema_step_{step}", cfg["checkpoint_policy"]
run = f"{W}/runs/main"; src = f"{run}/ema_{step:06d}.pt"
ev = [json.loads(x) for x in open(f"{run}/eval.jsonl")] if os.path.exists(f"{run}/eval.jsonl") else []
log = [json.loads(x) for x in open(f"{run}/log.jsonl")]
info = {"arm": arm, "rule": f"fixed EMA @ {step} (frozen before training; no evaluation scene involved)", "chosen_step": step,
        "last_logged_step": log[-1]["step"], "evaluated_steps": [e["step"] for e in ev], "source": src, "time": time.strftime("%F %T"),
        "monitor_sc": {str(e["step"]): {k: v for k, v in e.items() if k.startswith("fm_") or k.startswith("samp_") or k.startswith("recon_")} for e in ev}}
print(json.dumps({k: v for k, v in info.items() if k != "monitor_sc"}, indent=1))
if dry:
    sys.exit(0)
tl = open(f"{W}/logs/train.log").read().splitlines()
if not any(x.startswith("EXIT 0 ") for x in tl):
    sys.exit("train.log has no 'EXIT 0' line -- the run has not finished cleanly; nothing published")
if log[-1]["step"] != step or not os.path.exists(src):
    sys.exit(f"run did not reach step {step} or {src} missing; nothing published")
os.makedirs(f"{W}/checkpoints", exist_ok=True)
dst = f"{W}/checkpoints/final.pt"
if os.path.exists(dst):
    sys.exit(f"{dst} exists -- frozen once published; remove it by hand if this is a deliberate redo")
shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
info["sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
json.dump(info, open(f"{W}/checkpoints/final.json", "w"), indent=1)
print("published", dst, info["sha256"][:16])
