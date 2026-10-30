"""Apply the frozen checkpoint-selection rule (protocol.md v1.0 §3) to one arm's finished run and publish the chosen EMA
as work_dirs/exp6_v3/<task>/checkpoints/final.pt (+ final.json with the scores and sha256).
score(step) = ES_in(step) / ES_in(5k) + ES_out(step) / ES_out(5k) over the evaluated steps >= 5k, minimum wins, ties ->
the later step; no evaluation scene is involved (VALSEL and S-c only).  Usage: python select_final.py t0|t1 [--dry]"""
import hashlib, json, os, shutil, sys, time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, R)
from seen2scene.v3 import trainer as T  # noqa: E402

arm = sys.argv[1]; dry = "--dry" in sys.argv
task = "task0_vae" if arm == "t0" else "task1_xattn"
W = f"{R}/work_dirs/exp6_v3/{task}"; run = f"{W}/runs/main"
ev = [json.loads(x) for x in open(f"{run}/eval.jsonl")]
sc = T.selection_score(ev)
if not sc:
    sys.exit("no evaluated step >= 5000")
best = min(sc.values()); step = max(k for k, v in sc.items() if v == best)       # ties -> later step
src = f"{run}/ema_{step:06d}.pt"
log = [json.loads(x) for x in open(f"{run}/log.jsonl")]
info = {"arm": arm, "rule": "min over steps >= 5000 of ES_in/ES_in(5k) + ES_out/ES_out(5k); ties -> later step",
        "scores": {str(k): v for k, v in sorted(sc.items())}, "chosen_step": step, "chosen_score": best,
        "last_logged_step": log[-1]["step"], "evaluated_steps": [e["step"] for e in ev], "source": src,
        "early_stopped": log[-1]["step"] < json.load(open(f"{T.CFG_DIR}/train_v3.json"))["steps"], "time": time.strftime("%F %T")}
print(json.dumps({k: v for k, v in info.items() if k != "scores"}, indent=1)); print("scores", {k: round(v, 4) for k, v in sc.items()})
if dry:
    sys.exit(0)
tl = open(f"{W}/logs/train.log").read().splitlines()
if not any(x.startswith("EXIT 0 ") for x in tl):
    sys.exit("train.log has no 'EXIT 0' line -- the run has not finished cleanly; nothing published")
os.makedirs(f"{W}/checkpoints", exist_ok=True)
dst = f"{W}/checkpoints/final.pt"
if os.path.exists(dst):
    sys.exit(f"{dst} exists -- the selection is frozen once published; remove it by hand if this is a deliberate redo")
shutil.copy2(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
info["sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
json.dump(info, open(f"{W}/checkpoints/final.json", "w"), indent=1)
print("published", dst, info["sha256"][:16])
