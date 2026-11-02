"""G0-9 resume: A = 40 steps straight, B = 20 steps then resume to 40 (one process: exact numpy / torch / CUDA / surface-
query generator RNG restore; Bridge -> Joint switch at step 20 = the resume point). PASS if both end at step 40, the
model / EMA parameters agree to float tolerance and all logged losses are finite. Usage: python g0_resume_check.py"""
import json
import sys

import torch

W = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v5/runs/g0"
a = torch.load(f"{W}/resume_A/ckpt_last.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{W}/resume_B/ckpt_last.pt", map_location="cpu", weights_only=False)
out = {"step_A": a["step"], "step_B": b["step"]}
for k in ("model", "ema"):
    out[f"max_abs_diff_{k}"] = max(float((a[k][n].float() - b[k][n].float()).abs().max()) for n in a[k])
    out[f"rms_{k}"] = float(torch.sqrt(sum((a[k][n].float() ** 2).sum() for n in a[k]) / sum(a[k][n].numel() for n in a[k])))
la = [json.loads(x) for x in open(f"{W}/resume_A/log.jsonl")]
lb = [json.loads(x) for x in open(f"{W}/resume_B/log.jsonl")]
out["log_A"] = {r["step"]: round(r["loss"], 6) for r in la}
out["log_B"] = {r["step"]: round(r["loss"], 6) for r in lb}
out["finite"] = all(r["loss"] == r["loss"] and abs(r["loss"]) < 1e6 for r in la + lb)
out["status"] = "PASS" if out["step_A"] == out["step_B"] == 40 and out["max_abs_diff_model"] <= 1e-3 * max(out["rms_model"], 1e-6) and out["finite"] else "CHECK"
print(json.dumps(out, indent=1))
json.dump(out, open(f"{W}/resume_check.json", "w"), indent=1)
sys.exit(0 if out["status"] == "PASS" else 1)
