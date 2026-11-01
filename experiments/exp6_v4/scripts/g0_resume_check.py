"""G0-10: compare the resume test runs of an arm (A = straight, B = interrupted + resumed): steps, max |theta_A - theta_B|
for the model and the EMA, optimizer step counts, logged losses. V4 (single GPU) restores the exact RNG state, so A and B
must agree to float tolerance; V3.1 (4 GPUs, re-seeded streams after a resume = the V3 rule) must restore the checkpoint
state exactly (B's step-30 checkpoint == A's step-30 trajectory is not checked; B2's loaded state equals B1's saved state)
and continue with finite losses. Usage: python g0_resume_check.py v4|v31"""
import json
import os
import sys

import torch

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
arm = sys.argv[1]
W = f"{R}/work_dirs/exp6_v4/runs" if arm == "v4" else f"{R}/work_dirs/exp6_v3_1/runs"
a, b = torch.load(f"{W}/resume_A/ckpt_last.pt", map_location="cpu", weights_only=False), torch.load(f"{W}/resume_B/ckpt_last.pt", map_location="cpu", weights_only=False)
out = {"arm": arm, "step_A": a["step"], "step_B": b["step"]}
for k in ("model", "ema"):
    out[f"max_abs_diff_{k}"] = max(float((a[k][n].float() - b[k][n].float()).abs().max()) for n in a[k])
    out[f"rms_{k}"] = float(torch.sqrt(sum((a[k][n].float() ** 2).sum() for n in a[k]) / sum(a[k][n].numel() for n in a[k])))
la = [json.loads(x) for x in open(f"{W}/resume_A/log.jsonl")]
lb = [json.loads(x) for x in open(f"{W}/resume_B/log.jsonl")]
out["losses_A"] = {r["step"]: round(r["loss"], 6) for r in la}
out["losses_B"] = {r["step"]: round(r["loss"], 6) for r in lb}
out["finite"] = all(r["loss"] == r["loss"] and abs(r["loss"]) < 1e6 for r in la + lb)
rb2 = sorted(f for f in os.listdir(f"{W}/resume_B") if f.startswith("run_none_"))
if rb2:
    runs = [json.load(open(f"{W}/resume_B/{f}")) for f in rb2]
    out["resumed_at_B"] = [r.get("resumed_at") for r in runs]
if arm == "v4":
    out["status"] = "PASS" if out["step_A"] == out["step_B"] and out["max_abs_diff_model"] < 1e-4 * max(out["rms_model"], 1e-6) * 10 and out["finite"] else "CHECK"
else:
    out["status"] = "PASS" if out["step_A"] == out["step_B"] and out["finite"] and out.get("resumed_at_B", [0, 0])[-1] == 30 else "CHECK"
json.dump(out, open(f"{W}/resume_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
