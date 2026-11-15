"""Experiment G -- compare a train_g.py run with the archived V3-T0 run (work_dirs/exp6_v3/task0_vae/runs/main), protocol G-v1 section 6.
--mode flow  (a) flag-off: eval step 0 (every value), log step 1 (every value) bitwise equal; run manifest hashes equal (config, model
             config, common init, data manifest); later logged steps reported (not expected bitwise: nondeterministic backward).
--mode det   (d) D0 smoke / run: common init + data manifest + config equal; log step 1 rec / kl / rays / beta / lr equal bitwise (same init,
             batch, VAE noise); eval step 0 recon_* equal bitwise; loss finite and decreasing; model_cfg = T0 config + head det only.
Usage: python check_train_vs_t0.py --run DIR --mode flow|det --out JSON"""
import argparse
import glob
import json
import math
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T0 = os.path.join(R, "work_dirs/exp6_v3/task0_vae/runs/main")


def jl(p):
    return [json.loads(x) for x in open(p)]


def run_manifest(d):
    fs = sorted(glob.glob(os.path.join(d, "run_none_*.json")))
    return json.load(open(fs[0]))


def diff(a, b, path=""):
    """list of paths where a != b (exact)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} missing")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif a != b:
        out.append(f"{path}: {a} != {b}")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--mode", required=True, choices=("flow", "det"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
mt, mr = run_manifest(T0), run_manifest(a.run)
et, er = {e["step"]: e for e in jl(os.path.join(T0, "eval.jsonl"))}, {e["step"]: e for e in jl(os.path.join(a.run, "eval.jsonl"))}
lt, lr = {e["step"]: e for e in jl(os.path.join(T0, "log.jsonl"))}, {e["step"]: e for e in jl(os.path.join(a.run, "log.jsonl"))}
res = {"run": a.run, "mode": a.mode, "t0_run": T0}
for k in ("config_sha256", "common_init_sha256", "data_manifest_sha256", "world_size", "params", "data_counts", "views", "torch", "gpu"):
    res[f"manifest_{k}_equal"] = mt[k] == mr[k]
mcfg_t0 = dict(mt["model_cfg"])
mcfg_r = dict(mr["model_cfg"])
head = mcfg_r.pop("head", None)
res["model_cfg_equal_except_head"] = mcfg_t0 == mcfg_r
res["model_cfg_head"] = head
tc_t0, tc_r = dict(mt["train_cfg"]), dict(mr["train_cfg"])
res["train_cfg_diff"] = diff(tc_t0, tc_r)
res["overrides"] = mr["overrides"]
res["host"], res["t0_host"] = mr["host"], mt["host"]
if a.mode == "flow":
    res["eval0_diff"] = diff(et[0], er[0]) if 0 in er else ["missing eval 0"]
    s1 = {k: v for k, v in lt[1].items() if k not in ("s_per_step", "mem_gb")}
    r1 = {k: v for k, v in lr[1].items() if k not in ("s_per_step", "mem_gb")}
    res["log1_diff"] = diff(s1, r1)
    res["later_steps"] = {s: {"t0": {k: lt[s][k] for k in ("loss", "fm", "rec", "kl", "grad_norm")}, "run": {k: lr[s][k] for k in ("loss", "fm", "rec", "kl", "grad_norm")}}
                          for s in sorted(lr) if s > 1 and s in lt}
    res["pass"] = (not res["eval0_diff"] and not res["log1_diff"] and res["model_cfg_equal_except_head"] and head is None
                   and all(res[f"manifest_{k}_equal"] for k in ("config_sha256", "common_init_sha256", "data_manifest_sha256", "world_size", "params")))
else:
    keys = ("rec", "kl", "beta", "lr", "views", "rays_indoor", "rays_outdoor", "rec_diag", "posterior", "kl_per_ch")
    res["log1_same_fields"] = {k: lt[1][k] == lr[1][k] for k in keys}
    res["log1_fm"] = {"t0_flow_velocity_mse": lt[1]["fm"], "d0_det_mse": lr[1]["fm"]}
    res["eval0_recon_equal"] = {k: et[0][k] == er[0][k] for k in ("recon_in_val", "recon_out_sc")} if 0 in er else None
    res["eval_fields_equal"] = {s: sorted(er[s]) == sorted(et[s]) and all(sorted(er[s][k]) == sorted(et[s][k]) for k in er[s] if isinstance(er[s][k], dict))
                                for s in er if s in et}
    steps = sorted(lr)
    losses = [lr[s]["loss"] for s in steps]
    res["loss_curve"] = {s: {k: lr[s][k] for k in ("loss", "fm", "rec", "kl", "grad_norm", "s_per_step")} for s in steps}
    res["loss_finite"] = all(math.isfinite(x) for x in losses)
    res["loss_decreasing"] = len(losses) >= 3 and losses[-1] < losses[1] < losses[0] if len(losses) >= 3 else None
    res["pass"] = (all(res["log1_same_fields"].values()) and res["loss_finite"] and bool(res["loss_decreasing"]) and res["model_cfg_equal_except_head"]
                   and head == "det" and (res["eval0_recon_equal"] is None or all(res["eval0_recon_equal"].values()))
                   and all(res[f"manifest_{k}_equal"] for k in ("config_sha256", "common_init_sha256", "data_manifest_sha256", "world_size", "params")))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("loss_curve", "later_steps")}, default=str)[:3000])
print("PASS" if res["pass"] else "FAIL")
