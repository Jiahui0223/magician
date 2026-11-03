"""EXP6 V2 (B2 / E2) -- arm configuration loader and checkpoint verification. No model code lives here: the V2 model,
predictor, planner core and DistP1 stay in their shared modules (seen2scene/futureview/{model,predictor,rays,geometry,
scoring}.py, experiments/exp5_future_view/task1/scripts/v2_plan_core.py, task2 / task3 simulators and planners).

load(arm)            -> the arm config (experiments/exp6_v2/<arm>/configs/<arm>.json) with absolute paths
verify(arm)          -> checks: config fields, registry link == original file, sha256, step, embedded model_cfg ==
                        model_v1.json, strict load into V2Model, parameter count (control arm of E2 included)
core(arm, dev, ...)  -> v2_plan_core.Core of the arm's checkpoint read through the registry path (sha enforced)
CLI: python v2_arm.py B2|E2 [--json OUT]   (seen2scene env; CPU is enough)"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
sys.path.insert(0, R)


def load(arm):
    arm = arm.upper()
    cfg = json.load(open(os.path.join(E, arm, "configs", f"{arm.lower()}.json")))
    assert cfg["arm"] == arm, cfg["arm"]
    return cfg


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(R, p)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_ckpt(entry, model_cfg_file):
    import torch
    from seen2scene.futureview import model as M
    new, orig = abspath(entry.get("path") or entry["checkpoint"]), abspath(entry["original_path"])
    out = {"path": entry.get("path") or entry["checkpoint"], "original_path": entry["original_path"], "is_symlink": os.path.islink(new),
           "same_file_as_original": os.path.samefile(new, orig)}
    out["sha256"] = sha256(new)
    out["sha256_ok"] = out["sha256"] == entry["sha256"]
    ck = torch.load(new, map_location="cpu", weights_only=False)
    out["step"] = ck.get("step")
    out["step_ok"] = ck.get("step") == entry["step"]
    out["model_cfg_equals_config_file"] = ck["model_cfg"] == json.load(open(abspath(model_cfg_file)))
    m = M.V2Model(ck["model_cfg"])
    res = m.load_state_dict(ck["ema"], strict=True)
    out["strict_load"] = not res.missing_keys and not res.unexpected_keys
    out["params"] = int(sum(p.numel() for p in m.parameters()))
    out["ok"] = all(out[k] for k in ("is_symlink", "same_file_as_original", "sha256_ok", "step_ok", "model_cfg_equals_config_file", "strict_load"))
    return out


def verify(arm):
    cfg = load(arm)
    mf = cfg["architecture"]["model_config"]
    res = {"arm": cfg["arm"], "config": os.path.relpath(os.path.join(E, cfg["arm"], "configs", f"{cfg['arm'].lower()}.json"), R)}
    missing = []
    for k in ("architecture", "checkpoint", "planner", "evaluation"):
        if k not in cfg:
            missing.append(k)
    for p in [mf, cfg["planner"]["scoring_config"], *cfg["planner"]["group_configs"], *[x.split(" ")[0] for x in cfg["planner"]["implementation"]]]:
        if not os.path.exists(abspath(p)):
            missing.append(p)
    res["missing"] = missing
    res["planner"] = {k: cfg["planner"][k] for k in ("p1_mode", "K", "flow_steps", "n_win_indoor", "refresh", "score")}
    res["checkpoint"] = check_ckpt(cfg["checkpoint"], mf)
    res["params_ok"] = res["checkpoint"]["params"] == cfg["architecture"]["params"]
    if "control_arm_E2S10" in cfg:
        c = dict(cfg["control_arm_E2S10"], path=cfg["control_arm_E2S10"]["checkpoint"])
        res["control_e2s10"] = check_ckpt(c, mf)
    res["ok"] = not missing and res["checkpoint"]["ok"] and res["params_ok"] and res.get("control_e2s10", {"ok": True})["ok"]
    return res


def core(arm, dev, control=False):
    """v2_plan_core.Core of the arm checkpoint (registry path; sha enforced by the frozen core)."""
    sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task1", "scripts"))
    import v2_plan_core as PC
    cfg = load(arm)
    c = cfg["control_arm_E2S10"] if control else cfg["checkpoint"]
    return PC.Core(abspath(c.get("path") or c["checkpoint"]), dev, expect_sha=c["sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["B2", "E2", "b2", "e2"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = verify(a.arm)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    sys.exit(0 if r["ok"] else 1)
