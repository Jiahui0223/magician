"""EXP Density A/B -- Task 0: checkpoint and E2-pair audit (seen2scene env, CPU, read only).

Checks (results/t0_ckpt.json):
  1. the three checkpoints (B2 = V2-Mix EMA@17.5k, E2-S10 EMA@5k, E2-D100 EMA@5k): resolved path, sha256, keys, step,
     model_cfg, parameter count, strict load into the frozen V2 model
  2. E2 init: both arms' config init sha == B2 sha; ema_000000 of both arms == B2 EMA weights tensor-wise
  3. E2 pair alignment: config diff, run metadata (world size, view counts, data-file key sets, code fingerprint),
     per-logged-step views / views_indoor / lr equality
  4. B0 / B2 / B1 historical closed loops: checkpoint sha used by the planner (group config) and the P1 mode
Usage: python da_t0_ckpt.py
"""
import glob
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts"))
T3W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
E2T = os.path.join(T3W, "e2", "training")
E2C = os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "configs")
CK = {"B2": os.path.join(REPO, "work_dirs", "checkpoints", "exp6_v2", "B2", "v2mix_ema17500.pt"),
      "E2S": os.path.join(REPO, "work_dirs", "checkpoints", "exp6_v2", "E2", "control_e2s10", "e2s10_ema5000.pt"),
      "E2D": os.path.join(REPO, "work_dirs", "checkpoints", "exp6_v2", "E2", "e2d100_ema5000.pt")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sd_equal(a, b):
    if set(a) != set(b):
        return False, "key sets differ"
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    return not bad, f"{len(bad)} tensors differ" if bad else "all tensors bitwise equal"


def main():
    sys.path.insert(0, REPO)
    from seen2scene.futureview import model as M
    out = {"checkpoints": {}, "e2_init": {}, "e2_pair": {}, "historical_groups": {}}
    sds = {}
    for k, p in CK.items():
        rp = os.path.realpath(p)
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        sd = ck["ema"] if "ema" in ck else ck["model"]
        sds[k] = sd
        net = M.V2Model(ck["model_cfg"])
        strict = None
        if net is not None:
            r = net.load_state_dict(sd, strict=True)
            strict = f"missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}"
        out["checkpoints"][k] = {"registry_path": os.path.relpath(p, REPO), "real_path": os.path.relpath(rp, REPO), "sha256": sha(rp),
                                 "keys": sorted(ck.keys()), "step": ck.get("step"), "weights_key": "ema" if "ema" in ck else "model",
                                 "n_params": int(sum(v.numel() for v in sd.values())), "model_cfg_version": ck.get("model_cfg", {}).get("version"),
                                 "strict_load": strict, "rays_fc_shape": list(sd["rays.fc.weight"].shape)}
    out["model_cfg_equal_all"] = len({json.dumps(torch.load(os.path.realpath(p), map_location="cpu", weights_only=False)["model_cfg"], sort_keys=True)
                                      for p in CK.values()}) == 1
    # ---- E2 init
    b2sha = out["checkpoints"]["B2"]["sha256"]
    for arm in ("s10", "d100"):
        cfg = json.load(open(os.path.join(E2C, f"e2_{arm}_v1.json")))
        e0 = torch.load(os.path.join(E2T, arm, "ema_000000.pt"), map_location="cpu", weights_only=False)
        eq, msg = sd_equal(e0["ema"], sds["B2"])
        out["e2_init"][arm] = {"init_checkpoint": cfg["init_checkpoint"], "init_sha_cfg": cfg["init_checkpoint_sha256"],
                               "init_sha_eq_B2": cfg["init_checkpoint_sha256"] == b2sha, "ema0_eq_B2_weights": eq, "detail": msg,
                               "ema0_step": e0.get("step")}
    # ---- E2 pair alignment
    ca = json.load(open(os.path.join(E2C, "e2_s10_v1.json")))
    cb = json.load(open(os.path.join(E2C, "e2_d100_v1.json")))
    out["e2_pair"]["config_keys_differing"] = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    out["e2_pair"]["config_s10"] = ca
    ra = json.load(open(os.path.join(E2T, "s10", "run_none.json")))
    rb = json.load(open(os.path.join(E2T, "d100", "run_none.json")))
    out["e2_pair"]["run_meta"] = {"world_size": [ra["world_size"], rb["world_size"]], "views_equal": ra["views"] == rb["views"], "views": ra["views"],
                                  "data_file_keys_equal": set(ra["data_files_sha256"]) == set(rb["data_files_sha256"]),
                                  "n_data_files": len(ra["data_files_sha256"]),
                                  "n_data_files_same_bytes": sum(ra["data_files_sha256"][k] == rb["data_files_sha256"].get(k) for k in ra["data_files_sha256"]),
                                  "code_sha_equal": ra["code_sha256"] == rb["code_sha256"], "resumed_at": [ra["resumed_at"], rb["resumed_at"]],
                                  "gpu": [ra["gpu"], rb["gpu"]], "host": [ra["host"], rb["host"]],
                                  "final_loss_mean50": [ra["final_loss_mean50"], rb["final_loss_mean50"]]}
    la = [json.loads(x) for x in open(os.path.join(E2T, "s10", "log.jsonl"))]
    lb = [json.loads(x) for x in open(os.path.join(E2T, "d100", "log.jsonl"))]
    out["e2_pair"]["log"] = {"n_logged": [len(la), len(lb)], "steps_equal": [x["step"] for x in la] == [x["step"] for x in lb],
                             "views_equal": all(x["views"] == y["views"] and x["views_indoor"] == y["views_indoor"] for x, y in zip(la, lb)),
                             "lr_equal": all(x["lr"] == y["lr"] for x, y in zip(la, lb)),
                             "rays_indoor_mean": [sum(x["rays_indoor"] for x in la) / len(la), sum(x["rays_indoor"] for x in lb) / len(lb)],
                             "rays_outdoor_mean": [sum(x["rays_outdoor"] for x in la) / len(la), sum(x["rays_outdoor"] for x in lb) / len(lb)]}
    ckl = {arm: torch.load(os.path.join(E2T, arm, "ckpt_last.pt"), map_location="cpu", weights_only=False) for arm in ("s10", "d100")}
    out["e2_pair"]["ckpt_last_step"] = {arm: ckl[arm].get("step") for arm in ckl}
    out["e2_pair"]["ema5000_eq_registry"] = {arm: sha(os.path.join(E2T, arm, "ema_005000.pt")) == out["checkpoints"][k]["sha256"]
                                             for arm, k in (("s10", "E2S"), ("d100", "E2D"))}
    # ---- historical groups
    cfg_in = json.load(open(os.path.join(REPO, "experiments", "exp5_future_view", "task2", "configs", "eval_indoor_v1.json")))
    g = cfg_in["groups"]["V2M"]
    out["historical_groups"]["V2M_group_ckpt"] = {"checkpoint": g["checkpoint"], "sha256": g["sha256"], "eq_B2": g["sha256"] == b2sha}
    for grp in ("B0", "B1", "B2", "E2S", "E2D"):
        errs = sorted(glob.glob(os.path.join(T3W, "closed_loop", "indoor", grp, "V2M", "*", "planner.err")))
        modes, ckpts = set(), set()
        for f in errs:
            for line in open(f, errors="ignore"):
                if line.startswith("T3_P1_MODE="):
                    modes.add(line.strip().split("=", 1)[1])
                if line.startswith("T3_CKPT="):
                    ckpts.add(line.strip().split("=", 1)[1])
        out["historical_groups"][grp] = {"n_indoor_episodes": len(errs), "p1_modes_logged": sorted(modes) or ["(none logged: Task 2 run = sparse)"],
                                         "t3_ckpt": sorted(ckpts) or ["(unset: group V2M checkpoint)"]}
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "results", "t0_ckpt.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "e2_pair"}, indent=1, default=str)[:4000])
    print(json.dumps({k: v for k, v in out["e2_pair"].items() if k != "config_s10"}, indent=1, default=str))


if __name__ == "__main__":
    main()
