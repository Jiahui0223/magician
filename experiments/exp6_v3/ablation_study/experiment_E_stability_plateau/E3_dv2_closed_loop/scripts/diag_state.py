"""E3 diagnostic (Dev scene; offline; GT only to score): one planner state of a D1 replicate episode (experiment E runs/rep/D1),
per candidate: true one-step gain (E-B oracle of the rep-0 episode -- valid while the pose history is identical), D1 score (off),
D-v2 score, first-layer-only score, first-layer cut value and wrongly cut value. Output logs/diag/state_<scene>_s<k>_r<rep>_t<t>.json."""
import json
import os
import sys

import dv2_fixed as FX
import numpy as np
import torch
import yaml


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--states", default="3")
    a = ap.parse_args()
    s, k, rep = a.scene, a.start, a.rep
    sys.argv += ["--scene", s]
    dev = torch.device("cuda")
    FX.DP.install("d1")
    cfg = yaml.safe_load(open(os.path.join(FX.E3, "configs", "dv2_v2.yaml")))["dv2"]
    grid = FX.DV2.install(s, dev, cfg)
    root = os.path.join(FX.R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "runs", "rep", "D1")
    ep = json.load(open(os.path.join(root, "episodes", f"ep_V2M_{s}_s{k}_r{rep}.json")))
    svc = os.path.join(root, "svc", ep["tag"])
    logs = {json.loads(l)["t"]: json.loads(l) for l in open(os.path.join(root, "planner_logs", f"pl_{ep['tag']}.steps.jsonl"))}
    orow = {r["t"]: r for r in json.load(open(os.path.join(FX.ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.json")))["steps"]}
    hcfg = yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(FX.R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    fr, dims, dom, _ = FX.TD.scene_geometry(s)
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    sb = json.load(open(os.path.join(FX.DATA, s, "settings.json")))["scene"]
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
    isect = FX.gt_mesh(s)
    kM = float(fr.k_M)
    FX.HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
    states = [int(x) for x in a.states.split(",")]
    K, h, eps = 4, None, None
    os.makedirs(os.path.join(FX.E3, "logs", "diag"), exist_ok=True)
    for t in range(max(states) + 1):
        rec = FX.MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        fu.integrate_record(rec)
        if t % 10 == 0:
            geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = FX.PR.noise(dev, K, lambda kk, tr=t: FX.RY.sha_seed(FX.LC.seed_key(s, k, rep, tr, kk)))
        if t not in states:
            continue
        res = {}
        for name, mode in (("off", "off"), ("dv2", "full"), ("first", "first")):
            FX.DV2.STATE["mode"] = mode
            geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            out, _, views = FX.PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
            res[name] = {"scores": [float(x) for x in out["distp1"]], "one_step": out.get("d_one_step"), "pick": FX.MS.choose([float(x) for x in out["distp1"]], s, k, t, True)["pick"]}
            if name == "dv2":
                per = []
                for j, v in enumerate(views):
                    q = FX.cut_quality(grid, fr, [v], {"hits": [out["hits"][j]]}, geo, dims, isect, z0, kM, K, dev)
                    per.append(q)
                res["cut"] = per
        FX.DV2.STATE["mode"] = "off"
        lg = logs[t]
        reg = all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(res["off"]["scores"], lg["scores"]))
        dc = [c["dcov_pp"] for c in orow[t]["cands"]] if t in orow and len(orow[t]["cands"]) == len(res["off"]["scores"]) else None
        keys = [str(list(map(int, r))) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
        json.dump({"t": t, "keys": keys, "true_dcov_pp": dc, **res, "off_equals_log": reg}, open(os.path.join(FX.E3, "logs", "diag", f"state_{s}_s{k}_r{rep}_t{t}.json"), "w"), default=float)
        print(f"t={t} off == log: {reg} | picks off {res['off']['pick']} dv2 {res['dv2']['pick']} first {res['first']['pick']}")
        print("cand key | true gain pp | D1 score (one-step) | D-v2 score (one-step) | first-only | L1 novel off -> on | cut | wrong cut")
        for j, key in enumerate(keys):
            q = res["cut"][j]
            print(f"{j} {key} | {dc[j] if dc else float('nan'):.2f} | {res['off']['scores'][j]:.0f} ({res['off']['one_step'][j]:.0f}) | {res['dv2']['scores'][j]:.0f} ({res['dv2']['one_step'][j]:.0f}) | "
                  f"{res['first']['scores'][j]:.0f} | {q['nov_off']:.0f} -> {q['nov_on']:.0f} | {q['cut_w']:.0f} | {q['wrong_w']:.0f}")


if __name__ == "__main__":
    main()
