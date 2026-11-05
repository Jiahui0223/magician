"""Geometry / ROI audit J (seen2scene env, 1 GPU, read-only): does the FULL two-step D-v2 scoring (first layer + lookahead + union,
D-v2 filter in both layers) write anything into the real maps? Replay of a Table 1 D1 episode with d_policy d1 and dv2 (mode full)
installed exactly as in the planner child; at every audited state the checksums of the D100 P1, the S10 twin, the observed-point
list of the H / D policy and the D-v2 grid are taken before and after score_state. Output results/j_lookahead_check.json."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
E3 = os.path.join(REPO, "experiments", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "E3_dv2_closed_loop")
sys.path.insert(0, os.path.join(E3, "scripts"))
import dv2_fixed as FX  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


def chk(fu, grid, HP):
    h = hashlib.sha256()
    for t in (fu.sum, fu.cnt, fu.empty, fu.sparse.sum, fu.sparse.cnt, fu.sparse.empty, grid.occ):
        h.update(t.detach().cpu().numpy().tobytes())
    h.update(str((len(HP.STATE["obs"]), sum(len(x) for x in HP.STATE["obs"]), sorted(HP.STATE["visited"]))).encode())
    return h.hexdigest()


def main():
    out = {}
    dev = torch.device("cuda")
    for s, k, states in (("colosseum", 1, (0, 5, 10, 20, 40)), ("pisa", 1, (0, 10, 30))):
        sys.argv = [sys.argv[0], "--scene", s]
        if not out:
            FX.DP.install("d1")
        cfg = yaml.safe_load(open(os.path.join(E3, "configs", "dv2_v2.yaml")))["dv2"]
        # one install per scene would stack wrappers; a fresh grid is fed manually instead and the installed value_cov wrapper reads it
        if not out:
            grid = FX.DV2.install(s, dev, cfg)
            grids = {"g": grid}
        else:
            ng = FX.DV2.ObservedGrid(s, dev)
            grid.__dict__.update(ng.__dict__)                       # same object the installed wrappers hold, re-initialised for the new scene
        FX.DV2.STATE["mode"] = "full"
        rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
        ep = json.load(open(os.path.join(REPO, rec1["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(REPO, rec1["episode_file"])))
        svc = os.path.join(root, "svc", ep["tag"])
        z = np.load(os.path.join(REPO, "work_dirs", "exp6_v3", "H", "pose_tables", f"{s}.npz"))
        FX.HP.STATE["pose"] = {"lut": {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, "R": z["R"], "T": z["T"], "C": z["C"], "grid": json.loads(str(z["grid"]))}
        hcfg = yaml.safe_load(open(os.environ["H_CONFIG"]))
        core = FX.PV3.V3Core(os.path.join(REPO, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
        fr, dims, dom, _ = FX.TD.scene_geometry(s)
        fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
        sb = json.load(open(os.path.join(FX.DATA, s, "settings.json")))["scene"]
        z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
        FX.HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
        K, h, eps, res = 4, None, None, []
        for t in range(max(states) + 1):
            rec = FX.MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
            fu.integrate_record(rec)                              # wrappers: D-policy capture + b2 fusion + D-v2 grid update
            if t % 10 == 0:
                geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
                h = core.pred.encode(geo.coarse())
                eps = FX.PR.noise(dev, K, lambda kk, tr=t: FX.RY.sha_seed(FX.LC.seed_key(s, k, 0, tr, kk)))
            if t not in states:
                continue
            geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            for key in FX.DV2.STATS:
                FX.DV2.STATS[key] = [0, 0, 0] if key != "sec" else [0.0, 0.0, 0.0]
            c0 = chk(fu, grid, FX.HP)
            out_, _, views = FX.PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0)
            c1 = chk(fu, grid, FX.HP)
            res.append({"t": t, "unchanged": c0 == c1, "lookahead_views_scored": int(out_["d"]["predicted_views"]), "layer2_rays_filtered": int(FX.DV2.STATS["cut"][2]),
                        "layer1_rays_filtered": int(FX.DV2.STATS["cut"][1])})
            print(s, k, res[-1], flush=True)
            del geo, out_, views
            torch.cuda.empty_cache()
        out[f"{s}_s{k}"] = res
        del fu, core
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(AUD, "results", "j_lookahead_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
