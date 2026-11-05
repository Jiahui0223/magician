"""E3 Task 1 diagnostic (Dev states only; offline; GT only to score): what are D-v2 support-9 'wrong cuts'? For first-layer views of
the D1 replay at the given states, every cut D1-novel predicted hit whose true first surface is not in front of it is described by:
ray elevation (runtime y-up direction y-component), t_obs, t_pred, t_true, whether the ray leaves the occupied layer between t_obs and
t_pred (a non-occupied sample after entry), and the occupied-layer depth along the ray. Output logs/diag/wrong_cut_<scene>_s<k>.json (raw per-ray records; moved to work_dirs/.../E3_dv2_closed_loop/diag after the run, summary in logs/diag/wrong_cut_summary.json)."""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv_saved = list(sys.argv)
import dv2_fixed as FX  # noqa: E402  (imports and installs nothing at import time except module setup)
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--states", default="0,5,10,15")
    a = ap.parse_args()
    s, k = a.scene, a.start
    sys.argv += ["--scene", s]
    dev = torch.device("cuda")
    FX.DP.install("d1")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(FX.E3, "configs", "dv2_v1.yaml")))["dv2"]   # diagnostic of the v1 design (support 9, single ray)
    cfg["footprint"] = 0.0
    grid = FX.DV2.install(s, dev, cfg)
    FX.DV2.CFG["footprint"] = 0.0
    FX.DV2.STATE["mode"] = "off"
    rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
    ep = json.load(open(os.path.join(FX.R, rec1["episode_file"])))
    root = os.path.dirname(os.path.dirname(os.path.join(FX.R, rec1["episode_file"])))
    svc = os.path.join(root, "svc", ep["tag"])
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
    K, h, eps, res = 4, None, None, []
    for t in range(max(states) + 1):
        rec = FX.MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        fu.integrate_record(rec)
        if t % 10 == 0:
            geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = FX.PR.noise(dev, K, lambda kk, tr=t: FX.RY.sha_seed(FX.LC.seed_key(s, k, 0, tr, kk)))
        if t not in states or len(rec["legal_R"]) == 0:
            continue
        geo = FX.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        out, _, views = FX.PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        occ_raw, occ_rel = grid.occ, grid.reliable()
        for j, v in enumerate(views):
            o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
            d_rt = FX.MC.to_y_up(v["d"].double().cpu().numpy())
            d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
            t_true = FX.gt_first_hit(isect, o_rt, d_rt, float(v["tcap"].max().item()) * kM)
            tob = grid.t_obs(o_rt, d_rt, (v["tcap"].double() * kM).cpu().numpy()).cpu().numpy()
            for kk in range(K):
                tp = (out["hits"][j]["t"][kk].double() * kM).cpu().numpy()
                cut = np.isfinite(tp) & (tp > tob + cfg["margin"])
                wrong = cut & np.isfinite(t_true) & (t_true >= tp - 1.0)
                right = cut & ~wrong
                for lab, m in (("wrong", wrong), ("right", right)):
                    idx = np.nonzero(m)[0]
                    if len(idx) > 400:
                        idx = np.random.default_rng(0).choice(idx, 400, replace=False)
                    for i in idx:
                        ts = np.arange(tob[i], tp[i], 0.25)
                        p = o_rt[None] + d_rt[i][None] * ts[:, None]
                        ijk = np.floor((p - grid.lo) / cfg["cell"]).astype(int)
                        ok = (ijk >= 0).all(1) & (ijk < grid.n).all(1)
                        ijk = np.clip(ijk, 0, grid.n - 1)
                        o_ = torch.as_tensor(ijk, device=dev)
                        inocc = occ_raw[o_[:, 0], o_[:, 1], o_[:, 2]].cpu().numpy() & ok
                        ex = np.nonzero(~inocc)[0]
                        res.append({"t": t, "cls": lab, "elev_deg": float(np.degrees(np.arcsin(d_rt[i][1]))), "t_obs": float(tob[i]), "t_pred": float(tp[i]),
                                    "t_true": float(t_true[i]) if np.isfinite(t_true[i]) else None,
                                    "t_exit": float(ts[ex[0]]) if len(ex) else None, "layer_len": float(ts[ex[0]] - tob[i]) if len(ex) else float(tp[i] - tob[i])})
    os.makedirs(os.path.join(FX.E3, "logs", "diag"), exist_ok=True)
    json.dump(res, open(os.path.join(FX.E3, "logs", "diag", f"wrong_cut_{s}_s{k}.json"), "w"))
    for lab in ("wrong", "right"):
        r = [x for x in res if x["cls"] == lab]
        if not r:
            continue
        el = np.array([x["elev_deg"] for x in r]); ll = np.array([x["layer_len"] for x in r])
        exit_before = np.array([x["t_exit"] is not None and x["t_exit"] < x["t_pred"] - cfg["margin"] for x in r])
        print(f"{lab}: n={len(r)} elev median {np.median(el):.1f} deg (|elev|<10: {np.mean(np.abs(el) < 10):.2f}, <0: {np.mean(el < 0):.2f}) "
              f"occupied-layer length median {np.median(ll):.2f} (>1.0: {np.mean(ll > 1.0):.2f}) leaves layer before t_pred-margin: {exit_before.mean():.2f}")


if __name__ == "__main__":
    main()
