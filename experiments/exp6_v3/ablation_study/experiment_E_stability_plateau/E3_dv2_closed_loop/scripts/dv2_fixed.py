"""E3 Task 1 -- fixed-state check of D-v2 on the full D1 decision (seen2scene env, 1 GPU; offline; GT used only to score).
Replays the D1 Table 1 planner state (svc records, same replay as E2 geom_audit / ksv_fixed) with D-v1 d1 (depth 2, lambda 0)
and dv2.install(). At every `state_every` step the full D1 score_state (first layer + two-step lookahead + union) is evaluated on
the same state in four modes:
  off        switch off -> must equal the planner log (scores and pick)                              [regression]
  ksv1       footprint 0 (single ray), all layers = E2 KSV-v1 (must equal E2's fixed-state ON scores)   [reference]
  dv2        footprint 0.5, all layers  = D-v2                                                      [candidate]
  dv2_first  footprint 0.5, first layer only                                                        [diagnostic ablation]
Per mode: scores, pick (m_score.choose), cuts per layer, wall time; with the E-B oracle (true one-step gain of every candidate):
Top-1, selected true gain (pp), Spearman. For ksv1 / dv2 the first-layer cut quality against GT (embree first hit): cut value,
'wrong' cut value (true first surface not in front of the predicted hit: t_true >= t_pred - 1), novel value beyond the true surface
before / after. Output results/fixed/dv2_fixed_<scene>_s<k>.jsonl."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E3))))
R = os.path.dirname(X)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("D_CONFIG", os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "configs", "d_config.yaml"))
os.environ.setdefault("H_CONFIG", os.path.join(X, "exp6_v3", "H", "FROZEN_CONFIG.yaml"))
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task1", "scripts"))
import v2_common as LC  # noqa: E402
import v2_frozen as VF  # noqa: E402,F401
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task3_full_depth_p1", "scripts"))
import p1_full as PF  # noqa: E402
PF.patch("b2")
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
import planner_v3 as PV3  # noqa: E402
PC = PV3.install()
sys.path.insert(0, os.path.join(X, "exp6_v3", "H", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
sys.path.insert(0, HERE)
import h_policy as HP  # noqa: E402
import d_policy as DP  # noqa: E402
import dv2 as DV2  # noqa: E402
import m_p1  # noqa: E402
import m_common as MC  # noqa: E402
import m_score as MS  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import trimesh  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402

DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
ORA = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "oracle")
VS = 0.011
MODES = (("off", "off", None), ("ksv1", "full", 0.0), ("dv2", "full", None), ("dv2_first", "first", None))     # None -> config footprint


def gt_mesh(scene):
    import glob
    m = trimesh.load(sorted(glob.glob(os.path.join(DATA, scene, "*.obj")))[0], force="mesh", process=False, skip_materials=True)
    m.vertices = np.asarray(m.vertices, np.float64) * 10.0
    return trimesh.ray.ray_pyembree.RayMeshIntersector(m)


def gt_first_hit(isect, o_rt, d_rt, max_rt):
    n = len(d_rt)
    out = np.full(n, np.inf)
    _, idx_ray, loc = isect.intersects_id(np.repeat(o_rt[None], n, 0), d_rt, multiple_hits=False, return_locations=True)
    if len(idx_ray):
        out[idx_ray] = np.linalg.norm(loc - o_rt[None], axis=1)
    out[out > max_rt] = np.inf
    return out


def cut_quality(grid, fr, views, out, geo, dims, isect, z0, kM, K, dev):
    tree = HP.obs_tree()
    acc = dict.fromkeys(("cut_w", "wrong_w", "nov_off", "nov_on", "beyond_off", "beyond_on"), 0.0)
    for j, v in enumerate(views):
        o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
        d_rt = MC.to_y_up(v["d"].double().cpu().numpy())
        d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
        t_true = torch.as_tensor(gt_first_hit(isect, o_rt, d_rt, float(v["tcap"].max().item()) * kM), device=dev)
        tdec = out["hits"][j]["t"]
        vis = DV2.visible_mask(grid, fr, v, tdec, kM)
        for kk in range(K):
            tp = tdec[kk].double() * kM
            fin = torch.isfinite(tp)
            pm = v["o"][None] + v["d"] * torch.where(torch.isfinite(tdec[kk]), tdec[kk], torch.zeros_like(tdec[kk]))[:, None]
            ijk = torch.floor(pm / VS).long()
            okk = (ijk >= 0).all(-1) & (ijk[:, 0] < dims[0]) & (ijk[:, 1] < dims[1]) & (ijk[:, 2] < dims[2])
            q = ijk.clamp(min=0)
            q[:, 0].clamp_(max=dims[0] - 1); q[:, 1].clamp_(max=dims[1] - 1); q[:, 2].clamp_(max=dims[2] - 1)
            indom = okk & geo.domain[q[:, 0], q[:, 1], q[:, 2]]
            dd, _ = tree.query(np.asarray(fr.model_to_runtime(pm.double().cpu().numpy())), k=1, distance_upper_bound=1.0)
            fin = fin & indom & torch.as_tensor(~np.isfinite(dd), device=dev)          # D1-novel predicted hits only
            w = torch.clamp((tp * v["dz"].double() / z0) ** 2, max=1.0)
            cut = fin & ~vis[kk]
            wrong = cut & torch.isfinite(t_true) & (t_true >= tp - 1.0)
            bey = fin & torch.isfinite(t_true) & (tp > t_true + 1.0)
            acc["cut_w"] += float(w[cut].sum()) / K
            acc["wrong_w"] += float(w[wrong].sum()) / K
            acc["nov_off"] += float(w[fin].sum()) / K
            acc["nov_on"] += float(w[fin & vis[kk]].sum()) / K
            acc["beyond_off"] += float(w[bey].sum()) / K
            acc["beyond_on"] += float(w[bey & vis[kk]].sum()) / K
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--state-every", type=int, default=5)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(E3, "results", "fixed"))
    a = ap.parse_args()
    s, k = a.scene, a.start
    sys.argv += ["--scene", s]
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    DP.install("d1")
    cfg = yaml.safe_load(open(os.path.join(E3, "configs", "dv2_v2.yaml")))["dv2"]
    grid = DV2.install(s, dev, cfg)
    rec1 = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
    ep = json.load(open(os.path.join(R, rec1["episode_file"])))
    root = os.path.dirname(os.path.dirname(os.path.join(R, rec1["episode_file"])))
    svc = os.path.join(root, "svc", ep["tag"])
    logs = {json.loads(l)["t"]: json.loads(l) for l in open(os.path.join(root, "planner_logs", f"pl_{ep['tag']}.steps.jsonl"))}
    orp = os.path.join(ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.json")
    orow = {r["t"]: r for r in json.load(open(orp))["steps"]} if os.path.exists(orp) else {}
    hcfg = yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = PV3.V3Core(os.path.join(R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    fr, dims, dom, _ = TD.scene_geometry(s)
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    sb = json.load(open(os.path.join(DATA, s, "settings.json")))["scene"]
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
    isect = gt_mesh(s)
    kM = float(fr.k_M)
    HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
    K, h, eps = 4, None, None
    T = len(ep["steps"]) - 1 if a.tmax is None else a.tmax + 1
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, f"dv2_fixed_{s}_s{k}.jsonl.tmp")
    fo = open(tmp, "w")
    for t in range(T):
        rec = MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        fu.integrate_record(rec)                       # D-policy capture + frozen b2 fusion + D-v2 grid update (wrappers)
        if t % 10 == 0:
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, K, lambda kk, tr=t: RY.sha_seed(LC.seed_key(s, k, 0, tr, kk)))
        if t % a.state_every or len(rec["legal_R"]) == 0:
            continue
        lg = logs.get(t) or {}
        row = {"t": t, "n": len(rec["legal_R"])}
        for name, mode, fp in MODES:
            DV2.STATE["mode"] = mode
            DV2.CFG["footprint"] = float(cfg["footprint"]) if fp is None else fp
            for key in DV2.STATS:
                DV2.STATS[key] = [0, 0, 0]
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            torch.cuda.synchronize()
            tt = time.time()
            out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
            torch.cuda.synchronize()
            dt = time.time() - tt
            sc = [float(x) for x in out["distp1"]]
            ch = MS.choose(sc, s, k, t, True)
            r = {"scores": sc, "pick": ch["pick"], "seconds": dt, "h_mean": out.get("h_mean"),
                 "calls": DV2.STATS["calls"][1:], "rays": DV2.STATS["rays"][1:], "cut": DV2.STATS["cut"][1:]}
            if name in ("ksv1", "dv2"):
                r["cut_quality"] = cut_quality(grid, fr, views, out, geo, dims, isect, z0, kM, K, dev)
            row[name] = r
            del geo, out, views
            torch.cuda.empty_cache()
        DV2.STATE["mode"] = "off"
        logsc = lg.get("scores")
        row["regression_off_equals_log"] = bool(logsc is not None and len(logsc) == len(row["off"]["scores"]) and
                                                 all(abs(x - y) <= 1e-9 * max(1.0, abs(y)) for x, y in zip(row["off"]["scores"], logsc)) and
                                                 row["off"]["pick"] == (lg.get("choice") or {}).get("pick"))
        oc = (orow.get(t) or {}).get("cands")
        if oc and len(oc) == row["n"]:
            dc = np.asarray([c["dcov_pp"] for c in oc])
            best = dc.max()
            for name, _, _ in MODES:
                r = row[name]
                r["top1"] = bool(dc[r["pick"]] >= best - 1e-9) if r["pick"] is not None else None
                r["sel_true_gain_pp"] = float(dc[r["pick"]]) if r["pick"] is not None else None
                v_ = np.asarray(r["scores"])
                r["rho"] = float(spearmanr(v_, dc)[0]) if np.ptp(v_) > 0 and np.ptp(dc) > 0 else None
            row["best_true_gain_pp"] = float(best)
        fo.write(json.dumps(row, default=float) + "\n")
        fo.flush()
        print(f"{s} s{k} t={t} reg={row['regression_off_equals_log']} pick off {row['off']['pick']} ksv1 {row['ksv1']['pick']} dv2 {row['dv2']['pick']} "
              f"first {row['dv2_first']['pick']} | cut L1/L2 dv2 {row['dv2']['cut']} | s {row['off']['seconds']:.1f}/{row['dv2']['seconds']:.1f}", flush=True)
    fo.close()
    os.replace(tmp, tmp[:-4])


if __name__ == "__main__":
    main()
