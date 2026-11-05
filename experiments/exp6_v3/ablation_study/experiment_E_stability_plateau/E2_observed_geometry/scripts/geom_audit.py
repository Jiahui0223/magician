"""E2 -- observed-geometry consistency audit of the V3-D1 planner (seen2scene env, 1 GPU; offline diagnostic, GT never reaches a planner).
For one canonical D1 episode, the planner state is replayed every `state_every` steps exactly as experiment E's e_predict.py
(b2 P1: D100 fusion for inputs / projection / decoding, S10 twin band_dil; h_policy.capture for the S10 observed points; R10
encode; closed-loop noise keys) and ONE frozen score_state(keep_hits=True) is evaluated over the legal candidates. For every
candidate ray and sample k the following are compared:
  t_pred   decoded first hit (geometry.decode: a hit is only ever placed at an UNKNOWN sample BEFORE t_known, otherwise = t_known)
  t_known  first zero crossing of the D100 BAND TSDF along the ray (SceneCaster.cast on vol_obs) -- the planner's known surface
  t_true   GT first hit (embree ray cast of the GT mesh, runtime frame; offline only)
  D100 / S10 state at points: D100 BAND within d100_near_vox voxels; S10 band_dil (DistP1 novelty); S10 observed point within
           1.0 runtime units (value_cov novelty = D1's value)
and each predicted hit that D1's value counts as novel gets the class flags A / B / C / T / F / N of configs/audit_config.yaml.
B rays get a leak diagnosis from the D100 cast volume sampled along the ray around t_true: band present on the ray? negative
(behind-surface) values present? first value already <= 0 (ray starts inside / approaches from behind)?
Output: results/audit_<scene>_s<k>.jsonl (one row per state: per-candidate class weights / counts, executed index, true gains)
and results/audit_<scene>_s<k>_cells.npz (cell ids of the executed view's B / C novel hits, for recurrence)."""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E2))))      # experiments/
R = os.path.dirname(X)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
import h_policy as HP  # noqa: E402
import m_p1  # noqa: E402
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
import trimesh  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402

DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
ORA = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "oracle")
VS = 0.011


def gt_mesh(scene):
    import glob
    obj = sorted(glob.glob(os.path.join(DATA, scene, "*.obj")))[0]
    m = trimesh.load(obj, force="mesh", process=False, skip_materials=True)
    m.vertices = np.asarray(m.vertices, np.float64) * 10.0                       # scene_scale_factor (runtime units)
    return trimesh.ray.ray_pyembree.RayMeshIntersector(m)


def gt_first_hit(isect, o_rt, d_rt, max_rt):
    """distance (runtime) to the first GT surface along each ray, inf if none within max_rt."""
    n = len(d_rt)
    out = np.full(n, np.inf)
    idx_tri, idx_ray, loc = isect.intersects_id(np.repeat(o_rt[None], n, 0), d_rt, multiple_hits=False, return_locations=True)
    if len(idx_ray):
        dist = np.linalg.norm(loc - o_rt[None], axis=1)
        out[idx_ray] = dist
    out[out > max_rt] = np.inf
    return out


def near(mask_dil, p_model, dims):
    """bool per point: voxel of p is inside the (dilated) mask; points outside the grid -> False."""
    ijk = torch.floor(p_model / VS).long()
    ok = (ijk >= 0).all(-1) & (ijk[:, 0] < dims[0]) & (ijk[:, 1] < dims[1]) & (ijk[:, 2] < dims[2])
    res = torch.zeros(len(p_model), dtype=torch.bool, device=p_model.device)
    if ok.any():
        q = ijk[ok]
        res[ok] = mask_dil[q[:, 0], q[:, 1], q[:, 2]]
    return res


def leak_diag(vol, o, d, t_lo, t_hi, dims, n=24):
    """D100 cast-volume values along each ray segment [t_lo, t_hi] (model units) -> (band_on_ray, neg_on_ray, first_nonpos)."""
    ts = t_lo[:, None] + (t_hi - t_lo)[:, None] * torch.linspace(0, 1, n, device=o.device)[None]
    p = o[None, None] + d[:, None] * ts[..., None]
    dimt = torch.as_tensor(dims, dtype=p.dtype, device=p.device)
    uvw = p / VS - 0.5
    g = 2 * uvw / (dimt - 1) - 1
    grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)[None, :, :, None]
    val = F.grid_sample(vol[None, None], grid.to(vol.dtype), mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, :, 0]
    band = (val.abs() < MC.TRUNC * 0.999).any(1)
    neg = (val <= 0).any(1)
    first = val[:, 0] <= 0
    return band, neg, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--config", default=os.path.join(E2, "configs", "audit_config.yaml"))
    ap.add_argument("--out", default=os.path.join(E2, "results"))
    ap.add_argument("--tmax", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tol = cfg["tolerances"]
    t0 = time.time()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    s, k = a.scene, a.start
    rec_t1 = json.load(open(os.path.join(EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
    ep = json.load(open(os.path.join(R, rec_t1["episode_file"])))
    root = os.path.dirname(os.path.dirname(os.path.join(R, rec_t1["episode_file"])))
    svc = os.path.join(root, "svc", ep["tag"])
    logs = {}
    for line in open(os.path.join(root, "planner_logs", f"pl_{ep['tag']}.steps.jsonl")):
        r = json.loads(line)
        logs[r["t"]] = r
    orp = os.path.join(ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.json")
    orow = {r["t"]: r for r in json.load(open(orp))["steps"]} if os.path.exists(orp) else {}
    hcfg = yaml.safe_load(open(os.path.join(X, "exp6_v3", "H", "FROZEN_CONFIG.yaml")))
    ckpt = os.path.join(R, hcfg["model"]["checkpoint"])
    core = PV3.V3Core(ckpt, dev, expect_sha=hcfg["model"]["sha256"])
    vcfg = hcfg["policies"]["h1"]
    fr, dims, dom, _ = TD.scene_geometry(s)
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)                      # = P1FusionB2 after PF.patch("b2")
    sb = json.load(open(os.path.join(DATA, s, "settings.json")))["scene"]
    z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
    isect = gt_mesh(s)
    kM = float(fr.k_M)
    HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})
    K, h, eps = 4, None, None
    T = len(ep["steps"]) - 1 if a.tmax is None else a.tmax + 1
    os.makedirs(a.out, exist_ok=True)
    fo = open(os.path.join(a.out, f"audit_{s}_s{k}.jsonl.tmp"), "w")
    cellsB, cellsC = [], []
    dense_cells = set()                    # every valid depth pixel of every executed frame (incl. outside the P1 domain), 0.5-unit cells
    from scipy.spatial import cKDTree
    for t in range(T):
        rec = MC.load_service_record(os.path.join(svc, f"svc_{s}_s{k}_t{t}.npz"))
        HP.capture(rec)
        fu.integrate_record(rec)
        dp, _ = PF.full_points(rec)
        dense_cells.update(map(tuple, np.unique(np.floor(np.asarray(dp, np.float64) / 0.5).astype(np.int64), axis=0)))
        if t % 10 == 0:
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            eps = PR.noise(dev, K, lambda kk, tr=t: RY.sha_seed(LC.seed_key(s, k, 0, tr, kk)))
        if t % cfg["state_every"] or len(rec["legal_R"]) == 0:
            continue
        ts_ = time.time()
        geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        c = geo.caster
        d100_band = c.obs_band                                                      # D100 BAND (b2 replaces only band_dil)
        r_ = int(tol["d100_near_vox"])
        d100_near = F.max_pool3d(d100_band[None, None].half(), 2 * r_ + 1, 1, r_)[0, 0] > 0
        s10_dil = c.band_dil                                                        # S10 sparse twin, dilated 3x3x3
        tree = HP.obs_tree()
        dtree = cKDTree((np.asarray(list(dense_cells), np.float64) + 0.5) * 0.5 if dense_cells else np.full((1, 3), 1e9))
        lg = (logs.get(t) or {}).get("all_scores") or {}
        cands = []
        for j, v in enumerate(views):
            o, d, dz, pr = v["o"], v["d"], v["dz"], v["pr"]
            t_dec, hit_u = out["hits"][j]["t"], out["hits"][j]["hit_u"]           # [K, R]
            tk = pr["t_known"]
            o_rt = np.asarray(fr.model_to_runtime(o.double().cpu().numpy()[None]))[0]
            d_rt = MC.to_y_up(d.double().cpu().numpy()) if hasattr(MC, "to_y_up") else None
            d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
            max_rt = float(v["tcap"].max().item()) * kM
            t_true = torch.as_tensor(gt_first_hit(isect, o_rt, d_rt, max_rt) / kM, device=dev, dtype=t_dec.dtype)   # model units
            t_true = torch.where(t_true <= v["tcap"], t_true, torch.full_like(t_true, float("inf")))
            # sanity of the known surfaces: decoded-known rays vs GT
            fin_k = torch.isfinite(tk) & torch.isfinite(t_true)
            known_ok = float(((tk[fin_k] - t_true[fin_k]).abs() * kM <= tol["t_rt"]).float().mean()) if fin_k.any() else float("nan")
            # true occluder on a D100 surface?
            p_true = o[None] + d * torch.where(torch.isfinite(t_true), t_true, torch.zeros_like(t_true))[:, None]
            occ_known = near(d100_near, p_true, dims) & torch.isfinite(t_true)
            ijt = torch.floor(p_true / VS).long()
            okt = (ijt >= 0).all(-1) & (ijt[:, 0] < dims[0]) & (ijt[:, 1] < dims[1]) & (ijt[:, 2] < dims[2])
            qt = ijt.clamp(min=0)
            qt[:, 0].clamp_(max=dims[0] - 1); qt[:, 1].clamp_(max=dims[1] - 1); qt[:, 2].clamp_(max=dims[2] - 1)
            occ_indom = okt & geo.domain[qt[:, 0], qt[:, 1], qt[:, 2]] & torch.isfinite(t_true)
            ptr_rt = np.asarray(fr.model_to_runtime(p_true.double().cpu().numpy()))
            dd, _ = dtree.query(ptr_rt, k=1, distance_upper_bound=1.0)
            occ_obs = torch.as_tensor(np.isfinite(dd), device=dev) & torch.isfinite(t_true)   # true occluder seen in some executed depth image
            agg = {f"{q}_{c_}": 0.0 for q in ("w", "n") for c_ in ("A", "B", "C", "T", "F", "N", "AC", "BC", "novel", "all",
                                                                  "A_out_obs", "A_out_unobs", "A_in_obs", "A_in_unobs")}
            leak = {"band_on_ray": 0, "neg_on_ray": 0, "first_nonpos": 0, "B_rays": 0}
            for kk in range(K):
                tp = t_dec[kk]
                fin = torch.isfinite(tp)
                p = o[None] + d * torch.where(fin, tp, torch.zeros_like(tp))[:, None]
                ijk = torch.floor(p / VS).long()
                inb = fin & (ijk >= 0).all(-1) & (ijk[:, 0] < dims[0]) & (ijk[:, 1] < dims[1]) & (ijk[:, 2] < dims[2])
                q = ijk.clamp(min=0)
                q[:, 0].clamp_(max=dims[0] - 1); q[:, 1].clamp_(max=dims[1] - 1); q[:, 2].clamp_(max=dims[2] - 1)
                indom = inb & geo.domain[q[:, 0], q[:, 1], q[:, 2]]
                pr_rt = np.asarray(fr.model_to_runtime(p.double().cpu().numpy()))
                dist, _ = tree.query(pr_rt, k=1, distance_upper_bound=float(tol["s10_cov_eps_rt"]))
                nov = indom & torch.as_tensor(~np.isfinite(dist), device=dev)                       # D1 value_cov novelty
                z = (tp * kM * dz).double()
                w = torch.clamp((z / z0) ** 2, max=1.0)
                w = torch.where(nov, w, torch.zeros_like(w))
                on_d100 = near(d100_near, p, dims) & fin
                known_hit = fin & ~hit_u[kk]                                                      # decoded = t_known
                dt = (tp - t_true) * kM
                beyond = fin & torch.isfinite(t_true) & (dt > tol["t_rt"])
                front = fin & torch.isfinite(t_true) & (dt < -tol["t_rt"])
                cons = fin & torch.isfinite(t_true) & (dt.abs() <= tol["t_rt"])
                fA = beyond & ~occ_known
                fB = beyond & occ_known
                fC = on_d100 | known_hit
                fT = cons & ~fC
                fN = fin & ~torch.isfinite(t_true)
                for nm, f_ in (("A", fA), ("B", fB), ("C", fC), ("T", fT), ("F", front), ("N", fN), ("AC", fA & fC), ("BC", fB & fC), ("novel", nov),
                               ("A_out_obs", fA & ~occ_indom & occ_obs), ("A_out_unobs", fA & ~occ_indom & ~occ_obs),
                               ("A_in_obs", fA & occ_indom & occ_obs), ("A_in_unobs", fA & occ_indom & ~occ_obs)):
                    agg[f"w_{nm}"] += float(w[f_ & nov].sum()) / K
                    agg[f"n_{nm}"] += float((f_ & nov).sum()) / K
                agg["w_all"] += float(w.sum()) / K
                agg["n_all"] += float(fin.sum()) / K
                bi = torch.nonzero(fB & nov)[:, 0]
                if bi.numel():
                    lo = (t_true[bi] - tol["t_rt"] / kM).clamp(min=0)
                    hi = t_true[bi] + tol["t_rt"] / kM
                    band, neg, first = leak_diag(geo.vol_obs, o, d[bi], lo, hi, dims)
                    leak["band_on_ray"] += int(band.sum()); leak["neg_on_ray"] += int(neg.sum()); leak["first_nonpos"] += int(first.sum()); leak["B_rays"] += int(bi.numel())
                if j == (logs.get(t) or {}).get("choice", {}).get("pick", -1):
                    cid = np.floor(pr_rt / 1.0).astype(np.int64)
                    key = (cid[:, 0] << 40) ^ (cid[:, 1] << 20) ^ cid[:, 2]
                    cellsB.append(np.unique(key[(fB & nov).cpu().numpy()]))
                    cellsC.append(np.unique(key[(fC & nov).cpu().numpy()]))
            hm = lg.get("h_mean")
            hv = float(np.mean([float(HP.value_cov(t_dec[kk], v, geo, z0, vcfg)[1].sum()) for kk in range(K)]))     # D1 per-candidate value (replay check)
            rel = abs(hv - hm[j]) / max(abs(hm[j]), 1e-9) if hm else None
            oc = (orow.get(t) or {}).get("cands")
            cands.append({"j": j, "agg": agg, "leak": leak, "known_surface_agrees_with_gt": known_ok,
                          "n_known_rays": int(torch.isfinite(tk).sum()), "n_unknown_rays": int((pr["cls"] == 0).sum()),
                          "log_h_mean": (hm[j] if hm else None), "replay_h_mean": hv, "replay_rel_err": rel, "true_dcov_pp": (oc[j]["dcov_pp"] if oc and j < len(oc) else None)})
        pick = (logs.get(t) or {}).get("choice", {}).get("pick")
        fo.write(json.dumps({"t": t, "executed": pick, "n": len(views), "cands": cands, "seconds": time.time() - ts_}, default=float) + "\n")
        fo.flush()
        del geo, out, views
        torch.cuda.empty_cache()
        print(f"{s} s{k} t={t} {time.time() - ts_:.1f}s", flush=True)
    fo.close()
    os.replace(os.path.join(a.out, f"audit_{s}_s{k}.jsonl.tmp"), os.path.join(a.out, f"audit_{s}_s{k}.jsonl"))
    np.savez_compressed(os.path.join(a.out, f"audit_{s}_s{k}_cells.npz"), **{f"B_{i}": x for i, x in enumerate(cellsB)}, **{f"C_{i}": x for i, x in enumerate(cellsC)})
    print("done", s, k, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
