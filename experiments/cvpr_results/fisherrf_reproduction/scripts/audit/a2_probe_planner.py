"""FisherRF audit A2 -- planner side (fisherrf env, 1 L40S; NO GT): map health + scorer values at fixed recorded states.

Replays the online map of one recorded FisherRF-full closed loop from its service records + RGB side files exactly as
fisherrf_full_planner.py builds it (official GaussianSLAM, same cfg overrides, D12 docstring init, D13 skip, same seeds; the
planner occupancy update copied verbatim). Not bitwise (risk R1: atomicAdd), representative.
At every probe state t (map after frame t = what the planner had when deciding at t):
  map health: Gaussian count, 'phantom' Gaussians (log_scale = -inf: the get_pointcloud max-pool quirk puts them at the camera
    centre), extent vs scene box; render at every executed pose j <= t with valid depth (PSNR on valid pixels, depth L1 in
    runtime units, silhouette coverage of valid pixels, silhouette > 0.5 on pixels WITHOUT valid depth = floaters); render at
    pose t+1 BEFORE it is integrated (novel-view prediction of the observed geometry).
  occupancy: planner labels (unknown / free / occupied) and how many 'unknown' voxels were in fact observed once (the official
    argmax rule needs >= 2 frames to leave 'unknown').
  scorer: candidates = legal one-step neighbours of record t (D3a decision set) + 48 seeded uniform poses of the whole settings
    grid (the F1c / f1c4 candidate distribution). Per candidate: official slam.pose_eval EIG (lambda 0.1) and its per-Gaussian
    decomposition (phantom / never-seen-by-keyframes (H_train row ~ 0) / rest); EIG without phantoms; f1c4 'unknown in FOV'
    count (planner labels, as candidates_magician) and 'never observed in FOV' count; rendered-silhouette novelty.
Output: <out>/<scene>_s<k>.json (the GT side a3_gt_gain.py adds the true coverage gain of the same candidate keys).
Usage: SLURM_JOB_QOS=kd-high python a2_probe_planner.py --scene S --start K --states 5 15 30 50 75 --out DIR
       [--root closed_loop_f1c4 dir]
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
sys.path.insert(0, SCR)
sys.dont_write_bytecode = True
import fisherrf_planner as FP  # noqa: E402
import fisherrf_full_planner as FF  # noqa: E402  (PoseGrid only; its main() is not run)
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(SCR, "..", "..", "..", ".."))
ROOT = os.path.join(REPO, "work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_f1c4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--states", type=int, nargs="+", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-uniform", type=int, default=48)
    a = ap.parse_args()
    assert os.environ.get("SLURM_JOB_QOS") == "kd-high"
    sys.path.insert(0, FP.AM)
    sys.path.insert(0, FP.STUBS)
    import torch
    import torch.nn.functional as Fnn
    import random
    import importlib.util
    spec = importlib.util.spec_from_file_location("m_common", FP.M_COMMON)
    MC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MC)
    from configs.base_config import get_cfg_defaults
    import models.SLAM.gaussian as G
    from models.SLAM.utils.slam_helpers import transformed_params2rendervar, transformed_params2depthplussilhouette

    tag = f"{a.scene}_s{a.start}"
    epdir = os.path.join(a.root, tag)
    svc_dir = os.path.join(epdir, "svc", f"FisherRF_{tag}_r0")
    ep = json.load(open(os.path.join(epdir, "episodes", f"ep_FisherRF_{tag}_r0.json")))
    keys = [json.loads(k) for k in ep["executed_keys"]]
    settings = json.load(open(os.path.join(FF.DATA, a.scene, "settings.json")))
    grid = FF.PoseGrid(settings["camera"], np)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    s_unit = 7.0
    seed = FP.seed_of(f"fisherrf|{a.scene}|{a.start}")                  # identical to the closed loop
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(FP.AM_YAML)
    wd = os.path.join(a.out, "planner_work")
    over = {"SLAM.Dataset.Calibration.fx": FP.FX, "SLAM.Dataset.Calibration.fy": FP.FX, "SLAM.Dataset.Calibration.cx": FP.CX,
            "SLAM.Dataset.Calibration.cy": FP.CY, "SLAM.Dataset.Calibration.width": FP.W_IMG,
            "SLAM.Dataset.Calibration.height": FP.H_IMG, "map_every": 1, "keyframe_every": 1,
            "mapping.num_iters": 60, "workdir": os.path.abspath(wd), "run_name": tag,
            "use_wandb": False, "report_iter_progress": False}
    for k, v in over.items():
        node = cfg
        *path, leaf = k.split(".")
        for p in path:
            node = node[p]
        node[leaf] = v
    slam = G.GaussianSLAM(cfg)
    slam.visualize_frame = lambda *x, **y: None                           # skip the PNG only (no state change)

    def to_w2c(R, T):
        Fm = np.diag(FP.F_FLIP)
        w = np.eye(4)
        w[:3, :3] = Fm @ np.asarray(R, np.float64).reshape(3, 3).T
        w[:3, 3] = Fm @ np.asarray(T, np.float64).reshape(3) / s_unit
        return w

    # ---------------------------------------------------------------- planner occupancy (verbatim copy of fisherrf_full_planner)
    sc = settings["scene"]
    box_lo, box_hi = 10.0 * np.asarray(sc["x_min"], np.float64), 10.0 * np.asarray(sc["x_max"], np.float64)
    lo = np.minimum(box_lo, grid.x_min) - 1e-3
    hi = np.maximum(box_hi, grid.x_max) + 1e-3
    VOX = float(cfg.explore.cell_size) * s_unit
    DIMS = tuple(int(math.ceil((h - l) / VOX)) for l, h in zip(lo, hi))
    LO_T = torch.tensor(lo, dtype=torch.float32, device=dev)
    DIMS_T = torch.tensor(DIMS, device=dev)
    acc = torch.zeros((3,) + DIMS, device=dev)
    acc[0] = 1.0
    nobs = torch.zeros(DIMS, device=dev)                                 # audit only: frames that touched the voxel
    vv, uu = torch.meshgrid(torch.arange(FP.H_IMG, device=dev).float(), torch.arange(FP.W_IMG, device=dev).float(), indexing="ij")
    RAY = torch.stack([(uu - FP.CX) / FP.FX, (vv - FP.CY) / FP.FX, torch.ones_like(uu)], -1)
    FAR = float(cfg.policy.pcd_far_distance)

    def vox_index(P):
        q = torch.floor((P - LO_T) / VOX).long()
        inb = ((q >= 0) & (q < DIMS_T)).all(1)
        return q[inb], inb

    def flat(q):
        return (q[:, 0] * DIMS[1] + q[:, 1]) * DIMS[2] + q[:, 2]

    @torch.no_grad()
    def occ_update(depth_fr, c2w):
        d = depth_fr
        m = d > 0
        if not bool(m.any()):
            return
        rays = RAY[m]
        dv = d[m]
        Rm, tv = c2w[:3, :3], c2w[:3, 3]
        hit = dv < FAR
        Pw = ((rays[hit] * dv[hit, None]) @ Rm.T + tv) * s_unit
        qh, _ = vox_index(Pw)
        hit_flat = flat(qh)
        n_tot = DIMS[0] * DIMS[1] * DIMS[2]
        hits = torch.bincount(hit_flat, minlength=n_tot).float() if len(hit_flat) else torch.zeros(n_tot, device=dev)
        line = torch.zeros(n_tot, dtype=torch.bool, device=dev)
        dmax = float(torch.clamp(dv, max=FAR).max()) * 0.95 * s_unit
        nk = int(min(max(math.ceil(dmax / (0.5 * VOX)) + 1, 11), 600))
        fr = torch.linspace(1e-3, 0.95, nk, device=dev)
        dirs_w = rays @ Rm.T
        for c in range(0, nk, 16):
            z = fr[c:c + 16, None] * dv[None, :]
            keep = z < FAR
            P = (dirs_w[None] * z[..., None])[keep] + tv
            qf, _ = vox_index(P * s_unit)
            line[flat(qf)] = True
        line |= hits > 0
        h = hits.view(DIMS)
        lv = line.view(DIMS)
        occ = 100.0 * h
        free = lv.float()
        tot = occ + free
        nz = tot > 0
        acc[1][nz] += occ[nz] / tot[nz]
        acc[2][nz] += free[nz] / tot[nz]
        nobs[nz] += 1
        cam = torch.floor((tv * s_unit - LO_T) / VOX).long()
        sl = [slice(max(int(cam[i]) - 1, 0), max(min(int(cam[i]) + 2, DIMS[i]), 0)) for i in range(3)]
        acc[2][sl[0], sl[1], sl[2]] = 1e3

    # ---------------------------------------------------------------- probes
    @torch.no_grad()
    def render(w2c_t):
        pts = slam.params["means3D"]
        tp = (w2c_t @ torch.cat([pts, torch.ones_like(pts[:, :1])], 1).T).T[:, :3]
        im, _, _ = G.Renderer(raster_settings=slam.cam)(**transformed_params2rendervar(slam.params, tp))
        ds, _, _ = G.Renderer(raster_settings=slam.cam)(**transformed_params2depthplussilhouette(slam.params, slam.first_frame_w2c, tp))
        return im, ds[0], ds[1]

    def frame_metrics(w2c_t, color_chw, depth_hw):
        im, dr, sil = render(w2c_t)
        m = depth_hw > 0
        out = {"valid_frac": float(m.float().mean())}
        if int(m.sum()) > 0:
            mse = ((im.clamp(0, 1) - color_chw) ** 2)[:, m].mean().item()
            cov = sil[m] > 0.5
            out.update({"psnr": float(-10 * np.log10(max(mse, 1e-12))), "sil_cov_valid": float(cov.float().mean())})
            if int(cov.sum()) > 0:
                dd = (dr[m][cov] / sil[m][cov].clamp(min=1e-6) - depth_hw[m][cov]).abs()   # silhouette-normalised depth
                out["depth_l1_rt"] = float(dd.mean() * s_unit)
                out["depth_rel"] = float((dd / depth_hw[m][cov]).mean())
                out["depth_rel_median"] = float((dd / depth_hw[m][cov]).median())
        inv = ~m
        if int(inv.sum()) > 0:
            out["floater_frac_invalid"] = float((sil[inv] > 0.5).float().mean())
        return out

    def load(t):
        p = os.path.join(svc_dir, f"svc_{a.scene}_s{a.start}_t{t}.npz")
        rec = MC.load_service_record(p)
        z = np.load(p[:-4] + "_rgb.npz")
        assert int(z["t"]) == t and np.array_equal(z["R"], rec["R"])
        zb, mk = rec["zbuf"].astype(np.float64), rec["mask"].astype(bool)
        valid = mk & (zb > 0) & (zb < FP.SENSOR_RANGE)
        depth = np.where(valid, zb / s_unit, 0.0).astype(np.float32)
        w2c = to_w2c(rec["R"], rec["T"])
        assert tuple(int(x) for x in rec["pose_idx"]) == tuple(keys[t]), (t, rec["pose_idx"], keys[t])
        return rec, torch.from_numpy(z["rgb"]).to(dev), torch.from_numpy(depth).to(dev).reshape(1, FP.H_IMG, FP.W_IMG), \
            torch.from_numpy(w2c).float().to(dev), bool(valid.any())

    ALL = np.array([(i, j, k, e, z) for i in range(grid.n[0]) for j in range(grid.n[1]) for k in range(grid.n[2])
                    for e in range(grid.n_theta) for z in range(grid.n_azim)], dtype=np.int64)

    def fov_count(U, w2c):
        Pc = (U / s_unit) @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]
        u = FP.FX * Pc[:, 0] / z.clamp(min=1e-6) + FP.CX
        v = FP.FX * Pc[:, 1] / z.clamp(min=1e-6) + FP.CY
        return int(((z > 0.5 / s_unit) & (z < FP.SENSOR_RANGE / s_unit) & (u >= 0) & (u < FP.W_IMG) & (v >= 0) & (v < FP.H_IMG)).sum())

    @torch.no_grad()
    def probe(t, rec, frames):
        tb = time.time()
        P = slam.params
        ls = P["log_scales"]
        phantom = torch.isneginf(ls).any(1)
        cams = torch.stack([torch.linalg.inv(f[3])[:3, 3] for f in frames])
        dcam = torch.cdist(P["means3D"].detach(), cams).min(1).values * s_unit
        means_rt = P["means3D"].detach().cpu().numpy() * s_unit
        inbox = np.all((means_rt >= box_lo - 1.0) & (means_rt <= box_hi + 1.0), 1)
        op = torch.sigmoid(P["logit_opacities"])[:, 0]
        res = {"t": t, "n_gauss": int(len(ls)), "n_phantom_neginf_scale": int(phantom.sum()),
               "n_within_0.35rt_of_cam": int((dcam < 0.35).sum()), "phantom_opacity_mean": float(op[phantom].mean()) if phantom.any() else None,
               "frac_means_in_scene_box": float(inbox.mean()), "n_keyframes": len(slam.keyframe_list),
               "means_rt_q01": np.quantile(means_rt, 0.01, 0).tolist(), "means_rt_q99": np.quantile(means_rt, 0.99, 0).tolist(),
               "scene_box_rt": [box_lo.tolist(), box_hi.tolist()]}
        # renders at executed poses (frames <= t with valid depth)
        fm = []
        for (tj, color, depth, w2c, hd) in frames:
            if hd:
                r = frame_metrics(w2c, color.permute(2, 0, 1).float() / 255, depth[0])
                r["t"] = tj
                fm.append(r)
        res["render_executed"] = fm
        smax = torch.exp(ls.max(1).values) * s_unit                         # largest axis scale, runtime units
        fin = torch.isfinite(smax)
        qs = [0.5, 0.9, 0.99, 0.999, 1.0]
        res["scale_rt_q"] = dict(zip(map(str, qs), torch.quantile(smax[fin].float().cpu(), torch.tensor(qs)).tolist()))
        res["n_scale_gt"] = {str(th): int((smax[fin] > th).sum()) for th in (3.5, 7.0, 35.0, 70.0)}
        # floater attribution on the executed frame with the lowest PSNR: silhouette on pixels WITHOUT valid depth, rendered
        # from (a) all Gaussians, (b) only Gaussians with scale > 3.5 rt, (c) all but those, (d) phantoms only
        if fm:
            jw = min((x for x in fm if "psnr" in x), key=lambda x: x["psnr"], default=None)
            if jw is not None:
                fr = [f for f in frames if f[0] == jw["t"]][0]
                inv = ~(fr[2][0] > 0)
                big = (smax > 3.5) & fin
                att = {"t": jw["t"], "psnr": jw["psnr"], "invalid_frac": float(inv.float().mean())}
                keep_all = {k: v for k, v in P.items()}
                for name, msk in (("all", torch.ones_like(big)), ("big_only", big), ("no_big", ~big), ("phantom_only", phantom)):
                    sub = {k: (v[msk] if k in ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales") else v) for k, v in keep_all.items()}
                    if int(msk.sum()) == 0:
                        att[name] = None
                        continue
                    pts = sub["means3D"]
                    tp = (fr[3] @ torch.cat([pts, torch.ones_like(pts[:, :1])], 1).T).T[:, :3]
                    ds, _, _ = G.Renderer(raster_settings=slam.cam)(**transformed_params2depthplussilhouette(sub, slam.first_frame_w2c, tp))
                    sil = ds[1]
                    att[name] = {"sil_on_invalid": float((sil[inv] > 0.5).float().mean()) if inv.any() else None,
                                 "sil_on_valid": float((sil[~inv] > 0.5).float().mean()) if (~inv).any() else None, "n": int(msk.sum())}
                res["floater_attribution"] = att
        # occupancy
        lab = acc.argmax(0)
        unk = lab == 0
        res["occ"] = {"n_unknown": int(unk.sum()), "n_free": int((lab == 2).sum()), "n_occ": int((lab == 1).sum()),
                      "n_unknown_but_observed": int((unk & (nobs > 0)).sum()), "n_observed": int((nobs > 0).sum()),
                      "n_observed_once": int((nobs == 1).sum()), "n_vox": int(unk.numel())}
        rng = np.random.default_rng(FP.seed_of(f"frfaudit|{a.scene}|{a.start}|{t}"))
        U_plan = torch.nonzero(unk)
        if len(U_plan) > 50000:
            U_plan = U_plan[torch.from_numpy(rng.choice(len(U_plan), 50000, replace=False)).to(dev)]
        U_plan = (U_plan.float() + 0.5) * VOX + LO_T
        U_true = torch.nonzero(nobs == 0)
        if len(U_true) > 50000:
            U_true = U_true[torch.from_numpy(rng.choice(len(U_true), 50000, replace=False)).to(dev)]
        U_true = (U_true.float() + 0.5) * VOX + LO_T
        # candidates
        cur = tuple(int(x) for x in rec["pose_idx"])
        legal = [tuple(int(x) for x in r) for r in rec["legal_idx"]]
        uni = []
        for i in rng.permutation(len(ALL)):
            p = tuple(int(x) for x in ALL[i])
            if p != cur and p not in legal and p not in uni:
                uni.append(p)
            if len(uni) >= a.n_uniform:
                break
        cands = [(p, "legal") for p in legal] + [(p, "uniform") for p in uni]
        nxt = tuple(keys[t + 1]) if t + 1 < len(keys) else None
        c2ws = []
        for p, _ in cands:
            R, Tt = grid.RT(p)
            c2ws.append(torch.from_numpy(np.linalg.inv(to_w2c(R, Tt))).float().to(dev))
        out_c = []
        if slam.initialize:
            scores, _ = slam.pose_eval(torch.stack(c2ws))                   # official EIG (lambda 0.1)
            Htr = G.GaussianSLAM.compute_H_train(slam)
            Hinv = torch.reciprocal(Htr + 0.1)
            unseen = Htr.sum(1) < 1e-12
            bigm = (torch.exp(ls.max(1).values) * s_unit > 3.5) & torch.isfinite(ls.max(1).values)   # scale > 0.5 FR
            res["n_never_seen_by_keyframes"] = int(unseen.sum())
            res["n_phantom_and_unseen"] = int((unseen & phantom).sum())
            hs = Htr.sum(1)
            qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            res["H_train_rowsum_q"] = dict(zip(map(str, qs), torch.quantile(hs[~phantom].float().cpu(), torch.tensor(qs)).tolist()))
            res["H_train_frac_any_comp_gt_lambda"] = float((Htr[~phantom] > 0.1).any(1).float().mean())
            res["H_train_frac_all_comp_lt_0.01lambda"] = float((Htr[~phantom] < 0.001).all(1).float().mean())
            for (p, grp), c2w, sc in zip(cands, c2ws, scores.numpy().astype(np.float64)):
                w2c = torch.linalg.inv(c2w)
                Hc = slam.compute_Hessian(w2c, return_points=True, random_gaussian_params=False)
                contrib = (Hc * Hinv).sum(1)
                tot = float(contrib.sum())
                vis_g = Hc.sum(1) > 0
                hc_rows = Hc.sum(1)[vis_g & ~phantom]
                _, _, sil = render(w2c)
                out_c.append({"key": str(list(p)), "group": grp, "executed_next": p == nxt, "eig": float(sc), "eig_recomputed": tot,
                              "eig_phantom": float(contrib[phantom].sum()), "eig_unseen": float(contrib[unseen].sum()),
                              "eig_unseen_nonphantom": float(contrib[unseen & ~phantom].sum()),
                              "eig_clean": float(contrib[~phantom].sum()),
                              "eig_big": float(contrib[bigm].sum()), "eig_clean_nobig": float(contrib[~phantom & ~bigm].sum()),
                              "eig_lambda1e-6": float((Hc * torch.reciprocal(Htr + 1e-6)).sum()),
                              "eig_no_train": float(Hc.sum() / 0.1),
                              "Hc_rowsum_median_visible": float(hc_rows.median()) if len(hc_rows) else 0.0, "n_vis_gauss": int((Hc.sum(1) > 0).sum()),
                              "n_vis_phantom": int(((Hc.sum(1) > 0) & phantom).sum()),
                              "unk_plan_fov": fov_count(U_plan, w2c), "unk_true_fov": fov_count(U_true, w2c),
                              "sil_mean": float(sil.mean()), "novelty": float((sil < 0.5).float().mean()),
                              "dist_l1": int(np.abs(np.array(p[:3]) - np.array(cur[:3])).sum())})
        else:
            for (p, grp), c2w in zip(cands, c2ws):
                w2c = torch.linalg.inv(c2w)
                out_c.append({"key": str(list(p)), "group": grp, "executed_next": p == nxt, "eig": None,
                              "unk_plan_fov": fov_count(U_plan, w2c), "unk_true_fov": fov_count(U_true, w2c),
                              "dist_l1": int(np.abs(np.array(p[:3]) - np.array(cur[:3])).sum())})
        res["cur"] = str(list(cur))
        res["executed_next"] = str(list(nxt)) if nxt else None
        res["candidates"] = out_c
        res["probe_s"] = time.time() - tb
        return res

    want = sorted(set(a.states))
    tmax = max(want) + 1
    frames, out = [], {"scene": a.scene, "start": a.start, "root": a.root, "states": [], "replay_note": "not bitwise (R1)"}
    t0 = time.time()
    for t in range(tmax + 1):
        rec, color, depth, w2c, hd = load(t)
        if t - 1 in want and out["states"] and out["states"][-1]["t"] == t - 1:   # novel-view prediction of frame t
            if slam.initialize and hd:
                r = frame_metrics(w2c, color.permute(2, 0, 1).float() / 255, depth[0])
                out["states"][-1]["predict_next"] = r
        if t == tmax:
            break
        with torch.no_grad():
            if hd and not slam.initialize:
                slam.init(color, depth, torch.linalg.inv(w2c))             # D12 docstring
            if hd:
                slam.track_rgbd(color, depth, w2c)
            if hd:
                occ_update(depth[0], torch.linalg.inv(w2c))
        frames.append((t, color, depth, w2c, hd))
        if t in want:
            st = probe(t, rec, frames)
            out["states"].append(st)
            c = st["candidates"]
            if c and c[0].get("eig") is not None:
                mx = max(abs(x["eig"] - x["eig_recomputed"]) / max(abs(x["eig"]), 1e-9) for x in c)
                print(f"{tag} t={t} n_gauss {st['n_gauss']} phantom {st['n_phantom_neginf_scale']} | cands {len(c)} "
                      f"eig recompute max rel diff {mx:.2e} | phantom share of EIG (median) "
                      f"{np.median([x['eig_phantom'] / max(x['eig'], 1e-9) for x in c]):.3f} | {st['probe_s']:.1f}s", flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"{tag}.json"), "w"), default=float)
    print("done", tag, "%.0fs" % out["seconds"])


if __name__ == "__main__":
    main()
