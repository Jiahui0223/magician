"""EXP5 Task 3 / E1.2 -- fixed-state BAND-density ablation (seen2scene env, 1 GPU per process; protocol.md v1.0).

Per episode of a stratum (S_B0: Task 2 V2M = B0 trajectories; S_B2: Task 3 B2 closed loops, states not shared with
S_B0), the service records are replayed four times, density x = 100, 75, 50, 10 in that order, ONE P1 map on the GPU at a
time, with the planner's rules: indoor = n_win = 5 region rebuilt from all records at the refresh step r = t - t % 10 plus
the increments r+1..t (only the refreshes some state needs are run; the map of state t depends on nothing else),
outdoor = one scene grid integrated every step; encoder input and noise at r. Per state and density:
  planner side  frozen V2-Mix scores of the density's own predictions with its own band_dil (own); the D100 predictions
                with this density's band_dil (scoring ablation S_x); in the D10 pass the D50 / D75 / D100 predictions with
                the sparse band_dil (input ablation I_x; I10 = own D10 = B0, I100 = S10 = B2)
  map           voxel counts, coarse grid, current-camera and candidate ray statistics (t_known = zero crossing,
                t_band_first, t_empty_end, phantom rates for two denominators and 3 tolerances, entry crossings, GT depth
                edges), timings, peak GPU memory of the pass
  evaluation    (GT read only after all scores of the state) prediction errors vs t*, novelty masks of the GT depth, of
                the D100 predictions and of the own predictions vs the pixel truth (GT point farther than the coverage eps
                from every past 10 % sample point)
Checks (STOP): S_B0: own D10 == stored V2-Mix scores, own D100 == Task 3 B1, I100 == Task 3 B2 (all bitwise);
S_B2: I100 == the online B2 scores of the closed loop; nested sample sets and counts on every frame.
Output W3/e12/fixed/<stratum>/<dataset>/<tag>.jsonl (+ <tag>.episode.json)
Usage: python e12_fixed.py --dataset indoor|outdoor --stratum B0|B2 --part i/n
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e12_paths as P  # noqa: E402

ORIG_W = P.A.W


def setup(stratum):
    P.set_stratum(stratum)
    global A, S2, D, np, torch, F, MC, G, PC, PR, RY, SCO, cKDTree
    A = P.A
    import s2_infer as S2_
    S2 = S2_
    import e12_density as D_
    D = D_
    np, torch = S2.np, S2.torch
    import torch.nn.functional as F_
    F = F_
    MC, G, PC, PR, RY, SCO = S2.MC, S2.G, S2.PC, S2.PR, S2.RY, S2.SCO
    from scipy.spatial import cKDTree as KD
    cKDTree = KD


DENS = (100, 75, 50, 10)
TOL_M = (1, 2, 3)
EDGE_RATIO = 0.85
ENTRY_FRAC = 0.9
GH, GW = 64, 114


def sync():
    torch.cuda.synchronize()
    return time.time()


def phase_of(dataset, t):
    T = 150 if dataset == "indoor" else 100
    return "early" if t < 0.2 * T else ("late" if t >= 0.8 * T else "middle")


def states_of(dataset, stratum):
    b0 = json.load(open(os.path.join(ORIG_W, "manifests", f"states_{dataset}.json")))["states"]
    b0 = [s for s in b0 if any(e["source"] == "V2M" for e in s["sources"])]
    if stratum == "B0":
        sts = b0
    else:
        seen = {s["sid"] for s in b0}
        sts = [s for s in A.load_manifest(dataset)["states"] if s["sid"] not in seen]
    out = {}
    for st in sts:
        out.setdefault(st["tag"], []).append(st)
    return sorted(out.items())


# ------------------------------------------------------------------------------------------------------ ray helpers
def vol_at(geo, o, d, t):
    p = o[None] + d * t[:, None]
    uvw = p / MC.VS - 0.5
    dims = torch.as_tensor(geo.dims, dtype=p.dtype, device=p.device)
    g = 2 * uvw / (dims - 1) - 1
    grid = torch.stack([g[:, 2], g[:, 1], g[:, 0]], -1)[None, :, None, None]
    return F.grid_sample(geo.vol_obs[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, 0, 0]


def struct_march(geo, o, d, tcap, chunk=1024):
    """First BAND sample and first non-EMPTY sample (nearest voxel, cast step) along every ray; model units, inf = none."""
    tmin, tmax, valid, M = G.march_range(o, d, tcap, geo.ext)
    R = d.shape[0]
    tb = torch.full((R,), float("inf"), dtype=d.dtype, device=d.device)
    te = tb.clone()
    dims = geo.struct.shape
    sf, df = geo.struct.reshape(-1), geo.domain.reshape(-1)
    for s in range(0, R, chunk):
        sl = slice(s, min(R, s + chunk))
        ts = G.sample_ts(tmin[sl], M)
        alive = (ts <= tmax[sl, None]) & valid[sl, None]
        p = o[None, None] + d[sl, None] * ts[..., None]
        ijk = torch.floor(p / MC.VS).long()
        inside = alive & (ijk >= 0).all(-1) & (ijk[..., 0] < dims[0]) & (ijk[..., 1] < dims[1]) & (ijk[..., 2] < dims[2])
        flat = ((ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2]).clamp(0, sf.numel() - 1)
        stv, dm = sf[flat], df[flat] & inside
        band = dm & (stv == G.BAND)
        nonempty = dm & (stv != G.EMPTY)
        inf = torch.full_like(ts[:, 0], float("inf"))
        tb[sl] = torch.where(band.any(1), ts.gather(1, torch.argmax(band.int(), 1)[:, None])[:, 0], inf)
        te[sl] = torch.where(nonempty.any(1), ts.gather(1, torch.argmax(nonempty.int(), 1)[:, None])[:, 0], inf)
    return tb, te


def depth_edge(z):
    Z = np.where(np.isfinite(z), z, np.inf).reshape(GH, GW)
    pad = np.pad(Z, 1, constant_values=np.inf)
    nb = np.min(np.stack([pad[1 + a:1 + a + GH, 1 + b:1 + b + GW] for a in (-1, 0, 1) for b in (-1, 0, 1)]), 0)
    return (np.isfinite(Z) & (nb < EDGE_RATIO * Z)).reshape(-1)


def ray_block(geo, v, ztrue, k_M, u, trunc_rt):
    """Counts / medians of one camera. ztrue: GT z-depth per ray (runtime, inf = none)."""
    pr = v["pr"]
    dz = v["dz"].double().cpu().numpy()
    tk = pr["t_known"]
    zk = tk.double().cpu().numpy() * k_M * dz
    gt = np.isfinite(ztrue) & (ztrue < MC.SENSOR_RANGE)
    known = np.isfinite(zk)
    kg = known & gt
    err = np.where(kg, zk - np.where(gt, ztrue, 0.0), 0.0)
    tol = {m: m * trunc_rt + 0.01 * np.where(gt, ztrue, 0.0) for m in TOL_M}
    ph = {m: kg & (err < -tol[m]) for m in TOL_M}
    P1m = ph[1]
    edge = depth_edge(np.where(gt, ztrue, np.inf))
    entry = np.zeros_like(P1m)
    if P1m.any():
        idx = torch.as_tensor(np.flatnonzero(P1m), device=tk.device)
        val = vol_at(geo, v["o"], v["d"][idx], (tk[idx] - G.H_STEP).clamp(min=0)).double().cpu().numpy()
        entry[np.flatnonzero(P1m)] = val >= ENTRY_FRAC * MC.TRUNC
    tb, te = struct_march(geo, v["o"], v["d"], v["tcap"])
    zb = tb.double().cpu().numpy() * k_M * dz
    ze = te.double().cpu().numpy() * k_M * dz

    def med(x, m):
        return float(np.median(x[m])) if m.any() else None
    cls = pr["cls"].cpu().numpy()
    out = {"rays": int(len(zk)), "gt": int(gt.sum()), "known": int(known.sum()), "known_gt": int(kg.sum()),
           "match": int((kg & (np.abs(err) <= tol[1])).sum()), "behind": int((kg & (err > tol[1])).sum()),
           "edge_gt": int((gt & edge).sum()), "known_gt_edge": int((kg & edge).sum()),
           "phantom_entry": int((P1m & entry).sum()), "phantom_edge": int((P1m & edge).sum()),
           "abs_err_med_u": med(np.abs(err) / u, kg), "phantom_mag_med_trunc": med(-err / trunc_rt, P1m),
           "off_band_first_med": med((zb - ztrue) / trunc_rt, gt & np.isfinite(zb)),
           "off_zero_med": med((zk - ztrue) / trunc_rt, kg),
           "off_empty_end_med": med((ze - ztrue) / trunc_rt, gt & np.isfinite(ze)),
           "band_first_gt": int((gt & np.isfinite(zb)).sum()), "empty_end_gt": int((gt & np.isfinite(ze)).sum()),
           "band_first_front": int((gt & np.isfinite(zb) & (zb - np.where(gt, ztrue, 0) < -tol[1])).sum()),
           "n_seg_sum": float(pr["n_seg"].float().sum()), "lu_sum_u": float((pr["n_u"].double().cpu().numpy() * G.H_STEP * k_M).sum() / u),
           "cls": [int((cls == c).sum()) for c in (G.CLS_UNKNOWN, G.CLS_KNOWN, G.CLS_EMPTY)]}
    for m in TOL_M:
        out[f"phantom@{m}"] = int(ph[m].sum())
    return out


def pool_blocks(bs):
    out = {}
    for k in bs[0]:
        if k == "cls":
            out[k] = np.sum([b[k] for b in bs], 0).tolist()
        elif k.endswith("_med") or k.endswith("_med_u") or k.endswith("_med_trunc"):
            vals = [b[k] for b in bs if b[k] is not None]
            out[k] = float(np.mean(vals)) if vals else None
        else:
            out[k] = float(np.sum([b[k] for b in bs]))
    out["n_cams"] = len(bs)
    return out


# ------------------------------------------------------------------------------------------- scoring / mask helpers
def distp1_k(views, hits, geo, k_M, z0, idx=None):
    out = []
    for j, (v, hh) in enumerate(zip(views, hits)):
        hh = hh.to(v["o"].device)
        ws = []
        for k in range(hh.shape[0]):
            _, w = SCO.new_set_from_depth(hh[k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
            ws.append(float(w.sum()))
        out.append(ws)
    return out


def voxel_flags(geo, v, t):
    """(hit in domain, in BAND, in band_dil) of the points o + d t (numpy bool [R])."""
    hit = torch.isfinite(t)
    p = v["o"][None] + v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / MC.VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    hit = hit & geo.domain[i, j, k]
    return hit.cpu().numpy(), (hit & geo.caster.obs_band[i, j, k]).cpu().numpy(), (hit & geo.caster.band_dil[i, j, k]).cpu().numpy()


def mask_counts(hit, exb, exd, truth_new, gt_ok, acc=None):
    tn, to = truth_new & gt_ok, ~truth_new & gt_ok
    c = {"hit": int(hit.sum()), "excl_band": int(exb.sum()), "excl_dil": int(exd.sum()),
         "hit_truth_new": int((hit & tn).sum()), "hit_truth_old": int((hit & to).sum()),
         "false_excl_dil": int((exd & tn).sum()), "false_excl_band": int((exb & tn).sum()),
         "false_reward_dil": int((hit & ~exd & to).sum()), "false_reward_band": int((hit & ~exb & to).sum())}
    if acc is not None:
        c["false_excl_dil_acc"] = int((exd & tn & acc).sum())
    return c


def add(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


def pred_metrics(t_k, tgt, dz, k_M, u, z0, truth_new, gt_ok):
    """Errors of K predicted views vs t* on rays where both hit (sums; z in runtime units, err in u)."""
    zk = t_k.double().cpu().numpy() * k_M * dz[None]
    zg = tgt * k_M * dz
    ph = np.isfinite(zk)
    out = {"n_gt": int(gt_ok.sum()) * zk.shape[0], "n_pred_hit": int(ph.sum()), "hit_agree": int((ph == gt_ok[None]).sum()),
           "n_rays": int(zk.size)}
    both = ph & gt_ok[None]
    e = np.where(both, zk - np.where(gt_ok, zg, 0)[None], 0.0) / u
    out.update({"n_both": int(both.sum()), "sae": float(np.abs(e).sum()), "sse": float((e ** 2).sum()),
                "srel": float((np.abs(e) * u / np.where(gt_ok, zg, 1.0)[None])[both].sum())})
    rz = np.where(gt_ok, zg / z0, np.inf)
    for nm, lo, hi in (("b0", 0, 0.5), ("b1", 0.5, 1), ("b2", 1, 2), ("b3", 2, np.inf)):
        m = both & ((rz >= lo) & (rz < hi))[None]
        out[f"n_{nm}"], out[f"sae_{nm}"] = int(m.sum()), float(np.abs(e)[m].sum())
    for nm, sel in (("new", truth_new & gt_ok), ("old", ~truth_new & gt_ok)):
        m = both & sel[None]
        out[f"n_{nm}"], out[f"sae_{nm}"] = int(m.sum()), float(np.abs(e)[m].sum())
    allh = ph.all(0)
    out["n_allhit"] = int(allh.sum())
    out["hit_disagree"] = int((ph.any(0) & ~allh).sum())
    if allh.any():
        zz = zk[:, allh]
        out["rel_std_sum"] = float((zz.std(0) / np.maximum(zz.mean(0), 1e-9)).sum())
        m2 = allh & gt_ok
        if m2.sum() >= 10:
            from scipy.stats import spearmanr
            zz2 = zk[:, m2]
            r = spearmanr(zz2.std(0), np.abs(zz2.mean(0) - zg[m2])).correlation
            out["spread_err_rho"] = float(r) if np.isfinite(r) else None
    return out


# -------------------------------------------------------------------------------------------------------- episode
class Episode:
    def __init__(self, dataset, stratum, tag, sts, core, dev, geom_ref):
        self.dataset, self.stratum, self.tag, self.core, self.dev = dataset, stratum, tag, core, dev
        self.sts = {s["t"]: s for s in sts}
        self.ep = S2.episode_of(dataset, sts[0])
        self.rp = S2.StateReplay(dataset, self.ep, "V2M", {}, dev)
        self.T = max(self.sts)
        self.recs = [MC.load_service_record(A.svc_path(dataset, "V2M", self.ep, t)) for t in range(self.T + 1)]
        for t, r in enumerate(self.recs):
            assert int(r["t"]) == t
        self.u = 7.0 if dataset == "indoor" else 1.0
        self.geom_ref = geom_ref
        self.hits = {x: {} for x in DENS}
        self.rows = {s["sid"]: {"sid": s["sid"], "scene": s["scene"], "start": s["start"], "t": s["t"], "tag": tag,
                                "stratum": stratum, "phase": phase_of(dataset, s["t"]), "d": {}} for s in sts}
        self.truth = {}
        self.ep_info = {"tag": tag, "stratum": stratum, "passes": {}, "frames": self.frame_counts()}

    def frame_counts(self):
        rows = []
        for r in self.recs:
            e50, _, nv, n10 = D.extra_pixels(r, 50)
            e75 = D.extra_pixels(r, 75)[0]
            e100 = D.extra_pixels(r, 100)[0]
            assert len(e50) + n10 == (nv * 50) // 100 and len(e75) + n10 == (nv * 75) // 100 and len(e100) + n10 == nv
            assert np.isin(e50, e75).all() and np.isin(e75, e100).all() and not np.isin(r["sample_pixel_flat"], e100).any()
            rows.append([nv, n10, n10 + len(e50), n10 + len(e75), n10 + len(e100)])
        return rows

    def oracle(self, st):
        return np.load(os.path.join(A.state_dir(self.dataset, st["sid"]), "oracle.npz"))

    def truth_of(self, st, orc, views, k_M):
        """Pixel truth per candidate of C: GT point farther than eps from every past 10 % sample point (evaluation)."""
        if st["sid"] in self.truth:
            return self.truth[st["sid"]]
        pts = np.concatenate([np.asarray(r["sample_points"], np.float64) for r in self.recs[:st["t"] + 1]])
        tree = cKDTree(pts)
        if self.dataset == "indoor":
            eps = 0.08 * float(orc["k_M"])
        else:
            eps = float(orc["eps"])
        uu, vv = MC.scoring_pixel_grid()
        rec = self.recs[st["t"]]
        out = {}
        for j in st["cand"]:
            ts = orc["tstar"][j].astype(np.float64)
            d_y, dz, c = MC.pixel_rays(np.asarray(rec["legal_R"][j], np.float64), np.asarray(rec["legal_T"][j], np.float64), uu, vv)
            zg = ts * k_M * dz
            ok = np.isfinite(ts) & (zg < MC.SENSOR_RANGE)
            p = c[None] + d_y * np.where(ok, ts * k_M, 0.0)[:, None]
            dd, _ = tree.query(p[ok], k=1, distance_upper_bound=eps)
            new = np.zeros(len(ts), bool)
            new[np.flatnonzero(ok)] = ~np.isfinite(dd)
            out[j] = (new, ok, zg)
        self.truth[st["sid"]] = (out, eps)
        return self.truth[st["sid"]]

    def run_pass(self, x):
        cls = D.fusion_class(x)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tm = {"rebuild_s": [], "step_s": [], "encode_s": [], "geometry_s": []}
        want = sorted(self.sts)
        if self.dataset == "indoor":
            blocks = sorted({t - t % 10 for t in want})
            for r in blocks:
                fr, dims, dom, _, _ = self.rp.F.region_geometry(self.rp.card, self.recs[r]["cam_center"])
                region = (fr, dims, dom)
                fus = None
                torch.cuda.empty_cache()
                t0 = sync()
                fus = cls(*region, device=self.dev)
                for q in range(r + 1):
                    fus.integrate_record(self.recs[q])
                tm["rebuild_s"].append(sync() - t0)
                h, eps = self.refresh(region, fus, r, tm)
                last = max(t for t in want if t - t % 10 == r)
                for t in range(r, last + 1):
                    if t > r:
                        t0 = sync()
                        fus.integrate_record(self.recs[t])
                        tm["step_s"].append(sync() - t0)
                    if t in self.sts:
                        self.state(x, self.sts[t], region, fus, h, eps, tm)
                del fus
        else:
            fr, dims, dom, _ = self.rp.frame
            region = (fr, dims, dom)
            fus = cls(*region, device=self.dev)
            blocks = {t - t % 10 for t in want}
            h = eps = None
            for t in range(self.T + 1):
                t0 = sync()
                fus.integrate_record(self.recs[t])
                tm["step_s"].append(sync() - t0)
                if t in blocks:
                    h, eps = self.refresh(region, fus, t, tm)
                if t in self.sts:
                    self.state(x, self.sts[t], region, fus, h, eps, tm)
            del fus
        torch.cuda.synchronize()
        self.ep_info["passes"][str(x)] = {k: (float(np.mean(v)) if v else None) for k, v in tm.items()}
        self.ep_info["passes"][str(x)]["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()

    def refresh(self, region, fus, r, tm):
        t0 = sync()
        geo = PC.Geometry.from_fusion(*region, fus, self.dev)
        tm["geometry_s"].append(sync() - t0)
        t0 = sync()
        h = self.core.pred.encode(geo.coarse())
        tm["encode_s"].append(sync() - t0)
        eps = PR.noise(self.dev, A.K_MAIN, lambda kk, tr=r: RY.sha_seed(self.rp.seed_key(tr, kk)))
        del geo
        return h, eps

    def state(self, x, st, region, fus, h, eps, tm):
        sid, row = st["sid"], self.rows[st["sid"]]
        rec = self.recs[st["t"]]
        t0 = sync()
        geo = PC.Geometry.from_fusion(*region, fus, self.dev)
        tm["geometry_s"].append(sync() - t0)
        k_M = float(geo.fr.k_M)
        z0 = self.rp.z0
        t0 = sync()
        out, tmm, views = PC.score_state(self.core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
        tmm["total_s"] = sync() - t0
        self.hits[x][sid] = [hh["t"].detach().cpu() for hh in out["hits"]]
        dr = row["d"].setdefault(str(x), {})
        dr["own_k"] = out["distp1_k"]
        dr["score_timing"] = tmm
        dr["S_k"] = out["distp1_k"] if x == 100 else distp1_k(views, self.hits[100][sid], geo, k_M, z0)
        if x == 10:
            row["I_k"] = {str(y): (out["distp1_k"] if y == 10 else distp1_k(views, self.hits[y][sid], geo, k_M, z0)) for y in (50, 75, 100)}
            row["I_k"]["10"] = out["distp1_k"]
            self.check(st, row)
        # ---- map statistics
        s, dom = geo.struct, geo.domain
        coarse = geo.coarse().float() / 255.0
        cd = coarse[2] > 0
        nd = int(dom.sum())
        dr["map"] = {"domain": nd, "band": int(((s == G.BAND) & dom).sum()), "empty": int(((s == G.EMPTY) & dom).sum()),
                     "unknown": int(((s == G.UNKNOWN) & dom).sum()), "band_over_carved": int(((fus.cnt > 0) & fus.empty).sum()),
                     "coarse_band_mean": float(coarse[1][cd].mean()), "coarse_empty_mean": float(coarse[0][cd].mean()),
                     "coarse_band_cells": float((coarse[1][cd] > 0).float().mean())}
        # ---- evaluation side (GT) from here on
        orc = self.oracle(st)
        if self.dataset == "indoor":
            assert abs(float(orc["k_M"]) - k_M) < 1e-9 * k_M, (float(orc["k_M"]), k_M)
        truth, eps_rt = self.truth_of(st, orc, views, k_M)
        row["eps_rt"] = eps_rt
        trunc_rt = MC.TRUNC * k_M
        uu, vv = MC.scoring_pixel_grid()
        iu, iv = np.round(uu).astype(int), np.round(vv).astype(int)
        zcur = np.asarray(rec["zbuf"], np.float64)[iv, iu]
        zcur = np.where(np.asarray(rec["mask"], bool)[iv, iu], zcur, np.inf)
        vc = geo.view(rec["R"], rec["T"])
        dr["cur"] = ray_block(geo, vc, zcur, k_M, self.u, trunc_rt)
        cb, pm, mk = [], {}, {"gt": {}, "pred100": {}, "own": {}}
        for j in st["cand"]:
            v = views[j]
            new, ok, zg = truth[j]
            dzn = v["dz"].double().cpu().numpy()
            cb.append(ray_block(geo, v, np.where(ok, zg, np.inf), k_M, self.u, trunc_rt))
            pq = pred_metrics(self.hits[x][sid][j], orc["tstar"][j].astype(np.float64), dzn, k_M, self.u, z0, new, ok)
            pm.setdefault("rho_list", []).append(pq.pop("spread_err_rho", None))
            add(pm, pq)
            tgt = torch.as_tensor(np.where(ok, orc["tstar"][j].astype(np.float64), np.inf), dtype=v["o"].dtype, device=v["o"].device)
            fg = voxel_flags(geo, v, tgt)
            add(mk["gt"], mask_counts(*fg, new, ok))
            row.setdefault("masks", {}).setdefault(str(j), {}).setdefault("gt", {})[str(x)] = np.packbits(fg[2]).tobytes().hex()
            for nm, hk in (("pred100", self.hits[100][sid][j]), ("own", self.hits[x][sid][j])):
                for k in range(hk.shape[0]):
                    tk = hk[k].to(v["o"].device)
                    zp = tk.double().cpu().numpy() * k_M * dzn
                    acc = ok & np.isfinite(zp) & (np.abs(zp - np.where(ok, zg, 0)) <= trunc_rt + 0.01 * np.where(ok, zg, 0))
                    fl = voxel_flags(geo, v, tk)
                    add(mk[nm], mask_counts(*fl, new, ok, acc))
                    if nm == "pred100":
                        row["masks"][str(j)].setdefault("pred100", {}).setdefault(str(x), []).append(np.packbits(fl[2]).tobytes().hex())
        dr["cand"] = pool_blocks(cb)
        rl = [r for r in pm.pop("rho_list") if r is not None]
        pm["spread_err_rho_mean"] = float(np.mean(rl)) if rl else None
        dr["pred"] = pm
        dr["masks"] = mk
        dr["peak_gb_sofar"] = torch.cuda.max_memory_allocated() / 2 ** 30
        del geo, views, out
        if x == 10:
            self.finish_masks(st, row)

    def finish_masks(self, st, row):
        """Per pixel: excluded by band_dil of density x but not of D10, for the GT depth and the D100 predictions;
        how many of them are truth-new / accurate / at a GT depth edge (evaluation only)."""
        truth, _ = self.truth[st["sid"]]
        res = {}
        for j in st["cand"]:
            new, ok, zg = truth[j]
            edge = depth_edge(np.where(ok, zg, np.inf))
            m = row["masks"][str(j)]
            n = len(new)
            unpack = lambda hx: np.unpackbits(np.frombuffer(bytes.fromhex(hx), np.uint8))[:n].astype(bool)  # noqa: E731
            for x in (50, 75, 100):
                for src in ("gt", "pred100"):
                    a = [unpack(h) for h in (m[src][str(x)] if src == "pred100" else [m[src][str(x)]])]
                    b = [unpack(h) for h in (m[src]["10"] if src == "pred100" else [m[src]["10"]])]
                    for ax, bx in zip(a, b):
                        only = ax & ~bx
                        add(res.setdefault(f"{src}_{x}", {}), {"only_dense": int(only.sum()), "only_dense_truth_new": int((only & new & ok).sum()),
                                                                "only_dense_edge": int((only & edge).sum()),
                                                                "only_dense_truth_new_edge": int((only & new & ok & edge).sum()),
                                                                "only_sparse": int((bx & ~ax).sum())})
        row["dense_vs_sparse_excl"] = res
        del row["masks"]

    def check(self, st, row):
        own10 = row["I_k"]["10"]
        if self.stratum == "B0":
            stored = json.load(open(os.path.join(A.state_dir(self.dataset, st["sid"]), "v2.json")))["models"]["V2M"]["distp1_k"]
            g = self.geom_ref[st["sid"]]
            ok = {"D10_eq_stored_V2M": own10 == stored, "D100_own_eq_B1": row["d"]["100"]["own_k"] == g["B1"],
                  "I100_eq_B2": row["I_k"]["100"] == g["B2"]}
        else:
            onl = st["sources"][0]["online"]["scores"]
            ok = {"I100_eq_online_B2": [float(np.mean(x)) for x in row["I_k"]["100"]] == onl}
        row["repro"] = ok
        if not all(ok.values()):
            raise SystemExit(f"STOP: reproduction failed {self.dataset} {self.stratum} {st['sid']} {ok}")

    def run(self):
        for x in DENS:
            self.run_pass(x)
        return [self.rows[s["sid"]] for s in sorted(self.sts.values(), key=lambda s: s["t"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["indoor", "outdoor"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    setup(a.stratum)
    pi, pn = map(int, a.part.split("/"))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    core = S2.load_cores(dev)["V2M"]
    geom_ref = {}
    if a.stratum == "B0":
        for p in glob.glob(os.path.join(P.W3, "geometry", a.dataset, "part*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                geom_ref[r["sid"]] = r["scores_k"]
    od = os.path.join(P.E12, "fixed", a.stratum, a.dataset)
    os.makedirs(od, exist_ok=True)
    eps_ = states_of(a.dataset, a.stratum)
    if a.tags:
        eps_ = [e for e in eps_ if e[0] in a.tags.split(",")]
    for tag, sts in eps_[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        E = Episode(a.dataset, a.stratum, tag, sts, core, dev, geom_ref)
        rows = E.run()
        E.ep_info["seconds"] = time.time() - tb
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        json.dump(E.ep_info, open(os.path.join(od, f"{tag}.episode.json"), "w"))
        os.replace(po + ".tmp", po)
        print(a.dataset, a.stratum, tag, len(rows), "states %.0fs" % (time.time() - tb),
              "peak", {k: round(v["peak_gb"], 1) for k, v in E.ep_info["passes"].items()}, flush=True)
        del E
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
