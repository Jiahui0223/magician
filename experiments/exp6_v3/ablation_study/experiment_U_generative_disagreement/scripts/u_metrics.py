"""Experiment U -- disagreement metrics of one candidate view from the planner's own K decoded samples (protocol U-v1,
configs/u_config.yaml). Pure functions; no GT, no model call, no P1 change.

Inputs (exactly what the frozen v2_plan_core.score_state returns with keep_hits=True):
  t      [K, R] decoded first-hit ray distance, model units, inf = no hit (geometry.decode)
  hit_u  [K, R] bool, sample k decoded a surface on an UNKNOWN sample of the ray (geometry.decode)
  y      [K, R] raw y_hat of the sampler (only for the offline UNKNOWN-arc diagnostic)
  v      view dict of score_state: o [3], d [R, 3] (model frame, unit), pr (projection: n_u, valid, ...)
  fr     the scene frame (fr.model_to_runtime, fr.k_M)
R = 7,296 unpadded scoring rays (the 64 x 14 padding columns are removed by M.unpad_rays inside V3Predictor.sample).
"""
import numpy as np
import torch

OFF = 1 << 19
CELL = 1.0                          # = u_config value.cell_size = h_policy.value_cov cell grid (runtime units)


def ray_set(v):
    """R_c mask [R]: rays with >= 1 UNKNOWN in-domain sample before t_known (n_u > 0) and a valid march."""
    pr = v["pr"]
    return (pr["n_u"] > 0) & pr["valid"]


def cell_keys(p_rt):
    """runtime points [n, 3] float64 -> int64 cell keys of the value_cov grid (same expression as h_policy.value_cov)."""
    c = np.floor(p_rt / CELL).astype(np.int64) + OFF
    return (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]


def hit_points_rt(t, h, v, fr, k):
    """runtime points [n, 3] of the generated UNKNOWN hits of sample k on rays r with h[k, r]."""
    sel = torch.nonzero(h[k])[:, 0]
    if sel.numel() == 0:
        return np.zeros((0, 3))
    p = v["o"][None].double() + v["d"][sel].double() * t[k, sel].double()[:, None]
    return np.asarray(fr.model_to_runtime(p.cpu().numpy()), np.float64)


@torch.no_grad()
def view_metrics(t, hit_u, v, fr, y=None, diag=False):
    """-> dict with u1, u2, u3 (the frozen planner scores) and logged diagnostics. diag=True adds the offline-only fields
    (UNKNOWN-arc U2, coefficient of variation, pairwise Jaccard, Chamfer) and returns the per-sample cell sets / points."""
    K, R = t.shape
    rc = ray_set(v)
    h = hit_u & rc[None]
    n_rc = int(rc.sum())
    out = {"n_rays": int(R), "n_rc": n_rc}
    # ---------------- U1 hit disagreement
    p = h.double().mean(0)
    bern = 4.0 * p * (1.0 - p)
    out["u1"] = float(bern[rc].mean()) if n_rc else 0.0
    out["u1_sum"] = float(bern[rc].sum())                         # diagnostic: extensive version (fraction-of-image x n_rays)
    out["n_disagree_rays"] = int(((p > 0) & (p < 1) & rc).sum())
    out["n_anyhit_rays"] = int((p > 0).sum())
    out["n_allhit_rays"] = int((p >= 1).sum())
    # ---------------- U2 first-hit depth disagreement (runtime ray distance)
    nh = h.sum(0)
    r2 = nh >= 2
    out["n_r2"] = int(r2.sum())
    kM = float(fr.k_M)
    if out["n_r2"]:
        tt = torch.where(h, t.double() * kM, torch.zeros_like(t, dtype=torch.float64))[:, r2]
        hh = h[:, r2].double()
        n = hh.sum(0)
        mu = tt.sum(0) / n
        var = (hh * (tt - mu[None]) ** 2).sum(0) / (n - 1.0)
        out["u2"] = float(var.mean())
        out["u2_mean_hit_dist_rt"] = float(mu.mean())
        out["u2_median_var"] = float(var.median())
        if diag:
            out["u2_cv"] = float((var.sqrt() / mu.clamp(min=1e-9)).mean())
            # per-ray (var, mean distance) Spearman inside the view: depth-scale check
            if out["n_r2"] >= 10:
                from scipy.stats import spearmanr
                rr = spearmanr(var.cpu().numpy(), mu.cpu().numpy()).correlation
                out["u2_rho_var_dist"] = float(rr) if np.isfinite(rr) else None
            if y is not None:
                s = ((y.double() + 1.0) / 2.0).clamp(0.0, 1.0 - 1e-6)
                ss = torch.where(h, s, torch.zeros_like(s))[:, r2]
                ms = ss.sum(0) / n
                vs = (hh * (ss - ms[None]) ** 2).sum(0) / (n - 1.0)
                out["u2_rel"] = float(vs.mean())
    else:
        out["u2"] = 0.0
        out["u2_mean_hit_dist_rt"] = None
        out["u2_median_var"] = None
        if diag:
            out["u2_cv"] = 0.0
            out["u2_rel"] = 0.0
    # ---------------- U3 geometric surface disagreement on the value_cov cell grid
    sets, pts = [], []
    for k in range(K):
        pk = hit_points_rt(t, h, v, fr, k)
        pts.append(pk)
        sets.append(np.unique(cell_keys(pk)) if len(pk) else np.zeros(0, np.int64))
    allk = np.concatenate(sets) if any(len(s) for s in sets) else np.zeros(0, np.int64)
    if len(allk):
        u, cnt = np.unique(allk, return_counts=True)
        pv = cnt / float(K)
        b = 4.0 * pv * (1.0 - pv)
        out["u3"] = float(b.mean())
        out["u3_sum"] = float(b.sum())
        out["n_cells_union"] = int(len(u))
        out["n_cells_allk"] = int((cnt == K).sum())
    else:
        out["u3"], out["u3_sum"], out["n_cells_union"], out["n_cells_allk"] = 0.0, 0.0, 0, 0
    out["n_cells_k"] = [int(len(s)) for s in sets]
    if diag:
        jac = []
        for i in range(K):
            for j in range(i + 1, K):
                a, b_ = sets[i], sets[j]
                un = len(np.union1d(a, b_))
                jac.append(len(np.intersect1d(a, b_, assume_unique=True)) / un if un else None)
        jj = [x for x in jac if x is not None]
        out["jaccard_pair_mean"] = float(np.mean(jj)) if jj else None
        out["chamfer_rt"] = chamfer(pts)
        return out, sets, pts
    return out


def chamfer(pts, cap=3000):
    """mean over sample pairs of the symmetric mean nearest-neighbour distance (runtime units) between the generated
    UNKNOWN-hit point sets; deterministic stride subsampling to <= cap points; None if fewer than 2 non-empty samples."""
    from scipy.spatial import cKDTree
    P = []
    for p in pts:
        if len(p) == 0:
            continue
        st = int(np.ceil(len(p) / cap))
        P.append(p[::st])
    if len(P) < 2:
        return None
    T = [cKDTree(p) for p in P]
    v = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            a = T[j].query(P[i], k=1)[0].mean()
            b = T[i].query(P[j], k=1)[0].mean()
            v.append(0.5 * (a + b))
    return float(np.mean(v))


def overlap_agreement(sets_a, sets_b):
    """Cross-view noise-index consistency of two views (Phase D, U4 rule): on O = (union_k A_k) & (union_k B_k) (cells both views
    predict at least once), agreement a_kj = |A_k & B_j & O| / |(A_k | B_j) & O|. Returns pooled numerators / denominators of the
    same-index (k = j) and cross-index (k != j) pairs, and |O|."""
    K = len(sets_a)
    ua = np.unique(np.concatenate(sets_a)) if any(len(s) for s in sets_a) else np.zeros(0, np.int64)
    ub = np.unique(np.concatenate(sets_b)) if any(len(s) for s in sets_b) else np.zeros(0, np.int64)
    O = np.intersect1d(ua, ub, assume_unique=True)
    res = {"n_overlap": int(len(O)), "same_num": 0, "same_den": 0, "cross_num": 0, "cross_den": 0}
    if len(O) == 0:
        return res
    A = [np.intersect1d(s, O, assume_unique=True) for s in sets_a]
    B = [np.intersect1d(s, O, assume_unique=True) for s in sets_b]
    for k in range(K):
        for j in range(K):
            num = len(np.intersect1d(A[k], B[j], assume_unique=True))
            den = len(np.union1d(A[k], B[j]))
            if k == j:
                res["same_num"] += num
                res["same_den"] += den
            else:
                res["cross_num"] += num / (K - 1)
                res["cross_den"] += den / (K - 1)
    return res
