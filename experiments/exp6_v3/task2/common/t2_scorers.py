"""EXP6 V3 Task 2 -- coverage-consistent candidate scorers (seen2scene env, torch + scipy; protocol.md §3).

All scorers take ONE depth map of one candidate view (the K = 4 predicted first-hit distances t [R] of a sample, or the
true depth t* for diagnosis), the frozen planner geometry of the current state and the real observation history, and
return a scalar. Nothing here reads GT. Notation (model frame, unit = runtime / k_M, voxel VS = 0.011):
  o [3] camera, d [R,3] unit ray directions, dz [R] z-depth per unit runtime ray length, t [R] first hit (inf = none),
  z = t k_M dz = runtime z-depth, q = o + d t hit point, ijk = floor(q / VS) hit voxel, flat = linear id.
Components (each is a per-ray weight; the view score = sum over the UNIQUE hit voxels of the max per-ray weight, exactly
the dedup rule of the frozen DistP1 scoring.new_set_from_depth):
  S0   DistP1  : w = min(1, (z/z0)^2) on rays whose hit voxel is in the domain and NOT in the dilated sparse BAND
  S1   area    : A(q) = (alpha z)^2 dz / cos(theta): the exact surface footprint of one scoring ray (alpha = tangent
                 width of a 4-px block = 8 tan30 / 255; cos(theta) = |n . d| from the local normal of the 64 x 114 hit
                 grid, fallback 1, clipped >= 1/4); Ahat = A / (alpha z0)^2 so that Ahat = S0's weight at z = z0, normal
                 incidence.  S1a = Ahat (uncapped area, with cos)  S1b = min(1,(z/z0)^2) dz / cos (S0 + cos only)
                 S1c = (z/z0)^2 (S0 uncapped, no cos)  S1d = min(Ahat, Asat) with Asat = 1.6 pi eps^2 / (alpha z0)^2
                 (the surface area the ~1.6 coverage samples of a 4 x 4 block can cover: sampling saturation instead
                 of the z0 cap)
  S2   novelty : d_obs(q) = distance (runtime) from q to the nearest HISTORICAL 10 % sample point (records 0..t = the
                 reference set of the coverage metric itself); N_lin = clip(d_obs / (2 eps), 0, 1) (the uncovered
                 fraction of the eps-disc around q, linearised), N_hard = [d_obs > eps], N_disc = exact uncovered lens
                 fraction. S2 replaces the hard band_dil test (no band_dil exclusion); weight = min(1,(z/z0)^2) N.
  S3   consistency: V = 0 for a hit closer than TRUNC (3 voxels) in front of the DENSE known surface along the same ray
                 (t_known100 - t < TRUNC: the known surface itself, re-hit through the sparse band's holes) or behind it
                 (t > t_known100); else 1. The structural checks (hit in dense EMPTY, outside the domain) are counted.
  combos: S0S3, S2S3, S1aS2, S1bS2, S1dS2, S1aS2S3, S1dS2S3 (products of the per-ray weights; dedup as above).
  global (GT diagnosis only, R1 bridge): no region cap (t capped at tcap only), no domain / band test, novelty by
                 N (tree), dedup by a virtual global voxel id: G_hard = sum cap N_hard, G_soft = sum cap N_lin,
                 G_area = sum Ahat N_lin.
"""
import math

import numpy as np
import torch

from seen2scene.futureview import scoring as SCO

VS = 0.011
TRUNC = 3 * VS
GH, GW = 64, 114
ALPHA = 4 * 2 * math.tan(math.radians(30.0)) / 255.0          # tangent-space width of one scoring ray (4 px block)
N_SAMPLES_PER_BLOCK = 1.6                                     # 10 % of 16 pixels
COS_MIN = 0.25
NORMAL_REL_JUMP = 0.15
VARIANTS = ("S0", "S1a", "S1b", "S1c", "S1d", "S2", "S2h", "S2d", "S3", "S0S3", "S2S3", "S1aS2", "S1bS2", "S1dS2", "S1aS2S3", "S1dS2S3")
GLOBAL_VARIANTS = ("G_hard", "G_soft", "G_area")
Y_UP_TO_Z_UP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)


def hit_voxels(t, o, d, dims, domain):
    """(hit, flat, ijk) of a depth map: finite t whose voxel lies inside the grid domain (scoring.new_mask_from_depth
    without the band test)."""
    fin = torch.isfinite(t)
    p = o[None] + d * torch.where(fin, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, dims[a] - 1)
    hit = fin & domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return hit, flat, ijk


def cos_incidence(t, o, d):
    """|n . d| per ray from the local normal of the hit-point grid (64 x 114, row-major); invalid neighbourhoods
    (missing hits, relative depth jump > NORMAL_REL_JUMP) fall back to 1; clipped to >= COS_MIN."""
    fin = torch.isfinite(t)
    q = (o[None] + d * torch.where(fin, t, torch.zeros_like(t))[:, None]).reshape(GH, GW, 3)
    tt = torch.where(fin, t, torch.zeros_like(t)).reshape(GH, GW)
    ok = fin.reshape(GH, GW)
    cos = torch.ones(GH, GW, dtype=t.dtype, device=t.device)
    # central differences on the interior
    qu = q[1:-1, 2:] - q[1:-1, :-2]
    qv = q[2:, 1:-1] - q[:-2, 1:-1]
    tc = tt[1:-1, 1:-1]
    good = ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
    for nb in (tt[1:-1, 2:], tt[1:-1, :-2], tt[2:, 1:-1], tt[:-2, 1:-1]):
        good &= (nb - tc).abs() <= NORMAL_REL_JUMP * tc + 2 * VS
    n = torch.cross(qu, qv, dim=-1)
    nn = n.norm(dim=-1, keepdim=True)
    n = n / torch.where(nn > 0, nn, torch.ones_like(nn))
    c = (n * d.reshape(GH, GW, 3)[1:-1, 1:-1]).sum(-1).abs()
    cos[1:-1, 1:-1] = torch.where(good & (nn[..., 0] > 0), c, torch.ones_like(c))
    gm = torch.zeros(GH, GW, dtype=torch.bool, device=t.device)
    gm[1:-1, 1:-1] = good & (nn[..., 0] > 0)
    return cos.clamp(min=COS_MIN).reshape(-1), gm.reshape(-1)


def novelty_from_dist(d_obs, eps):
    """N_lin, N_hard, N_disc from the distance (runtime units) to the nearest historical sample point."""
    r = d_obs / float(eps)
    n_lin = r.div(2.0).clamp(0.0, 1.0)
    n_hard = (r > 1.0).to(d_obs.dtype)
    x = r.div(2.0).clamp(0.0, 1.0)                        # lens area of two unit discs at distance 2x: 2 acos(x) - 2 x sqrt(1 - x^2)
    lens = 2.0 * torch.acos(x) - 2.0 * x * torch.sqrt((1.0 - x * x).clamp(min=0.0))
    n_disc = 1.0 - lens / math.pi
    return n_lin, n_hard, n_disc


class Hist:
    """Historical observed surface = all 10 % sample points of the records so far (runtime y-up), with a KD-tree."""

    def __init__(self):
        self.pts = []
        self.tree = None
        self.n = 0

    def add(self, sample_points):
        p = np.asarray(sample_points, np.float64)
        if len(p):
            self.pts.append(p)
        self.tree = None

    def build(self):
        from scipy.spatial import cKDTree
        if self.tree is None:
            P = np.concatenate(self.pts) if self.pts else np.zeros((0, 3))
            self.n = len(P)
            self.tree = cKDTree(P) if len(P) else None
        return self.tree

    def dist(self, q_rt, workers=8):
        tree = self.build()
        if tree is None or len(q_rt) == 0:
            return np.full(len(q_rt), np.inf)
        return tree.query(np.asarray(q_rt, np.float64), k=1, workers=workers)[0]


def model_to_runtime_torch(q, fr):
    """Model-frame points [N,3] -> runtime y-up (torch float64)."""
    z = q.double() * float(fr.k_M) + torch.as_tensor(np.asarray(fr.origin_z, np.float64), device=q.device)
    return z @ Y_UP_TO_Z_UP.to(q.device)


def dedup_sum(flat, sel, w):
    """Sum over the unique voxels of the max per-ray weight (frozen DistP1 dedup)."""
    f = flat[sel]
    if f.numel() == 0:
        return 0.0
    ids, inv = torch.unique(f, return_inverse=True)
    wv = torch.zeros(len(ids), dtype=torch.float64, device=f.device).scatter_reduce(0, inv, w[sel].double(), "amax", include_self=False)
    return float(wv.sum())


@torch.no_grad()
def score_view(t, view10, view100, geo10, hist, eps_rt, z0, fr, variants=VARIANTS, t_global=None, want_global=False):
    """All scorer variants of one depth map t [R] (model units, inf = no hit) of one candidate view.
    view10 / view100: geo.view(R, T) dicts of the sparse and dense geometries (o, d, dz, tcap, pr with tmax / t_known);
    geo10: sparse Geometry (dims, domain, caster.band_dil); hist: Hist of the records so far; eps_rt: coverage
    tolerance in runtime units; z0: distance-weight scale (runtime); fr: SceneFrame (model_to_runtime).
    t_global: an uncapped depth map (t* on the whole mesh, capped at tcap only) for the global GT variants.
    Returns ({variant: score}, diag dict)."""
    o, d, dz = view10["o"], view10["d"], view10["dz"]
    k_M = float(fr.k_M)
    dims, domain, band_dil = geo10.dims, geo10.domain, geo10.caster.band_dil
    R = t.shape[0]
    out = {}
    # ---- S0 exactly as deployed (scoring.new_set_from_depth) + the regression of the generic path
    ids0, w0 = SCO.new_set_from_depth(t, o, d, dz, k_M, dims, domain, band_dil, z0)
    out["S0"] = float(w0.sum())
    hit, flat, ijk = hit_voxels(t, o, d, dims, domain)
    in_band = band_dil[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    new0 = hit & ~in_band
    z = t.double() * k_M * dz.double()
    cap = torch.where(hit, (z / float(z0)) ** 2, torch.zeros_like(z)).clamp(max=1.0)
    s0_generic = dedup_sum(flat, new0, cap)
    cos, good = cos_incidence(t, o, d)
    n_good = int(good.sum())
    Ahat = torch.where(hit, (z / float(z0)) ** 2 * dz.double() / cos.double(), torch.zeros_like(z))
    Asat = N_SAMPLES_PER_BLOCK * math.pi * float(eps_rt) ** 2 / (ALPHA * float(z0)) ** 2
    W = {"S1a": Ahat, "S1b": cap * dz.double() / cos.double(), "S1c": torch.where(hit, (z / float(z0)) ** 2, torch.zeros_like(z)), "S1d": Ahat.clamp(max=Asat)}
    for k, w in W.items():
        out[k] = dedup_sum(flat, new0, w)
    # ---- S2 novelty against the historical sample points (runtime)
    q = o[None] + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
    q_rt = model_to_runtime_torch(q[hit], fr).cpu().numpy() if hit.any() else np.zeros((0, 3))
    d_obs = torch.full((R,), float("inf"), dtype=torch.float64, device=t.device)
    if hit.any():
        d_obs[hit] = torch.as_tensor(hist.dist(q_rt), device=t.device, dtype=torch.float64)
    n_lin, n_hard, n_disc = novelty_from_dist(d_obs, eps_rt)
    out["S2"] = dedup_sum(flat, hit, cap * n_lin)
    out["S2h"] = dedup_sum(flat, hit, cap * n_hard)
    out["S2d"] = dedup_sum(flat, hit, cap * n_disc)
    # ---- S3 consistency with the dense known surface along the ray
    tk100 = view100["pr"]["t_known"].double()
    behind = hit & (t.double() > tk100)
    thin = hit & ~behind & ((tk100 - t.double()) < TRUNC)
    V = torch.where(behind | thin, torch.zeros_like(cap), torch.ones_like(cap))
    out["S3"] = dedup_sum(flat, new0, cap * V)
    out["S0S3"] = out["S3"]
    out["S2S3"] = dedup_sum(flat, hit, cap * n_lin * V)
    out["S1aS2"] = dedup_sum(flat, hit, W["S1a"] * n_lin)
    out["S1bS2"] = dedup_sum(flat, hit, W["S1b"] * n_lin)
    out["S1dS2"] = dedup_sum(flat, hit, W["S1d"] * n_lin)
    out["S1aS2S3"] = dedup_sum(flat, hit, W["S1a"] * n_lin * V)
    out["S1dS2S3"] = dedup_sum(flat, hit, W["S1d"] * n_lin * V)
    # structural checks of the hits against the dense map
    st100 = view100.get("struct")
    diag = {"n_hit": int(hit.sum()), "n_new0": int(new0.sum()), "n_in_band": int((hit & in_band).sum()), "n_behind_dense": int(behind.sum()),
            "n_thin_front": int(thin.sum()), "n_good_normals": n_good, "mean_cos": float(cos[hit].mean()) if hit.any() else None,
            "mean_z_rt": float(z[hit].mean()) if hit.any() else None, "mean_novelty_lin": float(n_lin[hit].mean()) if hit.any() else None,
            "frac_novel_hard": float(n_hard[hit].mean()) if hit.any() else None, "s0_generic_absdiff": abs(s0_generic - out["S0"]),
            "n_hit_in_dense_empty": int((st100.reshape(-1)[flat[hit]] == 0).sum()) if st100 is not None and hit.any() else None}
    if want_global and t_global is not None:
        tg = t_global
        fin = torch.isfinite(tg)
        qg = o[None] + d * torch.where(fin, tg, torch.zeros_like(tg))[:, None]
        zg = tg.double() * k_M * dz.double()
        capg = torch.where(fin, (zg / float(z0)) ** 2, torch.zeros_like(zg)).clamp(max=1.0)
        cosg, _ = cos_incidence(tg, o, d)
        Ag = torch.where(fin, (zg / float(z0)) ** 2 * dz.double() / cosg.double(), torch.zeros_like(zg))
        dg = torch.full((R,), float("inf"), dtype=torch.float64, device=t.device)
        if fin.any():
            dg[fin] = torch.as_tensor(hist.dist(model_to_runtime_torch(qg[fin], fr).cpu().numpy()), device=t.device, dtype=torch.float64)
        gl, gh, _ = novelty_from_dist(dg, eps_rt)
        vg = torch.floor(qg / VS).long()
        flatg = (vg[:, 0] * 1000003 + vg[:, 1]) * 1000003 + vg[:, 2]      # virtual global voxel id
        out["G_hard"] = dedup_sum(flatg, fin, capg * gh)
        out["G_soft"] = dedup_sum(flatg, fin, capg * gl)
        out["G_area"] = dedup_sum(flatg, fin, Ag * gl)
        diag["n_hit_global"] = int(fin.sum())
    return out, diag
