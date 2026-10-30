"""NBV scores of predicted views (EXP5 Task 1, protocol §4.1-§4.3).

new_set_from_depth : DistP1 of one predicted view = the frozen abl_score.new_set downstream of SceneCaster.cast, with the
                     cast's (t) replaced by a given first-hit distance (inf = no hit); line by line the same operations.
u_scores           : U_raw / DistU (+ hit / depth components) of K predicted views of one camera.
rank01, gu_scores  : GU = r(DistP1) + lambda r(DistU), average-tie ranks scaled to [0, 1].
"""
import torch

VS = 0.011
R_FIXED = 7296


def new_mask_from_depth(t, o, d, dims, domain, band_dil):
    """Per-ray new-hit mask, flat voxel index, hit voxel (SceneCaster.cast post-processing; t [R], inf = no hit)."""
    hit = torch.isfinite(t)
    p_hit = o[None] + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p_hit / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, dims[a] - 1)
    in_dom = domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    hit = hit & in_dom
    new = hit & ~band_dil[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return new, flat, hit


def new_set_from_depth(t, o, d, dz, k_M, dims, domain, band_dil, z0=None):
    """(ids int64 sorted, weights float64) of the unique new voxels of one view (abl_score.new_set semantics)."""
    new, flat, _ = new_mask_from_depth(t, o, d, dims, domain, band_dil)
    sel = new
    f = flat[sel]
    if f.numel() == 0:
        return f.new_zeros(0), torch.zeros(0, dtype=torch.float64, device=f.device)
    ids, inv = torch.unique(f, return_inverse=True)
    if z0 is None:
        return ids, torch.ones(len(ids), dtype=torch.float64, device=ids.device)
    z = t[sel].double() * float(k_M) * dz[sel]
    w = torch.clamp((z / float(z0)) ** 2, max=1.0)
    wv = torch.zeros(len(ids), dtype=torch.float64, device=ids.device).scatter_reduce(0, inv, w, "amax", include_self=False)
    return ids, wv


def u_scores(new, t, dz, k_M, z0, delta_vox, R=R_FIXED):
    """K predicted views of one camera: new [K, R] bool (DistP1-new hit), t [K, R] ray distance (model units),
    dz [R] (z per runtime ray length). Returns a dict of U_raw, DistU and their hit (H) / depth (D) parts; the
    denominator is the fixed number of rays R (protocol §4.2)."""
    K = new.shape[0]
    assert K >= 2, "U needs K >= 2"
    z = t.double() * float(k_M) * dz[None].double()
    w = torch.where(new, torch.clamp((z / float(z0)) ** 2, max=1.0), torch.zeros_like(z))
    tt = torch.where(new, t.double(), torch.zeros_like(z))
    P = K * (K - 1) // 2
    out = {k: torch.zeros(t.shape[1], dtype=torch.float64, device=t.device) for k in ("rH", "rD", "dH", "dD")}
    for i in range(K):
        for j in range(i + 1, K):
            mis = new[i] != new[j]
            both = new[i] & new[j]
            phi = torch.clamp((tt[i] - tt[j]).abs() / (delta_vox * VS), max=1.0)
            wij = torch.maximum(w[i], w[j])
            out["rH"] += mis.double()
            out["rD"] += both.double() * phi
            out["dH"] += wij * mis.double()
            out["dD"] += wij * both.double() * phi
    r = {k: float(v.sum() / P / R) for k, v in out.items()}
    return {"u_raw": r["rH"] + r["rD"], "u_raw_H": r["rH"], "u_raw_D": r["rD"],
            "dist_u": r["dH"] + r["dD"], "dist_u_H": r["dH"], "dist_u_D": r["dD"]}


def both_new_deltas(new, t):
    """|t_i - t_j| in voxels of all both-new pairs (input of the delta rule, protocol §4.2)."""
    K = new.shape[0]
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            both = new[i] & new[j]
            out.append(((t[i] - t[j]).abs() / VS)[both])
    return torch.cat(out) if out else torch.zeros(0, device=t.device)


def rank01(x):
    """Average-tie ranks scaled to [0, 1]: (rank - 1) / (M - 1); M = 1 -> 0."""
    x = torch.as_tensor(x, dtype=torch.float64)
    M = len(x)
    if M <= 1:
        return torch.zeros(M, dtype=torch.float64)
    order = torch.argsort(x, stable=True)
    r = torch.empty(M, dtype=torch.float64)
    r[order] = torch.arange(1, M + 1, dtype=torch.float64)
    for v in torch.unique(x):
        m = x == v
        if int(m.sum()) > 1:
            r[m] = r[m].mean()
    return (r - 1) / (M - 1)


def gu_scores(distp1, dist_u, lam):
    return (rank01(distp1) + float(lam) * rank01(dist_u)).tolist()
