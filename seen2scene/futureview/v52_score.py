"""V5.2 (EXP6) camera utility computed directly from the ray geometry belief (no flow, no depth image, no DistP1 call).

For one observed state (geometry `geo` of the planner: D100 P1 structure / coarse grid / domain, S10 band_dil novelty
mask -- exactly the B2 path of the V4 / V5 closed loops) and each legal candidate view:

  S_visible  = sum over fine voxels x (VS grid of DistP1) of  max over rays r of  sum over sub-points of r in x of
               p_first(r, i) / M * w_dist(z) * q_valid
               (first-hit mass of interval i spread uniformly over M sub-points; mass summed inside one (ray, voxel), then
               max over rays per voxel = de-duplication; w_dist(z) = min(1, (z / z0)^2), z = t k_M dz = the runtime
               z-depth, exactly the DistP1 weight; counted only in the scene domain, outside the S10 band_dil (DistP1's
               novelty rule); q_valid = 0 where the D100 P1 has confirmed FREE (EMPTY) -- reported separately)
  P_surf     = sum over fine voxels of max over (r, i) of  b(r, i) (1 - T(r, i)) min(1, dt_i / Delta) w_dist kappa
               (occluded-surface potential: belief b of a surface near t_i times the probability that ray r already hit
               something before I_i; same novelty / domain / FREE masks; kappa = exp(-max(0, t_i - tbar_r) / (128 VS))
               discounts surfaces far behind the expected first hit tbar_r; a heuristic joint proxy, never visible)
  P_space    = sum over coarse cells c (8 VS) of max over (r, i) of  T_mid(r, i) phi(c)
               (unexplored-space potential: UNKNOWN cells of the D100 coarse grid the candidate's rays reach with
               survival T; phi = 1 for frontier cells (26-neighbour of a cell with >= 25 % confirmed FREE), phi = eps for
               other UNKNOWN cells, 0 for observed / out-of-domain cells; 1 - b is never used as free space)
Score(C) = S_vis_hat + lambda (eta P_surf_hat + (1 - eta) P_space_hat), each term divided by max(max over the
candidate set, floor_k) (floors from Train-6 S-c calibration), i.e. in [0, 1].
"""
import torch

from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.futureview import v52_model as V52

VS, S0 = M.VS, V52.S0
DEFAULT = {"n_sub": 8, "thr_vis": 1e-3, "thr_surf": 1e-2, "thr_space": 1e-2, "delta_vox": 8.0, "kappa_vox": 128.0,
           "space_eps": 0.1, "frontier_free_frac": 0.25, "unknown_dom_frac": 0.5, "lambda": 0.0, "eta": 0.5,
           "floors": {"vis": 1e-6, "surf": 1e-6, "space": 1e-6}, "chunk": 8192}


def space_weight(coarse_u8, cfg):
    """coarse_u8 [3, X', Y', Z'] (EMPTY, BAND, in-domain fractions x 255 of the D100 P1) -> phi [X', Y', Z'] float."""
    c = coarse_u8.float() / 255.0
    empty, band, dom = c[0], c[1], c[2]
    unknown = (dom >= cfg["unknown_dom_frac"]) & ((empty + band) < 0.5 * dom)
    free_nb = torch.nn.functional.max_pool3d((empty >= cfg["frontier_free_frac"]).float()[None, None], 3, 1, 1)[0, 0] > 0
    phi = torch.where(unknown & free_nb, torch.ones_like(empty), torch.where(unknown, torch.full_like(empty, cfg["space_eps"]), torch.zeros_like(empty)))
    return phi, {"unknown_cells": int(unknown.sum()), "frontier_cells": int((unknown & free_nb).sum())}


def _fine_index(p, dims):
    ijk = torch.floor(p / VS).long()
    inside = ((ijk >= 0) & (ijk < dims)).all(-1)
    ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), dims - 1)
    return (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2], inside


def _dedup_max(keys, vals):
    if keys.numel() == 0:
        return 0.0, 0
    u, inv = torch.unique(keys, return_inverse=True)
    m = torch.zeros(len(u), dtype=torch.float64, device=vals.device).scatter_reduce(0, inv, vals.double(), "amax", include_self=False)
    return float(m.sum()), int(len(u))


@torch.no_grad()
def score_view(model, mem, view, ctx, k_M, z0, cfg):
    """One candidate view -> raw terms and diagnostics. ctx = per-state tensors (domain, band_dil, struct flat views,
    dims, phi, coarse dims)."""
    o, d, tcap, dz = view["o"].float(), view["d"].float(), view["tcap"].float(), view["dz"].float()
    ch = view["ch"].float() if "ch" in view else G.channels(view["pr"], view["tcap"], view["d"], view["o"]).float().t()
    R = d.shape[0]
    dims, dims_c = ctx["dims"], ctx["dims_c"]
    vk, vv, sk, sv, pk, pv = [], [], [], [], [], []
    diag = {"mass_hit": 0.0, "mass_dom": 0.0, "mass_known": 0.0, "mass_free": 0.0, "mass_counted": 0.0, "p_nohit_mean": 0.0}
    M_ = cfg["n_sub"]
    delta = cfg["delta_vox"] * VS
    for s in range(0, R, cfg["chunk"]):
        sl = slice(s, min(R, s + cfg["chunk"]))
        oo = o[None].expand(sl.stop - sl.start, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.belief(mem, oo, d[sl], tcap[sl], ch[sl])
        pf, logS, a = out["p_first"], out["log_surv"], out["a"]
        edges, t_mid, dt = out["edges"], out["t_mid"], out["dt"]
        b = torch.sigmoid(out["logit_b"])
        dzs = dz[sl]
        diag["p_nohit_mean"] += float(out["p_nohit"].sum())
        diag["mass_hit"] += float(pf.sum())
        rid = torch.arange(sl.start, sl.stop, device=d.device)
        # ---------------- visible: first-hit mass, M sub-points per interval
        r_i, i_i = torch.nonzero(pf >= cfg["thr_vis"], as_tuple=True)
        if r_i.numel():
            frac = (torch.arange(M_, device=d.device).float() + 0.5) / M_
            t_sp = edges[r_i, i_i][:, None] + dt[r_i, i_i][:, None] * frac[None]           # [K, M]
            p_sp = oo[r_i][:, None] + d[sl][r_i][:, None] * t_sp[..., None]
            flat, inside = _fine_index(p_sp, dims)
            in_dom = inside & ctx["domain"][flat]
            known = ctx["band"][flat]
            free = ctx["struct"][flat] == G.EMPTY
            w = torch.clamp((t_sp * float(k_M) * dzs[r_i][:, None] / float(z0)) ** 2, max=1.0)
            m = (pf[r_i, i_i][:, None] / M_).expand_as(t_sp)
            diag["mass_dom"] += float(m[in_dom].sum())
            diag["mass_known"] += float(m[in_dom & known].sum())
            diag["mass_free"] += float(m[in_dom & ~known & free].sum())
            keep = in_dom & ~known & ~free
            diag["mass_counted"] += float(m[keep].sum())
            vk.append(flat[keep] * R + rid[r_i][:, None].expand_as(flat)[keep])
            vv.append((m * w)[keep])
        # ---------------- occluded-surface potential
        T = torch.exp(logS)
        z_m = 1.0 - out["p_nohit"]
        tbar = torch.where(z_m > 1e-3, (pf * t_mid).sum(-1) / z_m.clamp(min=1e-3), tcap[sl])
        kappa = torch.exp(-torch.clamp(t_mid - tbar[:, None], min=0.0) / (cfg["kappa_vox"] * VS))
        wd = torch.clamp((t_mid * float(k_M) * dzs[:, None] / float(z0)) ** 2, max=1.0)
        ps = b * (1.0 - T) * torch.clamp(dt / delta, max=1.0) * wd * kappa
        r_s, i_s = torch.nonzero(ps >= cfg["thr_surf"], as_tuple=True)
        if r_s.numel():
            p_m = oo[r_s] + d[sl][r_s] * t_mid[r_s, i_s][:, None]
            flat, inside = _fine_index(p_m, dims)
            keep = inside & ctx["domain"][flat] & ~ctx["band"][flat] & (ctx["struct"][flat] != G.EMPTY)
            sk.append(flat[keep]); sv.append(ps[r_s, i_s][keep])
        # ---------------- unexplored-space potential (coarse cells)
        T_mid = torch.exp(logS - 0.5 * a)
        p_all = oo[:, None] + d[sl][:, None] * t_mid[..., None]
        ijk = torch.floor(p_all / S0).long()
        inside = ((ijk >= 0) & (ijk < dims_c)).all(-1)
        ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), dims_c - 1)
        cflat = (ijk[..., 0] * dims_c[1] + ijk[..., 1]) * dims_c[2] + ijk[..., 2]
        val = T_mid * ctx["phi"][cflat] * inside
        r_p, i_p = torch.nonzero(val >= cfg["thr_space"], as_tuple=True)
        if r_p.numel():
            pk.append(cflat[r_p, i_p]); pv.append(val[r_p, i_p])
    # ---------------- de-duplication
    s_vis, n_vis = 0.0, 0
    if vk:
        keys, vals = torch.cat(vk), torch.cat(vv)
        if keys.numel():
            u, inv = torch.unique(keys, return_inverse=True)
            ssum = torch.zeros(len(u), dtype=torch.float64, device=vals.device).scatter_add(0, inv, vals.double())
            s_vis, n_vis = _dedup_max(u // R, ssum)
    p_surf, n_surf = _dedup_max(torch.cat(sk), torch.cat(sv)) if sk else (0.0, 0)
    p_space, n_space = _dedup_max(torch.cat(pk), torch.cat(pv)) if pk else (0.0, 0)
    diag["p_nohit_mean"] /= R
    diag.update({"n_vox_visible": n_vis, "n_vox_surf": n_surf, "n_cells_space": n_space})
    return {"s_vis": s_vis, "p_surf": p_surf, "p_space": p_space, "diag": diag}


@torch.no_grad()
def score_state(model, h, geo, views, k_M, z0, cfg):
    cfg = dict(DEFAULT, **(cfg or {}))
    dev = views[0]["d"].device if views else None
    if not views:
        return {"s_vis": [], "p_surf": [], "p_space": [], "diag": [], "state": {}}
    mem = model.memory(h)
    coarse = geo.coarse().to(dev)
    phi, sinfo = space_weight(coarse, cfg)
    ctx = {"dims": torch.tensor(geo.dims, device=dev), "dims_c": torch.tensor(phi.shape, device=dev), "phi": phi.reshape(-1),
           "domain": geo.domain.reshape(-1), "band": geo.caster.band_dil.reshape(-1), "struct": geo.struct.reshape(-1)}
    res = {"s_vis": [], "p_surf": [], "p_space": [], "diag": [], "state": sinfo}
    for v in views:
        r = score_view(model, mem, v, ctx, k_M, z0, cfg)
        for k in ("s_vis", "p_surf", "p_space", "diag"):
            res[k].append(r[k])
    return res


def combine(raw, lam, eta=0.5, floors=None):
    """raw lists -> (scores, parts): each term divided by max(max over candidates, floor) then
    score = vis + lam (eta surf + (1 - eta) space)."""
    floors = floors or DEFAULT["floors"]
    t = {k: torch.as_tensor(raw[k], dtype=torch.float64) for k in ("s_vis", "p_surf", "p_space")}
    n = {}
    for k, f in (("s_vis", "vis"), ("p_surf", "surf"), ("p_space", "space")):
        n[k] = t[k] / max(float(t[k].max()) if len(t[k]) else 0.0, float(floors[f]))
    pot = eta * n["p_surf"] + (1.0 - eta) * n["p_space"]
    score = n["s_vis"] + lam * pot
    share = torch.where(score > 0, lam * pot / score.clamp(min=1e-12), torch.zeros_like(score))
    return score.tolist(), {"vis_hat": n["s_vis"].tolist(), "surf_hat": n["p_surf"].tolist(), "space_hat": n["p_space"].tolist(),
                            "pot_share": share.tolist()}
