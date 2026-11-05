"""E2-KSV-v1 -- Known-Surface Visibility constraint (switchable; no training; planner-side, online information only).
ObservedGrid: boolean occupancy of CELL-sized cells (runtime units, y-up) holding every valid depth pixel the agent has actually
observed in its executed frames 0..t (p1_full.full_points: all valid pixels, z < 70, INCLUDING geometry outside the P1 planning
domain, which the P1 fusion drops). For a candidate view, t_obs(ray) = distance to the first occupied cell along the ray
(marched every STEP, cells closer than MIN_T to the camera ignored). A predicted first hit with t_pred > t_obs + MARGIN lies behind
a surface the agent has already observed from somewhere -> it is not visible from this candidate -> it is removed before the value
is computed (h_policy.value_cov sees no hit on that ray). Everything else of the frozen D1 planner is unchanged.
install(): wraps P1Fusion.integrate_record (feeds the grid after the frozen integration) and h_policy.value_cov (used by the D1
first layer and every lookahead layer). Not installed -> the planner is byte-for-byte the frozen D1."""
import json
import os

import numpy as np
import torch

CELL, STEP, MARGIN, MIN_T, PAD_XZ, PAD_Y_LO, PAD_Y_HI = 0.5, 0.25, 1.0, 1.0, 80.0, 30.0, 80.0
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
STATS = {"rays": 0, "cut": 0, "views": 0}
ENABLED = {"on": True}          # switch (fixed-state A/B toggles it per call; closed loop keeps it on)


class ObservedGrid:
    def __init__(self, scene, device):
        sb = json.load(open(os.path.join(DATA, scene, "settings.json")))["scene"]
        lo = np.asarray(sb["x_min"], float) * 10.0
        hi = np.asarray(sb["x_max"], float) * 10.0
        self.lo = np.asarray([lo[0] - PAD_XZ, lo[1] - PAD_Y_LO, lo[2] - PAD_XZ])
        hi = np.asarray([hi[0] + PAD_XZ, hi[1] + PAD_Y_HI, hi[2] + PAD_XZ])
        self.n = np.ceil((hi - self.lo) / CELL).astype(int)
        self.dev = device
        self.occ = torch.zeros(tuple(self.n), dtype=torch.bool, device=device)
        self.lo_t = torch.as_tensor(self.lo, dtype=torch.float64, device=device)
        self.n_t = torch.as_tensor(self.n, dtype=torch.long, device=device)

    def add(self, pts_rt):
        p = torch.as_tensor(np.asarray(pts_rt, np.float64).reshape(-1, 3), device=self.dev)
        ijk = torch.floor((p - self.lo_t) / CELL).long()
        ok = (ijk >= 0).all(-1) & (ijk < self.n_t).all(-1)
        q = ijk[ok]
        self.occ[q[:, 0], q[:, 1], q[:, 2]] = True

    def t_obs(self, o_rt, d_rt, tmax_rt, chunk=2048):
        """first occupied-cell distance (runtime units) per ray, inf if none before tmax_rt. o_rt [3], d_rt [R, 3] unit, tmax_rt [R]."""
        o = torch.as_tensor(np.asarray(o_rt, np.float64), device=self.dev)
        d = torch.as_tensor(np.asarray(d_rt, np.float64), device=self.dev)
        tm = torch.as_tensor(np.asarray(tmax_rt, np.float64), device=self.dev)
        Rn = d.shape[0]
        out = torch.full((Rn,), float("inf"), dtype=torch.float64, device=self.dev)
        M = int(np.ceil(float(tm.max().item()) / STEP)) + 1
        ts = MIN_T + STEP * torch.arange(M, device=self.dev, dtype=torch.float64)
        for s in range(0, Rn, chunk):
            sl = slice(s, min(Rn, s + chunk))
            p = o[None, None] + d[sl, None] * ts[None, :, None]
            ijk = torch.floor((p - self.lo_t) / CELL).long()
            ok = (ijk >= 0).all(-1) & (ijk < self.n_t).all(-1) & (ts[None] <= tm[sl, None])
            q = ijk.clamp(min=0)
            for a in range(3):
                q[..., a] = q[..., a].clamp(max=int(self.n[a]) - 1)
            hit = ok & self.occ[q[..., 0], q[..., 1], q[..., 2]]
            has = hit.any(1)
            first = torch.argmax(hit.int(), dim=1)
            out[sl] = torch.where(has, ts[first], out[sl])
        return out


def visible_mask(grid, fr, v, t_model, k_M):
    """bool [R]: predicted hit (model units) is NOT behind an observed surface (t_pred <= t_obs + MARGIN, or no hit)."""
    import m_common as MC
    o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
    d_rt = MC.to_y_up(v["d"].double().cpu().numpy())
    d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
    if v.get("_ksv_tobs") is None:
        v["_ksv_tobs"] = grid.t_obs(o_rt, d_rt, (v["tcap"].double() * k_M).cpu().numpy())
    tob = v["_ksv_tobs"].to(t_model.device)
    tp = t_model.double() * k_M
    vis = ~torch.isfinite(tp) | (tp <= tob + MARGIN)
    return vis


def install(scene, device, fr_getter):
    """Online install in a planner child: grid fed by integrate_record, value_cov constrained. fr_getter() -> SceneFrame."""
    import m_p1
    import p1_full as PF
    import h_policy as HP
    grid = ObservedGrid(scene, device)
    cls = m_p1.P1Fusion
    orig_int = cls.integrate_record

    def integrate_record(self, rec, *a, **k):
        r = orig_int(self, rec, *a, **k)
        pts, _ = PF.full_points(rec)
        grid.add(pts)
        return r
    cls.integrate_record = integrate_record
    orig_vc = HP.value_cov

    def value_cov(t, v, geo, z0, cfg):
        if not ENABLED["on"]:
            return orig_vc(t, v, geo, z0, cfg)
        vis = visible_mask(grid, geo.fr, v, t, float(geo.fr.k_M))
        STATS["rays"] += int(torch.isfinite(t).sum())
        STATS["cut"] += int((torch.isfinite(t) & ~vis).sum())
        t2 = torch.where(vis, t, torch.full_like(t, float("inf")))
        return orig_vc(t2, v, geo, z0, cfg)
    HP.value_cov = value_cov
    return grid
