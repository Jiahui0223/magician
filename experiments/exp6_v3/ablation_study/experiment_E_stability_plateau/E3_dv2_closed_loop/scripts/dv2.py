"""EXP6 V3 / E3 -- D-v2 = D1 + Known-Surface Visibility constraint with a ray-FOOTPRINT occlusion test (planner side; no training).

Online information only: every valid depth pixel (z < 70, full density = D100 pixels, p1_full.full_points) of the agent's
EXECUTED frames 0..t, INCLUDING geometry outside the P1 planning domain (E2 audit: the missing constraint is there; inside the
domain the frozen decoder already enforces t <= t_known, D-v2 adds the observed geometry the P1 grid never holds and the band
leaks). ObservedGrid = occupancy of CELL cells (runtime units, y-up) of those points.

Footprint test (handles holes, wall boundaries, sparse returns and voxel error; an observed cell is NOT an infinite wall):
a ray is blocked at depth D only if the whole bundle {centre ray, 4 parallel rays offset by FOOTPRINT perpendicular to it} has
met an occupied cell by D:  t_obs = max over the bundle of the first occupied distance (inf if any bundle ray is free).
  - a hole narrower than 2 x FOOTPRINT or an edge closer than FOOTPRINT leaves one bundle ray free -> no cut;
  - an isolated / sparse cell cannot block a bundle -> no cut;
  - a ray grazing an observed plane clips its voxel layer early, the offset ray on the far side enters where the centre ray
    really hits -> predicted hits ON that surface are not cut (the S10 novelty rule is left to decide them, as in D1).
FOOTPRINT = 0 and SUPPORT = 1 reproduce E2 KSV-v1 (single ray, every occupied cell). SUPPORT > 1 (3x3x3 neighbour count) is kept
only for the v1 design record (configs/dv2_v1.yaml), D-v2 uses SUPPORT = 1.

A predicted first hit with t_pred > t_obs + MARGIN is not visible from that camera and is removed BEFORE h_policy.value_cov
forms its (cell id, weight) set, i.e. before the novelty test, the cell de-duplication and the two-step union.
Layers: mode "full" = first layer and every lookahead layer (d_policy.lookahead -> ensure -> HP.value_cov), each camera with its
own rays; "first" = first layer only (diagnostic ablation); "off" = value_cov is the frozen function -> D1 bitwise.
The lookahead never integrates records, so the grid only ever holds real executed observations (hypothetical second-step views
are decoded against the current P1 by the frozen code and filtered against the current real grid)."""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
CFG = {"cell": 0.5, "step": 0.25, "margin": 1.0, "min_t": 1.0, "support": 1, "footprint": 0.5, "pad_xz": 80.0, "pad_y_lo": 30.0, "pad_y_hi": 80.0}
STATE = {"mode": "off", "layer": 1}
STATS = {"rays": [0, 0, 0], "cut": [0, 0, 0], "calls": [0, 0, 0], "sec": [0.0, 0.0, 0.0]}      # index = layer (1 first, 2 lookahead)


class ObservedGrid:
    def __init__(self, scene, device):
        sb = json.load(open(os.path.join(DATA, scene, "settings.json")))["scene"]
        lo = np.asarray(sb["x_min"], float) * 10.0
        hi = np.asarray(sb["x_max"], float) * 10.0
        c = CFG
        self.lo = np.asarray([lo[0] - c["pad_xz"], lo[1] - c["pad_y_lo"], lo[2] - c["pad_xz"]])
        hi = np.asarray([hi[0] + c["pad_xz"], hi[1] + c["pad_y_hi"], hi[2] + c["pad_xz"]])
        self.n = np.ceil((hi - self.lo) / c["cell"]).astype(int)
        self.dev = device
        self.occ = torch.zeros(tuple(self.n), dtype=torch.bool, device=device)
        self.rel = None
        self.lo_t = torch.as_tensor(self.lo, dtype=torch.float64, device=device)
        self.n_t = torch.as_tensor(self.n, dtype=torch.long, device=device)

    def add(self, pts_rt):
        p = torch.as_tensor(np.asarray(pts_rt, np.float64).reshape(-1, 3), device=self.dev)
        ijk = torch.floor((p - self.lo_t) / CFG["cell"]).long()
        ok = (ijk >= 0).all(-1) & (ijk < self.n_t).all(-1)
        q = ijk[ok]
        self.occ[q[:, 0], q[:, 1], q[:, 2]] = True
        self.rel = None

    def reliable(self):
        if self.rel is None:
            S = int(CFG["support"])
            if S <= 1:
                self.rel = self.occ
            else:
                x = self.occ.half()[None, None]
                cnt = F.conv3d(x, torch.ones((1, 1, 3, 3, 3), dtype=torch.half, device=self.dev), padding=1)[0, 0]
                self.rel = self.occ & (cnt.round() >= S)
                del x, cnt
        return self.rel

    def _march(self, occ, o, d, tm, chunk):
        """first occupied distance per ray (o [R, 3], d [R, 3] unit, tm [R]); inf if none before tm."""
        Rn = d.shape[0]
        out = torch.full((Rn,), float("inf"), dtype=torch.float64, device=self.dev)
        M = int(np.ceil(float(tm.max().item()) / CFG["step"])) + 1
        ts = CFG["min_t"] + CFG["step"] * torch.arange(M, device=self.dev, dtype=torch.float64)
        for s in range(0, Rn, chunk):
            sl = slice(s, min(Rn, s + chunk))
            p = o[sl, None] + d[sl, None] * ts[None, :, None]
            ijk = torch.floor((p - self.lo_t) / CFG["cell"]).long()
            ok = (ijk >= 0).all(-1) & (ijk < self.n_t).all(-1) & (ts[None] <= tm[sl, None])
            q = ijk.clamp(min=0)
            for a in range(3):
                q[..., a] = q[..., a].clamp(max=int(self.n[a]) - 1)
            hit = ok & occ[q[..., 0], q[..., 1], q[..., 2]]
            has = hit.any(1)
            first = torch.argmax(hit.int(), dim=1)
            out[sl] = torch.where(has, ts[first], out[sl])
        return out

    def t_obs(self, o_rt, d_rt, tmax_rt, chunk=2048):
        """blocking distance (runtime units) per ray under the footprint test, inf if the bundle is not blocked before tmax_rt."""
        occ = self.reliable()
        o = torch.as_tensor(np.asarray(o_rt, np.float64), device=self.dev)
        d = torch.as_tensor(np.asarray(d_rt, np.float64), device=self.dev)
        tm = torch.as_tensor(np.asarray(tmax_rt, np.float64), device=self.dev)
        o = o[None].expand(d.shape[0], 3)
        out = self._march(occ, o, d, tm, chunk)
        rho = float(CFG["footprint"])
        if rho > 0:
            ref = torch.zeros_like(d)
            up = d[:, 1].abs() < 0.9
            ref[up, 1] = 1.0
            ref[~up, 0] = 1.0
            u = torch.linalg.cross(d, ref)
            u = u / u.norm(dim=1, keepdim=True)
            v = torch.linalg.cross(d, u)
            for off in (u, -u, v, -v):
                out = torch.maximum(out, self._march(occ, o + rho * off, d, tm, chunk))
        return out


def visible_mask(grid, fr, v, t_model, k_M):
    """bool like t_model: predicted hit is not behind a reliable observed surface (or there is no hit)."""
    import m_common as MC
    key = "_dv2_tobs_%d_%g" % (int(CFG["support"]), float(CFG["footprint"]))
    if v.get(key) is None:
        o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
        d_rt = MC.to_y_up(v["d"].double().cpu().numpy())
        d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
        v[key] = grid.t_obs(o_rt, d_rt, (v["tcap"].double() * k_M).cpu().numpy())
    tob = v[key].to(t_model.device)
    tp = t_model.double() * k_M
    return ~torch.isfinite(tp) | (tp <= tob + CFG["margin"])


def install(scene, device, cfg=None):
    """Install in a planner process AFTER d_policy.install(). Returns the grid. STATE['mode'] in {off, full, first}."""
    import m_p1
    import p1_full as PF
    import h_policy as HP
    import d_policy as DP
    if cfg:
        CFG.update({k: cfg[k] for k in CFG if k in cfg})
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

    def value_cov(t, v, geo, z0, cfg_):
        L = STATE["layer"]
        if STATE["mode"] == "off" or (STATE["mode"] == "first" and L != 1):
            return orig_vc(t, v, geo, z0, cfg_)
        t0 = time.time()
        vis = visible_mask(grid, geo.fr, v, t, float(geo.fr.k_M))
        fin = torch.isfinite(t)
        STATS["sec"][L] += time.time() - t0
        STATS["calls"][L] += 1
        STATS["rays"][L] += int(fin.sum())
        STATS["cut"][L] += int((fin & ~vis).sum())
        return orig_vc(torch.where(vis, t, torch.full_like(t, float("inf"))), v, geo, z0, cfg_)
    HP.value_cov = value_cov
    orig_la = DP.lookahead

    def lookahead(*a, **k):                    # every value_cov call made inside the lookahead is a layer >= 2 prediction
        STATE["layer"] = 2
        try:
            return orig_la(*a, **k)
        finally:
            STATE["layer"] = 1
    DP.lookahead = lookahead
    return grid


def install_step_log(path):
    """Outermost score_state wrapper (after d_policy.install + install): one JSON line per planning step with the D-v2 counters.
    Logging only -- scores are not touched."""
    import v2_plan_core as PC
    orig = PC.score_state
    n = {"i": 0}

    def score_state(*a, **k):
        for key in STATS:
            STATS[key] = [0, 0, 0] if key != "sec" else [0.0, 0.0, 0.0]
        t0 = time.time()
        r = orig(*a, **k)
        with open(path, "a") as f:
            f.write(json.dumps({"call": n["i"], "mode": STATE["mode"], "n": len(r[0].get("distp1", [])), "rays": STATS["rays"][1:], "cut": STATS["cut"][1:],
                                "calls": STATS["calls"][1:], "dv2_s": STATS["sec"][1:], "score_state_s": time.time() - t0}) + "\n")
        n["i"] += 1
        return r
    PC.score_state = score_state
