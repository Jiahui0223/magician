"""Experiment G3 -- combining the K = 4 generated samples into ONE map before planning (protocol G3-v1, reports/G3_PROTOCOL.md).
Thin wrapper over the frozen G4 sampling path; nothing frozen is edited. Two process-local hooks, installed by the G3 planner children /
the G3 fixed-state driver BEFORE the D1 policy is installed (d_policy captures PC.score_state at install time):
  noise hook   seen2scene.futureview.predictor.noise: when the planner asks for K = 1 map (the Table 3 K = 1 path) the hook also draws
               the 4 maps k = 0..3 with the SAME per-k seeds (independent generator per k -> bitwise the G4 maps) and stashes them.
  score hook   v2_plan_core.score_state: calls the frozen score_state with the stashed 4 maps (same candidates, same batch layout as G4),
               combines the 4 decoded samples into one map and returns it as the single (K = 1) prediction to D1:
     raw    b_bar = mean_k b_k, y_bar = mean_k y_k (network output space = UNKNOWN arc-length parameterisation), decoded with the unchanged
            rule geometry.decode (hit = b > 0 & y < 1 & n_u > 0)
     depth  each sample decoded (frozen); hit = (#samples with an UNKNOWN hit) >= 2 of 4 (a 2-2 tie counts as a hit); first hit =
            mean of the decoded ray distances t_k of the hitting samples (model units along the ray = metric depth / dz); rays
            without a combined hit keep the projection outcome t_known (identical in all samples)
     off    identity (eps and outputs passed through) -- regression only
  The H0 fields of the returned `out` (distp1, distp1_k) are recomputed for the combined map with the frozen scoring.new_set_from_depth
  (logging only; D1 replaces distp1 by its own score). The 4 underlying samples are kept in STATE["last"] for checks / figures.
"""
import torch

STATE = {"mode": "off", "eps4": None, "last": None, "calls": None, "installed": False}   # calls: list (fixed-state driver sets [] to record)


def combine_raw(b, y):
    """b, y [K, R] -> b_bar, y_bar [1, R]."""
    return b.mean(0, keepdim=True), y.mean(0, keepdim=True)


def combine_depth(t, hit_u, t_known, min_hits=2):
    """t [K, R] decoded ray distances (inf = no hit), hit_u [K, R] UNKNOWN hits, t_known [R] projection outcome ->
    t_c [1, R], hit_c [1, R]: hit if >= min_hits samples hit (ties count as hits); t = mean t_k over the hitting samples."""
    n = hit_u.sum(0)
    hit = n >= min_hits
    tt = torch.where(hit_u, t, torch.zeros_like(t)).sum(0) / n.clamp(min=1).to(t.dtype)
    t_c = torch.where(hit, tt, t_known.to(t.dtype).expand_as(tt))
    return t_c[None], hit[None]


def install(PC, PR, G, SCO):
    """PC = v2_plan_core, PR = futureview.predictor, G = futureview.geometry, SCO = futureview.scoring (module objects)."""
    assert not STATE["installed"]
    noise0, frozen = PR.noise, PC.score_state

    def noise(device, K, seed_fn):
        out = noise0(device, K, seed_fn)
        if STATE["mode"] != "off" and K == 1:
            e4 = noise0(device, 4, seed_fn)
            assert torch.equal(e4[:1], out), "noise map 0 differs"
            STATE["eps4"] = e4
        return out

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw):
        mode = STATE["mode"]
        if mode == "off":
            return frozen(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw)
        assert eps.shape[0] == 1 and STATE["eps4"] is not None and torch.equal(STATE["eps4"][:1], eps), "G3 combine expects the K = 1 path"
        want_hits = kw.pop("keep_hits", False)
        out, tm, views = frozen(core, geo, h, legal_R, legal_T, STATE["eps4"], z0, *a, keep_hits=True, **kw)
        if not views:
            if not want_hits:
                out.pop("hits", None)
            return out, tm, views
        k_M = geo.fr.k_M
        hits, dp, dpk, last = [], [], [], []
        for m, v in enumerate(views):
            hh = out["hits"][m]
            if mode == "raw":
                bc, yc = combine_raw(hh["b"], hh["y"])
                tc, hc = G.decode(bc, yc, v["o"], v["d"], v["pr"], geo.struct, geo.domain)
            elif mode == "depth":
                tc, hc = combine_depth(hh["t"], hh["hit_u"], v["pr"]["t_known"])
                bc = torch.where(hc, torch.ones_like(tc), -torch.ones_like(tc)).to(hh["b"].dtype)
                yc = torch.full_like(bc, float("nan"))
            else:
                raise ValueError(mode)
            ids, wv = SCO.new_set_from_depth(tc[0], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
            dp.append(float(wv.sum()))
            dpk.append([float(wv.sum())])
            hits.append({"t": tc, "hit_u": hc, "b": bc, "y": yc})
            last.append({"t4": hh["t"], "hit_u4": hh["hit_u"], "b4": hh["b"], "y4": hh["y"]})
        out["g3_h0_4sample_distp1"] = out["distp1"]
        out["distp1"], out["distp1_k"] = dp, dpk
        out["hits"] = hits
        out["g3_mode"] = mode
        STATE["last"] = last
        if STATE["calls"] is not None:
            STATE["calls"].append({"views": views, "samples": last, "hits": hits})
        if not want_hits:
            out.pop("hits", None)
        return out, tm, views

    PR.noise = noise
    PC.score_state = score_state
    STATE["installed"] = True
    return {"noise_hook": True, "score_hook": True}
