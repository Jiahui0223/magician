"""EXP5 Task 1 -- planner-side V2 scoring of one observed state (no GT; imported by the closed-loop planner child and by
the evaluation scripts). protocol §4.1-§4.3, §5.2.

score_state(core, sg, h, legal_R, legal_T, eps, ...) -> per legal action: DistP1 (K mean), DistP1 of sample 0,
U_raw / DistU (+ components) for the requested deltas, U_flow, the decoded hits (optional), timings.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import predictor as PR  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402


class Core:
    def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
        sha = VF.sha256_file(ckpt_path)
        if expect_sha is not None and sha != expect_sha:
            raise VF.InvalidRun(f"INVALID: checkpoint sha {sha} != frozen {expect_sha}")
        ck = torch.load(ckpt_path, map_location=device)
        self.model = M.V2Model(ck["model_cfg"]).to(device)
        self.model.load_state_dict(ck["ema"])
        self.model.eval()
        self.pred = PR.V2Predictor(self.model, device, steps)
        self.dev, self.ckpt_sha, self.ckpt = device, sha, ckpt_path


class Geometry:
    """Frozen SceneCaster of one state + observation-only cast volume + structure (from arrays)."""

    def __init__(self, fr, dims, dom, vol, struct, device):
        import m_common as MC
        import m_score as MS
        self.fr, self.dims = fr, [int(x) for x in dims]
        self.caster = MS.SceneCaster(fr, dims, dom, vol, device=device)
        self.vol_obs = self.caster.cast_volume(torch.full(self.caster.obs_surf.shape, MC.TRUNC, device=self.caster.dev))
        self.struct, self.domain, self.ext = struct, self.caster.domain, self.caster.ext

    @classmethod
    def from_fusion(cls, fr, dims, dom, fu, device):
        return cls(fr, dims, dom, fu.input_volume(), fu.structure(), device)

    def coarse(self):
        return G.coarse_grid(self.struct, self.domain)

    def view(self, R, T, jitter=(0.0, 0.0)):
        o, d, tcap, dz = VF.view_rays(self.caster, R, T, jitter)
        out = self.caster.cast(self.vol_obs, o, d, tcap)
        tk = torch.where(out["hit"], out["t"], torch.full_like(out["t"], float("inf")))
        pr = G.project(o, d, tcap, self.ext, tk, self.struct, self.domain)
        return {"o": o, "d": d, "tcap": tcap, "dz": dz, "pr": pr}


def sync():
    torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def score_state(core, geo, h, legal_R, legal_T, eps, z0, deltas=(), want_flow=False, want_feat=False,
                keep_hits=False, jitter=(0.0, 0.0)):
    """eps [K, 2, 64, 128] (shared by all candidates). Returns dict of per-action lists and the timing parts."""
    tm = {}
    t0 = sync()
    views = [geo.view(R, T, jitter) for R, T in zip(legal_R, legal_T)]
    t1 = sync()
    tm["projection_s"] = t1 - t0
    K = eps.shape[0]
    out = {"distp1": [], "distp1_k": [], "n_unknown_rays": []}
    for dl in deltas:
        for k in ("u_raw", "u_raw_H", "u_raw_D", "dist_u", "dist_u_H", "dist_u_D"):
            out[f"{k}@{dl}"] = []
    if want_flow:
        out["u_flow"] = []
    if want_feat:
        out["u_feat"] = []
    if keep_hits:
        out["hits"] = []
    if not views:
        tm.update({"cond_s": 0.0, "sampling_s": 0.0, "decode_score_s": 0.0})
        return out, tm, views
    cond, m_unk = core.pred.cond(h, [(v["o"], v["d"], v["tcap"], v["pr"]) for v in views])
    t2 = sync()
    tm["cond_s"] = t2 - t1
    r = core.pred.sample(cond, m_unk, eps, return_features=want_feat)
    b, y = r[0], r[1]
    t3 = sync()
    tm["sampling_s"] = t3 - t2
    k_M = geo.fr.k_M
    for m, v in enumerate(views):
        t_dec, hit_u = G.decode(b[m], y[m], v["o"], v["d"], v["pr"], geo.struct, geo.domain)
        ws, news = [], []
        for k in range(K):
            ids, wv = SCO.new_set_from_depth(t_dec[k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain,
                                            geo.caster.band_dil, z0)
            ws.append(float(wv.sum()))
            if K >= 2 and deltas:
                news.append(SCO.new_mask_from_depth(t_dec[k], v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)[0])
        out["distp1"].append(float(np.mean(ws)))
        out["distp1_k"].append(ws)
        mu = v["pr"]["cls"] == G.CLS_UNKNOWN
        out["n_unknown_rays"].append(int(mu.sum()))
        if K >= 2 and deltas:
            nw = torch.stack(news)
            for dl in deltas:
                for kk, vv in SCO.u_scores(nw, t_dec, v["dz"], k_M, z0, dl).items():
                    out[f"{kk}@{dl}"].append(vv)
        if want_flow:
            out["u_flow"].append(float((b[m][:, mu].var(0, unbiased=False) + y[m][:, mu].var(0, unbiased=False)).mean())
                                 if mu.any() else 0.0)
        if want_feat:
            fm = M.unpad_rays(r[2][m])                      # [K, C, R]
            out["u_feat"].append(float(fm[:, :, mu].var(0, unbiased=False).mean()) if mu.any() else 0.0)
        if keep_hits:
            out["hits"].append({"t": t_dec, "hit_u": hit_u, "b": b[m], "y": y[m]})
    tm["decode_score_s"] = sync() - t3
    return out, tm, views
