"""EXP6 V5.2 shared helpers (seen2scene env): model loading, the B2 geometry (D100 P1 for the model input, structure,
domain and FREE / UNKNOWN; S10 band_dil for novelty -- exactly p1_full._patch_geometry_b2 of the V4 / V5 closed
loops), score configuration, first-hit evaluation metrics against t* (evaluation side only)."""
import hashlib
import json
import os
import sys

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E52 = os.path.join(REPO, "experiments", "exp6_v5_2")
W52 = os.path.join(REPO, "work_dirs", "exp6_v5_2")
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from seen2scene.futureview import v52_model as V52  # noqa: E402
from seen2scene.futureview import v52_score as SC52  # noqa: E402

V5_PARENT = os.path.join(REPO, "work_dirs", "exp6_v5", "checkpoints", "v5_final.pt")
V5_PARENT_SHA = "5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["model_cfg"].get("arch") == "v52", "not a V5.2 checkpoint"
    m = V52.V52Model(ck["model_cfg"])
    m.load_state_dict(ck["ema"], strict=True)
    return m.to(dev).eval(), sha256_file(ckpt)


def score_cfg(name="score_v52.json"):
    p = os.path.join(E52, "configs", name)
    return json.load(open(p)) if os.path.exists(p) else dict(SC52.DEFAULT)


def b2_geometry(PC, MC, region, f100, f10, dev):
    """Geometry of the D100 fusion with caster.band_dil replaced by the S10 (sparse) map's dilated BAND, computed as
    p1_full._patch_geometry_b2 does (|v| < 0.999 trunc inside the domain, 3 x 3 x 3 max pool)."""
    geo = PC.Geometry.from_fusion(*region, f100, dev)
    c = geo.caster
    v = f10.input_volume().to(c.dev).to(c.dt)
    ob = (v.abs() < MC.TRUNC * 0.999) & c.domain
    c.band_dil = F.max_pool3d(ob[None, None].half(), 3, 1, 1)[0, 0] > 0
    del v, ob
    return geo


@torch.no_grad()
def firsthit_metrics(model, h, view, tstar, n_nohit_bins=10):
    """First-hit pmf of all rays of one candidate view vs the evaluation t* (model units; inf = no hit):
    NLL (nats / ray), top-interval accuracy (argmax over intervals + no-hit), no-hit calibration (10 bins of p_nohit),
    |mode - t*| and |E[t | hit] - t*| (voxels) on rays whose t* is a hit and whose predicted hit probability > 0.5.
    Returns sums / counts (aggregate across views)."""
    import seen2scene.futureview.geometry as G
    mem = model.memory(h)
    o, d, tcap = view["o"].float(), view["d"].float(), view["tcap"].float()
    ch = G.channels(view["pr"], view["tcap"], view["d"], view["o"]).float().t()
    R = d.shape[0]
    ts = torch.as_tensor(np.asarray(tstar, np.float64), device=d.device).float()
    unk = (view["pr"]["cls"] == G.CLS_UNKNOWN)
    acc = {}
    for s in range(0, R, 8192):
        sl = slice(s, min(R, s + 8192))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.belief(mem, o[None].expand(sl.stop - sl.start, 3), d[sl], tcap[sl], ch[sl])
        e = out["edges"]
        t = ts[sl]
        valid = ~(t < e[:, 0])                                              # a t* before the near clip is not scorable
        hit = torch.isfinite(t) & (t < e[:, -1])
        idx = torch.searchsorted(e.contiguous(), t[:, None].clamp(max=1e9).contiguous())[:, 0] - 1
        idx = idx.clamp(0, e.shape[1] - 2)
        lp = torch.where(hit, torch.gather(out["log_pf"], 1, idx[:, None])[:, 0], torch.log(out["p_nohit"].clamp(min=1e-30)))
        allp = torch.cat([out["p_first"], out["p_nohit"][:, None]], 1)
        top = allp.argmax(1)
        true = torch.where(hit, idx, torch.full_like(idx, e.shape[1] - 1))
        ph = 1.0 - out["p_nohit"]
        mode_t = torch.gather(out["t_mid"], 1, out["p_first"].argmax(1, keepdim=True))[:, 0]
        exp_t = (out["p_first"] * out["t_mid"]).sum(1) / ph.clamp(min=1e-6)
        dm = hit & (ph > 0.5) & valid
        for grp, m in (("all", valid), ("unknown", valid & unk[sl])):
            a = acc.setdefault(grp, {"n": 0, "nll": 0.0, "top": 0, "n_hit": 0, "n_dm": 0, "err_mode_vox": 0.0, "err_exp_vox": 0.0,
                                      "cal_n": [0] * n_nohit_bins, "cal_p": [0.0] * n_nohit_bins, "cal_y": [0.0] * n_nohit_bins})
            a["n"] += int(m.sum()); a["nll"] += float(-lp[m].sum()); a["top"] += int((top == true)[m].sum()); a["n_hit"] += int((hit & m).sum())
            mm = dm & m
            a["n_dm"] += int(mm.sum())
            a["err_mode_vox"] += float(((mode_t - t).abs() / V52.VS)[mm].sum()); a["err_exp_vox"] += float(((exp_t - t).abs() / V52.VS)[mm].sum())
            pn = out["p_nohit"][m]
            yn = (~hit[m]).float()
            bi = torch.clamp((pn * n_nohit_bins).long(), max=n_nohit_bins - 1)
            for k in range(n_nohit_bins):
                sel = bi == k
                a["cal_n"][k] += int(sel.sum()); a["cal_p"][k] += float(pn[sel].sum()); a["cal_y"][k] += float(yn[sel].sum())
    return acc


def merge_fh(a, b):
    for g, v in b.items():
        if g not in a:
            a[g] = json.loads(json.dumps(v))
            continue
        for k, x in v.items():
            a[g][k] = [p + q for p, q in zip(a[g][k], x)] if isinstance(x, list) else a[g][k] + x
    return a


def fh_summary(a):
    out = {}
    for g, v in a.items():
        n = max(v["n"], 1)
        cn = np.asarray(v["cal_n"], float)
        ece = float(np.sum(np.abs(np.asarray(v["cal_p"]) - np.asarray(v["cal_y"]))) / max(cn.sum(), 1))
        out[g] = {"rays": v["n"], "nll": v["nll"] / n, "top_acc": v["top"] / n, "hit_rate": v["n_hit"] / n,
                  "nohit_ece": ece, "mode_err_vox": v["err_mode_vox"] / max(v["n_dm"], 1), "exp_err_vox": v["err_exp_vox"] / max(v["n_dm"], 1),
                  "n_depth_err": v["n_dm"], "nohit_calibration": [{"pred": p / max(c, 1), "obs": y / max(c, 1), "n": int(c)} for p, y, c in zip(v["cal_p"], v["cal_y"], v["cal_n"])]}
    return out
