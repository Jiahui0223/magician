"""Development-set metrics of a V3 model (EXP6): the V2 definitions (task1 t12_train.eval_fm / eval_samples) with the
V3 batch (scene memory for T1) + the VAE reconstruction metrics. Deterministic noise per state chunk.
"""
import math

import torch

from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.v3 import data as D
from seen2scene.v3 import losses as L
from seen2scene.v3 import model as V3
from seen2scene.v3 import partial_tsdf as PT


def gen(dev, key):
    g = torch.Generator(device=dev)
    g.manual_seed(D.seed_of(key))
    return g


def chunks(states, max_views=64):
    out, cur, n = [], [], 0
    for s in states:
        k = len(s[3])
        if cur and n + k > max_views:
            out.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += k
    if cur:
        out.append(cur)
    return out


def per_view_loss(model, bt, eps, tau, autocast=True):
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"], bt.get("mem"))
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum((1, 2)), 2.0 * lm.sum((1, 2))


@torch.no_grad()
def eval_fm(model, data, states, name, dev, taus):
    model.eval()
    num = torch.zeros((), dtype=torch.float64, device=dev)
    den = torch.zeros((), dtype=torch.float64, device=dev)
    for ch in chunks(states):
        bt = D.build_cond(model, data, ch, dev)
        V = bt["cond"].shape[0]
        g = gen(dev, f"exp6-V3|evalfm|{name}|{ch[0][0]}|{ch[0][1]}")
        eps = torch.randn(V, 2, M.GH, M.GW_PAD, generator=g, device=dev)
        for tau in taus:
            e, c = per_view_loss(model, bt, eps, torch.full((V,), tau, device=dev))
            num += e.sum().double()
            den += c.sum().double()
    model.train()
    return float(num / den.clamp(min=1))


@torch.no_grad()
def eval_samples(model, data, states, name, dev, K=4, steps=10):
    """Sample metrics on M_unknown rays with a valid label, in unknown-arclength voxels (the V2 definitions)."""
    model.eval()
    acc = {"brier": [0.0, 0.0], "fw": [0.0, 0.0], "ph": [0.0, 0.0], "lh": [0.0, 0.0], "es": [0.0, 0.0],
           "spread": [0.0, 0.0], "ambig": [0.0, 0.0], "s_sq": [0.0, 0.0], "s_sum": [0.0, 0.0]}
    for ch in chunks(states, 32):
        bt = D.build_cond(model, data, ch, dev)
        V = bt["cond"].shape[0]
        g = gen(dev, f"exp6-V3|evalsample|{name}|{ch[0][0]}|{ch[0][1]}")
        eps = torch.randn(K, V, 2, M.GH, M.GW_PAD, generator=g, device=dev).reshape(K * V, 2, M.GH, M.GW_PAD)
        cond = bt["cond"].repeat(K, 1, 1, 1)
        mu = bt["m_unk"].repeat(K, 1, 1)
        mem = None if bt["mem"] is None else tuple(m.repeat(K, *([1] * (m.dim() - 1))) for m in bt["mem"])
        x = V3.sample(model, cond, mu, eps, mem, steps).reshape(K, V, 2, M.GH, M.GW_PAD)
        x = M.unpad_rays(x)
        b, y = x[:, :, 0], x[:, :, 1]
        lm = M.unpad_rays(bt["lmask"])
        cat = bt["cat"].reshape(V, -1)
        lab = (cat == G.CAT["HIT"]) & lm
        hit = (b > 0) & (y < 1.0)
        p = hit.float().mean(0)
        s = ((y + 1) / 2).clamp(0, 1)
        lu = bt["lu_vox"].reshape(V, -1)
        ell = torch.where(hit, s, torch.ones_like(s)) * lu
        s_lab = ((M.unpad_rays(bt["x1"])[:, 1] + 1) / 2).clamp(0, 1)
        ell_t = torch.where(lab, s_lab, torch.ones_like(s_lab)) * lu
        nohit = (cat == G.CAT["NOHIT"]) & lm
        for v in range(V):
            m = lm[v]
            if not m.any():
                continue
            rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))      # noqa: E731
            e1 = rms(ell[:, v] - ell_t[v][None]).mean()
            pair = sum(rms(ell[i, v] - ell[j, v]) for i in range(K) for j in range(K) if i != j)
            acc["es"][0] += float(e1 - pair / (2 * K * (K - 1)))
            acc["es"][1] += 1
        acc["brier"][0] += float(((p - lab.float()) ** 2)[lm].sum()); acc["brier"][1] += float(lm.sum())
        acc["fw"][0] += float(p[nohit].sum()); acc["fw"][1] += float(nohit.sum())
        acc["ph"][0] += float(p[lm].sum()); acc["ph"][1] += float(lm.sum())
        acc["lh"][0] += float(lab[lm].float().sum()); acc["lh"][1] += float(lm.sum())
        pd = torch.stack([(ell[i] - ell[j]).abs() for i in range(K) for j in range(i + 1, K)]).mean(0)
        acc["spread"][0] += float(pd[lm].sum()); acc["spread"][1] += float(lm.sum())
        acc["ambig"][0] += float((b.abs() < 0.5)[:, lm].float().sum()); acc["ambig"][1] += float(lm.sum()) * K
        hs = s[hit & lm[None]]
        acc["s_sum"][0] += float(hs.sum()); acc["s_sum"][1] += float(hs.numel())
        acc["s_sq"][0] += float((hs ** 2).sum()); acc["s_sq"][1] += float(hs.numel())
    model.train()
    r = {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items() if k not in ("s_sq", "s_sum")}
    n = acc["s_sum"][1]
    r["s_std_hits"] = math.sqrt(max(0.0, acc["s_sq"][0] / n - (acc["s_sum"][0] / n) ** 2)) if n else None
    r["hit_ratio"] = r["ph"] / r["lh"] if r.get("lh") else None
    return {"brier": r["brier"], "false_wall": r["fw"], "pred_hit_rate": r["ph"], "label_hit_rate": r["lh"],
            "hit_ratio": r["hit_ratio"], "fair_es_arclength_vox": r["es"], "spread_vox": r["spread"],
            "ambiguous_bit_rate": r["ambig"], "s_std_hits": r["s_std_hits"]}


@torch.no_grad()
def eval_recon(model, data, states, dev, max_states=64):
    """VAE reconstruction of the observed state with z = mu (deterministic): class accuracy per class, sdf MAE on
    BAND cells, KL per unit, active units -- averaged over the first max_states states of the fixed subset."""
    model.eval()
    seen, acc = set(), {"acc_empty": [], "acc_unknown": [], "acc_band": [], "sdf_mae": [], "kl": [], "active_units": [], "mu_std_mean": []}
    for ti, t, dlt, rows in states:
        if (ti, t) in seen or len(seen) >= max_states:
            continue
        seen.add((ti, t))
        x4, valid4 = data.caches[ti].state(t, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            h = model.encode(x4, valid4)
            out = model.decode(h["mu"])
        sdf_hat, logits = out[:, 0:1].float(), out[:, 1:4].float()
        cls, sdf_t, band_m, vmask = PT.recon_targets(x4, valid4)
        pred = logits.argmax(1)
        for c, nm in ((PT.EMPTY, "acc_empty"), (PT.UNKNOWN, "acc_unknown"), (PT.BAND, "acc_band")):
            mc = vmask & (cls == c)
            if mc.any():
                acc[nm].append(float((pred[mc] == c).float().mean()))
        nb = band_m.sum()
        if nb > 0:
            acc["sdf_mae"].append(float(((sdf_hat - sdf_t).abs() * band_m).sum() / nb))
        from seen2scene.v3 import vae as VAE
        kl, _ = VAE.kl_per_unit(h["mu"], h["logvar"], h["valid8"])
        ps = VAE.posterior_stats(h["mu"], h["logvar"], h["valid8"])
        acc["kl"].append(float(kl)); acc["active_units"].append(ps["active_units"]); acc["mu_std_mean"].append(ps["mu_std_mean"])
    model.train()
    return {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}
