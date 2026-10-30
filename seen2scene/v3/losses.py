"""L = L_FM + lambda_rec * L_recon + beta(step) * KL (EXP6 V3, protocol.md).

L_FM     rectified-flow velocity MSE on M_unknown rays with a valid label, summed over the batch and divided by
         2 x the number of such rays (the V2 / E2 definition, v2_data.fm_sum); with DDP the count is global.
L_recon  per state: class-balanced cross-entropy over the valid cells of the 4x grid (mean CE of every class present,
         averaged over those classes; EMPTY / UNKNOWN / BAND) + L1 of sdf_hat vs sdf on BAND cells; averaged over
         the states of the batch. Targets come from the SAME input grid (observed state), never from GT.
KL       seen2scene.v3.vae.kl_per_unit (per valid latent unit and channel), averaged over the states of the batch.
beta     linear warm-up from 0 to beta_max over kl_warmup steps after kl_start steps, then constant.
"""
import torch
import torch.nn.functional as F

from seen2scene.v3 import partial_tsdf as PT
from seen2scene.v3 import vae as VAE


def fm_terms(model, bt, eps, tau, autocast=True):
    """(sum of squared velocity errors over labelled M_unknown rays, number of such rays)."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"], bt.get("mem"))
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum(), lm.sum()


def recon_terms(model, states, generator=None, autocast=True):
    """states: [{mu, logvar, valid8, x4, valid4}] -> (L_recon mean over states, diagnostics dict)."""
    ce_tot, l1_tot, n = 0.0, 0.0, 0
    diag = {"acc_empty": [], "acc_unknown": [], "acc_band": [], "sdf_mae": []}
    for st in states:
        z = VAE.reparameterize(st["mu"], st["logvar"], generator)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model.decode(z)
        sdf_hat, logits = out[:, 0:1].float(), out[:, 1:4].float()
        cls, sdf_t, band_m, vmask = PT.recon_targets(st["x4"], st["valid4"])
        ce = F.cross_entropy(logits, cls, reduction="none")
        terms = []
        for c, nm in ((PT.EMPTY, "acc_empty"), (PT.UNKNOWN, "acc_unknown"), (PT.BAND, "acc_band")):
            mc = vmask & (cls == c)
            if mc.any():
                terms.append(ce[mc].mean())
                diag[nm].append(float((logits.argmax(1)[mc] == c).float().mean()))
        ce_c = torch.stack(terms).mean() if terms else logits.sum() * 0.0
        nb = band_m.sum().clamp(min=1.0)
        l1 = ((sdf_hat - sdf_t).abs() * band_m).sum() / nb
        diag["sdf_mae"].append(float(l1))
        ce_tot, l1_tot, n = ce_tot + ce_c, l1_tot + l1, n + 1
    n = max(n, 1)
    return (ce_tot + l1_tot) / n, {"ce": float(ce_tot) / n, "l1": float(l1_tot) / n,
                                    **{k: (sum(v) / len(v) if v else None) for k, v in diag.items()}}


def kl_terms(states):
    """-> (mean KL per unit over the states, per-channel mean [C])."""
    kls, chs = [], []
    for st in states:
        k, per_ch = VAE.kl_per_unit(st["mu"], st["logvar"], st["valid8"])
        kls.append(k)
        chs.append(per_ch)
    return torch.stack(kls).mean(), torch.stack(chs).mean(0)


def beta_at(step, cfg):
    k = cfg["kl"]
    if step < k["start"]:
        return 0.0
    return k["beta"] * min(1.0, (step - k["start"] + 1) / max(1, k["warmup"]))
