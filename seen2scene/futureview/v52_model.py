"""V5.2 (EXP6): Camera-Conditioned Ray Geometry Belief for direct NBV scoring -- no 2D flow, no future depth images.

Inherited from V5 (seen2scene/futureview/v5_model.py, loaded strictly from the V5 EMA@12000 checkpoint):
  encoder   3D U-Net on the D100 coarse P1 grid -> F0 / F1 scene memory
  dec       V5RayDecoder: camera ray query + geometry-guided local / coarse retrieval + cross-attention -> z_r, and its
            SurfaceMLP b_r(t) = sigmoid(surface_logits(z_r, t)) (multi-surface marginal: "a surface lies within the
            finite neighbourhood Delta of t"; not a first-hit pmf, 1 - b is not free space). The V5 BeliefSummary /
            CondAdapter (flow conditioning) are carried but unused and frozen; the V5 flow U-Net is NOT part of V5.2.
New:
  hazard    HazardHead: conditional first-hit hazard rho_ri >= 0 on ordered depth intervals I_i = [t_i^-, t_i^+) of
            every ray, from z_r, the DETACHED belief b_r(t_i) (+ its running max / mass before t_i), depth encodings and
            the observation state (EMPTY / BAND / in-domain fractions of the D100 coarse grid) at the interval centre.
            h_ri = 1 - exp(-rho_ri dt_i / S0); T_ri = prod_{j<i}(1 - h_rj); p_first_ri = T_ri h_ri;
            p_nohit_r = prod_j(1 - h_rj); computed in log space (sum_i p_first + p_nohit = 1 by telescoping).
            The belief enters detached, so the first-hit loss can not lower b behind the first surface.
Intervals: N per ray, uniform in u = log1p(t / c) between t_near = tcap / 140 (the 0.5 near clip of the 70 sensor range)
and t_far = tcap (sensor range); c = 2 F0 cells (16 VS): ~1 VS bins near the camera, ~relative bins far away.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seen2scene.futureview import model as M
from seen2scene.futureview import v5_model as V5

VS, S0 = M.VS, V5.S0
NEAR_FRAC = 0.5 / 70.0          # near clip / sensor range (both runtime z-depths) -> t_near = tcap * NEAR_FRAC
A_MAX = 30.0                    # cap of rho * dt per interval (h <= 1 - 1e-13)


def intervals(tcap, n, c):
    """tcap [R] -> edges [R, n + 1] (model units), uniform in log1p(t / c) from tcap * NEAR_FRAC to tcap."""
    lo = torch.log1p(tcap * NEAR_FRAC / c)
    hi = torch.log1p(tcap / c)
    u = lo[:, None] + (hi - lo)[:, None] * torch.linspace(0, 1, n + 1, device=tcap.device, dtype=tcap.dtype)[None]
    return c * torch.expm1(u)


def firsthit(a):
    """a = rho * dt / S0 [R, N] >= 0 -> dict: log_surv [R, N] (log T_i, survival before interval i), log_pf [R, N]
    (log p_first), p_first, p_nohit [R], all float32 (log-space accumulation, a capped at A_MAX)."""
    a = a.float().clamp(0.0, A_MAX)
    csum = torch.cumsum(a, -1)
    log_surv = torch.cat([torch.zeros_like(a[:, :1]), -csum[:, :-1]], -1)
    log_h = torch.log((-torch.expm1(-a)).clamp(min=1e-30))
    log_pf = log_surv + log_h
    return {"a": a, "log_surv": log_surv, "log_pf": log_pf, "p_first": torch.exp(log_pf), "p_nohit": torch.exp(-csum[:, -1])}


class HazardHead(nn.Module):
    def __init__(self, d=128, hidden=64, n_freq=6, rho0=0.02):
        super().__init__()
        self.n_freq = n_freq
        self.f_dim = 2 * n_freq + 2 + 2 + 2 + 4
        self.w_z = nn.Linear(d, hidden)
        self.w_f = nn.Linear(self.f_dim, hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.fill_(math.log(math.expm1(rho0)))          # softplus(bias) = rho0 per S0 at init

    def features(self, t_mid, dt, tcap, logit_b, obs):
        """t_mid, dt, logit_b [R, N]; obs [R, N, 4] -> [R, N, f_dim]. b enters detached."""
        b = torch.sigmoid(logit_b.detach().float())
        prev_max = torch.cat([torch.zeros_like(b[:, :1]), torch.cummax(b, -1).values[:, :-1]], -1)
        mass = torch.cumsum(b * dt / S0, -1)
        prev_mass = torch.cat([torch.zeros_like(b[:, :1]), mass[:, :-1]], -1)
        lt = torch.log1p(t_mid / S0)[..., None] / 5.0
        return torch.cat([V5.fourier(lt, self.n_freq), (t_mid / tcap[:, None])[..., None], torch.log(dt / S0).clamp(-10, 10)[..., None] / 5.0,
                          b[..., None], (logit_b.detach().float().clamp(-10, 10) / 10.0)[..., None], prev_max[..., None],
                          torch.log1p(prev_mass)[..., None] / 3.0, obs], -1)

    def forward(self, z, feats):
        """z [R, d], feats [R, N, f_dim] -> rho [R, N] (per S0 of ray length)."""
        hdn = self.w_z(z)[:, None] + self.w_f(feats.to(self.w_f.weight.dtype))
        return F.softplus(self.mlp(hdn)[..., 0].float())


def obs_at(mem, p):
    """Observation state of the D100 coarse cells at points p [R, N, 3] -> [R, N, 4] (EMPTY, BAND, in-domain fraction,
    inside-grid flag)."""
    ijk = torch.floor(p / S0).long()
    inside = ((ijk >= 0) & (ijk < mem.dims0)).all(-1)
    ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), mem.dims0 - 1)
    flat = (ijk[..., 0] * mem.dims0[1] + ijk[..., 1]) * mem.dims0[2] + ijk[..., 2]
    o3 = mem.obs0[0].flatten(1).float()[:, flat]                             # [3, R, N]
    o3 = o3.permute(1, 2, 0) * inside[..., None]
    return torch.cat([o3, inside[..., None].float()], -1)


class V52Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, r, v5, hz = cfg["encoder"], cfg["ray"], cfg["v5"], cfg["hazard"]
        self.cfg = cfg
        self.encoder = M.Encoder3D(e["in_ch"], tuple(e["ch"]), e["blocks_per_level"], e["dilation_L2"], e["groups"])
        c0, c1 = self.encoder.out_ch
        self.proj_ch = cfg["proj_ch"]
        self.dec = V5.V5RayDecoder(c0, c1, self.proj_ch, v5["d"], v5["heads"], v5["n0"], v5["n1"], v5["n_freq"], v5["d_sum"], r["out_ch"], v5["n_probe"],
                                   out_scale_init=v5.get("out_scale_init", 1.0))
        self.hazard = HazardHead(v5["d"], hz["hidden"], hz["n_freq"], hz["rho0"])
        self.n_bins, self.c_bins = hz["n_bins"], hz["c_bins_vox"] * VS
        self.chunk = hz.get("chunk", 8192)
        # the V5 flow-conditioning parts of the decoder are not used by V5.2
        for nme in ("sum_mlp", "adapter"):
            for p in getattr(self.dec, nme).parameters():
                p.requires_grad_(False)
        self.dec.out_scale.requires_grad_(False)

    def encode(self, coarse_in):
        F0, F1 = self.encoder(coarse_in)
        return F0, F1, coarse_in[:, 2:3], coarse_in

    def memory(self, h):
        return V5.Memory(h[0], h[1], h[3])

    def belief(self, mem, o, d, tcap, ch, edges=None):
        """Rays of ONE state: o [R, 3], d [R, 3], tcap [R], ch [R, 11] -> dict with z [R, d], edges [R, N + 1], t_mid, dt,
        logit_b (surface logits at the interval centres), rho, and the first-hit quantities of firsthit()."""
        _, aux = self.dec(mem, o, d, tcap, ch, return_aux=True)
        z = aux["z"]
        if edges is None:
            edges = intervals(tcap.float(), self.n_bins, self.c_bins)
        t_mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
        dt = (edges[:, 1:] - edges[:, :-1]).clamp(min=1e-6)
        logit_b = self.dec.surface_logits(z, t_mid, tcap)
        p = o[:, None] + d[:, None] * t_mid[..., None]
        feats = self.hazard.features(t_mid, dt, tcap.float(), logit_b, obs_at(mem, p))
        rho = self.hazard(z, feats)
        out = firsthit(rho * dt / S0)
        out.update({"z": z, "edges": edges, "t_mid": t_mid, "dt": dt, "logit_b": logit_b.float(), "rho": rho})
        return out


def load_v5_warmstart(model, v5_state):
    """Strict copy of the V5 encoder / decoder weights; the V5 flow U-Net is not loaded (not part of V5.2)."""
    sd = model.state_dict()
    copied = []
    for k, v in v5_state.items():
        if k.startswith("encoder.") or k.startswith("dec."):
            assert k in sd and sd[k].shape == v.shape, ("V5 -> V5.2 mismatch", k)
            sd[k] = v.clone()
            copied.append(k)
    missing = [k for k in sd if (k.startswith("encoder.") or k.startswith("dec.")) and k not in v5_state]
    assert not missing, missing[:5]
    model.load_state_dict(sd)
    return {"copied": len(copied), "copied_numel": int(sum(v5_state[k].numel() for k in copied)),
            "not_loaded_from_v5": sorted({k.split(".")[0] for k in v5_state if not (k.startswith("encoder.") or k.startswith("dec."))}),
            "hazard_numel": int(sum(p.numel() for p in model.hazard.parameters()))}
