"""V5 (EXP6): Geometry-Aware Ray-to-Scene Memory with Multi-Surface Belief and the original 2D Rectified Flow.

Inherited from V4 (same submodule names -> strict loading from the V4 checkpoint):
  encoder   futureview.model.Encoder3D: coarse P1 grid [1, 4, X', Y', 32] -> F0 [1, 32, X', Y', 32] (cell 8 VS),
            F1 [1, 64, X'/2, Y'/2, 16] (cell 16 VS)
  unet      futureview.model.UNet2D: the rectified-flow U-Net, input = x (2) + cond (76) channels, unchanged
New (the only V5 change; the V4 RayEncoder / RayXAttn / 32-point grid_sample are NOT part of this model):
  dec       V5RayDecoder: per ray a query q_r (direction, origin, ray range, the 11 analytic projection channels);
            retrieval of Local tokens (F0 <= n0, F1 <= n1: the grid cells nearest to the ray inside log-depth strata,
            integer cell indexing + gather, no interpolation), Coarse tokens (F1 pooled to 2 x 2 x 2 regions) and one
            learned null token; geometry-biased multi-head cross-attention -> z_r; SurfaceMLP b_r(t) queryable at any
            depth; BeliefSummary over 16 log-depth probes of b_r; CondAdapter([z_r, summary]) -> the 64 ray channels of
            the V4 flow condition. cond = [11 projection channels, 64 ray channels, valid column] = the V4 layout.
Tensor conventions follow futureview.model (model frame of the coarse grid, z-up, cell (i, j, k) of F0 covers
[8 i VS, 8 i + 8 VS)); rays o [N, 3], d [N, 3] (unit), tcap [N] model units.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from seen2scene.futureview import model as M

VS, BLOCK = M.VS, M.BLOCK
GH, GW, GW_PAD = M.GH, M.GW, M.GW_PAD
S0 = BLOCK * VS                       # F0 cell size (model units)


def fourier(x, n_freq):
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class Memory:
    """Scene memory of one state: F0 / F1 feature grids, observation state of every cell (EMPTY / BAND / in-domain
    fractions of the P1 coarse grid, pooled for F1) and the coarse context tokens."""

    def __init__(self, F0, F1, cin, n_coarse_side=2):
        self.F0, self.F1 = F0, F1                                   # [1, C, X, Y, Z]
        self.obs0 = cin[:, :3]                                      # [1, 3, X, Y, Z]: EMPTY, BAND, dom fractions
        self.obs1 = F.avg_pool3d(cin[:, :3].float(), 2, ceil_mode=True)[..., :F1.shape[2], :F1.shape[3], :F1.shape[4]]
        self.dims0 = torch.tensor(F0.shape[2:], device=F0.device)
        self.dims1 = torch.tensor(F1.shape[2:], device=F0.device)
        self.ext = self.dims0.float() * S0
        c = F.adaptive_avg_pool3d(F1.float(), n_coarse_side)       # [1, C1, 2, 2, 2]
        o = F.adaptive_avg_pool3d(self.obs1, n_coarse_side)
        g = torch.stack(torch.meshgrid(*[torch.arange(n_coarse_side, device=F0.device)] * 3, indexing="ij"), -1).reshape(-1, 3).float()
        self.coarse_f = c[0].flatten(1).t()                          # [8, C1]
        self.coarse_obs = o[0].flatten(1).t()                        # [8, 3]
        self.coarse_pos = (g + 0.5) / n_coarse_side * self.ext[None]  # [8, 3] region centres
        self.coarse_size = (self.ext / n_coarse_side).max()


def retrieve(mem, level, o, d, tcap, n_tok, n_cand=3):
    """Local tokens of one level along the rays: log-depth strata between one cell and tcap; inside each stratum the
    grid cell whose centre is nearest to the ray among n_cand probe cells (integer indexing); duplicates of the
    previous stratum, cells outside the grid / scene domain and cells behind the camera are masked.
    -> idx [N, n_tok] (flat cell index), valid [N, n_tok], tau, delta [N, n_tok] (model units)."""
    s = S0 * (1 if level == 0 else 2)
    dims = mem.dims0 if level == 0 else mem.dims1
    obs = mem.obs0 if level == 0 else mem.obs1
    N = o.shape[0]
    tmin = torch.full_like(tcap, s)
    lo = torch.log(tmin)
    hi = torch.log(torch.maximum(tcap, tmin * 1.0001))
    j = (torch.arange(n_tok, device=o.device).float()[:, None] + (torch.arange(n_cand, device=o.device).float()[None] + 0.5) / n_cand)  # [n_tok, n_cand]
    tq = torch.exp(lo[:, None, None] + (hi - lo)[:, None, None] * j[None] / n_tok)                   # [N, n_tok, n_cand]
    p = o[:, None, None] + d[:, None, None] * tq[..., None]
    ijk = torch.floor(p / s).long()
    inside = ((ijk >= 0) & (ijk < dims)).all(-1)
    ijk = torch.minimum(torch.maximum(ijk, torch.zeros_like(ijk)), dims - 1)
    cen = (ijk.float() + 0.5) * s
    rel = cen - o[:, None, None]
    tau = (rel * d[:, None, None]).sum(-1)
    delta = (rel - tau[..., None] * d[:, None, None]).norm(dim=-1)
    flat = (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2]
    dom = obs[0, 2].reshape(-1)[flat] > 0
    ok = inside & dom & (tau > 0)
    dd = torch.where(ok, delta, torch.full_like(delta, float("inf")))
    k = dd.argmin(-1)                                                                                 # [N, n_tok]
    g = lambda x: torch.gather(x, 2, k[..., None])[..., 0]  # noqa: E731
    idx, tau, delta, ok = g(flat), g(tau), g(delta), g(ok.float()) > 0
    dup = torch.zeros_like(ok)
    dup[:, 1:] = idx[:, 1:] == idx[:, :-1]
    return idx, ok & ~dup, tau, delta


class V5RayDecoder(nn.Module):
    def __init__(self, c0=32, c1=64, proj_ch=11, d=128, heads=4, n0=24, n1=12, n_freq=6, d_sum=32, out_ch=64, n_probe=16, ffn=2, out_scale_init=1.0):
        super().__init__()
        self.d, self.h, self.n0, self.n1, self.n_freq, self.n_probe = d, heads, n0, n1, n_freq, n_probe
        q_in = 2 * 3 * n_freq + 2 * 3 * n_freq + 2 + proj_ch
        self.q_mlp = nn.Sequential(nn.Linear(q_in, d), nn.SiLU(), nn.Linear(d, d))
        self.geo_dim = 2 * n_freq + 6 + 3 + 3                       # PE(log tau), tau/tcap, delta/s, log s, cos, level, obs(3), rel dir(3)
        self.f_proj = nn.ModuleList([nn.Linear(c0, d), nn.Linear(c1, d), nn.Linear(c1, d)])     # F0, F1, coarse
        self.geo_mlp = nn.Sequential(nn.Linear(self.geo_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.geo_bias = nn.Linear(self.geo_dim, heads)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.ln_kv = nn.LayerNorm(d)
        self.ln_q = nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn * d), nn.SiLU(), nn.Linear(ffn * d, d))
        # SurfaceMLP b(t) = sigmoid(MLP([z, PE(log(1 + t / S0)), t / tcap])) with the first layer split (z part once per ray)
        self.s_z = nn.Linear(d, d)
        self.s_t = nn.Linear(2 * n_freq + 2, d)
        self.s_out = nn.Sequential(nn.SiLU(), nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, 1))
        self.sum_mlp = nn.Sequential(nn.Linear(n_probe * 2 + 1, 64), nn.SiLU(), nn.Linear(64, d_sum))
        self.adapter = nn.Sequential(nn.LayerNorm(d + d_sum), nn.Linear(d + d_sum, out_ch))
        self.out_scale = nn.Parameter(torch.full((out_ch,), float(out_scale_init)))   # V4 ray-context RMS is ~0.03

    # ------------------------------------------------------------------ pieces
    def query(self, o, d, tcap, ch, ext):
        on = (2.0 * o / ext - 1.0).clamp(-2, 2)
        ltc = torch.log(tcap / S0)
        x = torch.cat([fourier(d, self.n_freq), fourier(on, self.n_freq), ltc[:, None] / 5.0, (tcap / ext.max())[:, None], ch], -1)
        return self.q_mlp(x)

    def geo(self, tau, delta, tcap, s, level, obs, reld):
        lt = torch.log(tau.clamp(min=1e-4) / S0)
        cos = (tau / (tau ** 2 + delta ** 2).sqrt().clamp(min=1e-6))
        lv = torch.full_like(tau, float(level))
        return torch.cat([fourier(lt[..., None] / 5.0, self.n_freq), (tau / tcap[:, None])[..., None], (delta / s)[..., None], torch.full_like(tau, math.log(s / S0))[..., None],
                          cos[..., None], lv[..., None], (lv == 2).float()[..., None], obs, reld], -1)

    def surface_logits(self, z, t, tcap):
        """z [N, d], t [N, Q] model units -> logits [N, Q] of 'a surface within the depth neighbourhood of t'."""
        pe = torch.cat([fourier(torch.log1p(t / S0)[..., None] / 5.0, self.n_freq), (t / tcap[:, None])[..., None], torch.log1p(t / S0)[..., None] / 5.0], -1)
        hdn = self.s_z(z)[:, None] + self.s_t(pe)
        return self.s_out(hdn)[..., 0]

    def probes(self, tcap):
        i = (torch.arange(self.n_probe, device=tcap.device).float() + 0.5) / self.n_probe
        lo = math.log(S0)
        return torch.exp(lo + (torch.log(tcap.clamp(min=S0 * 1.01)) - lo)[:, None] * i[None])          # [N, P]

    # ------------------------------------------------------------------ main
    def forward(self, mem, o, d, tcap, ch, return_aux=False):
        """o [N, 3], d [N, 3], tcap [N], ch [N, 11] -> H [N, out_ch] (+ aux: z, belief probes)."""
        N = o.shape[0]
        q = self.query(o, d, tcap, ch, mem.ext)
        toks, geos, oks = [], [], []
        for lv, (Fg, n) in enumerate(((mem.F0, self.n0), (mem.F1, self.n1))):
            idx, ok, tau, delta = retrieve(mem, lv, o, d, tcap, n)
            s = S0 * (1 if lv == 0 else 2)
            flatF = Fg[0].flatten(1).t()                                         # [cells, C]
            obs = (mem.obs0 if lv == 0 else mem.obs1)[0].flatten(1).t().to(flatF.dtype)
            f = flatF[idx]                                                       # [N, n, C]  (gather; no interpolation)
            ob = obs[idx].float()
            reld = d[:, None].expand(N, n, 3)
            e = self.geo(tau, delta, tcap, s, lv, ob, reld)
            toks.append(self.f_proj[lv](f.to(self.f_proj[lv].weight.dtype)))
            geos.append(e); oks.append(ok)
        rel = mem.coarse_pos[None] - o[:, None]                                  # [N, 8, 3]
        tau = (rel * d[:, None]).sum(-1)
        delta = (rel - tau[..., None] * d[:, None]).norm(dim=-1)
        nc = rel.shape[1]
        e = self.geo(tau, delta, tcap, float(mem.coarse_size), 2, mem.coarse_obs[None].expand(N, nc, 3).float(), F.normalize(rel, dim=-1))
        toks.append(self.f_proj[2](mem.coarse_f.to(self.f_proj[2].weight.dtype))[None].expand(N, nc, -1))
        geos.append(e); oks.append(torch.ones(N, nc, dtype=torch.bool, device=o.device))
        tok = torch.cat(toks, 1)
        geo = torch.cat(geos, 1).to(tok.dtype)
        ok = torch.cat(oks, 1)
        kv = self.ln_kv(tok + self.geo_mlp(geo))
        kv = torch.cat([self.null.expand(N, 1, self.d).to(kv.dtype), kv], 1)
        ok = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=o.device), ok], 1)
        bias = torch.cat([torch.zeros(N, 1, self.h, device=o.device, dtype=geo.dtype), self.geo_bias(geo)], 1)          # [N, T, h]
        hd = self.d // self.h
        qh = self.wq(self.ln_q(q)).reshape(N, self.h, 1, hd)
        kh = self.wk(kv).reshape(N, -1, self.h, hd).transpose(1, 2)
        vh = self.wv(kv).reshape(N, -1, self.h, hd).transpose(1, 2)
        logit = (qh.float() @ kh.float().transpose(-1, -2))[:, :, 0] / math.sqrt(hd) + bias.float().transpose(1, 2)   # [N, h, T]
        logit = logit.masked_fill(~ok[:, None], float("-inf"))
        w = torch.softmax(logit, -1)
        att = (w.to(vh.dtype)[:, :, None] @ vh)[:, :, 0].reshape(N, self.d)
        z = q + self.wo(att)
        z = z + self.ffn(z)
        tp = self.probes(tcap)
        bl = self.surface_logits(z, tp, tcap)                                    # [N, P]
        summ = self.sum_mlp(torch.cat([torch.sigmoid(bl), bl.clamp(-10, 10) / 10.0, torch.log(tcap / S0)[:, None] / 5.0], -1))
        H = self.adapter(torch.cat([z, summ], -1)) * self.out_scale
        if return_aux:
            return H, {"z": z, "probe_t": tp, "probe_logit": bl, "attn": w, "valid": ok}
        return H


class V5Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, r, u, v5 = cfg["encoder"], cfg["ray"], cfg["unet"], cfg["v5"]
        self.cfg = cfg
        self.encoder = M.Encoder3D(e["in_ch"], tuple(e["ch"]), e["blocks_per_level"], e["dilation_L2"], e["groups"])
        self.x_ch, self.proj_ch, self.ray_ch = cfg["flow"]["x_ch"], cfg["proj_ch"], r["out_ch"]
        self.cond_ch = self.proj_ch + self.ray_ch + 1
        self.unet = M.UNet2D(self.x_ch + self.cond_ch, self.x_ch, tuple(u["ch"]), u["blocks_per_level"], tuple(u["attn_levels"]), u["heads"], u["t_emb"], u["groups"])
        c0, c1 = self.encoder.out_ch
        self.dec = V5RayDecoder(c0, c1, self.proj_ch, v5["d"], v5["heads"], v5["n0"], v5["n1"], v5["n_freq"], v5["d_sum"], self.ray_ch, v5["n_probe"],
                                out_scale_init=v5.get("out_scale_init", 1.0))
        self.chunk = v5.get("chunk", 16384)
        self.dec_ckpt = True

    def encode(self, coarse_in):
        """coarse_in [1, 4, X', Y', Z'] (futureview.model.encoder_input) -> (F0, F1, dom, coarse_in) ; dom = the in-domain
        fraction channel as in V2 (kept for interface compatibility)."""
        F0, F1 = self.encoder(coarse_in)
        return F0, F1, coarse_in[:, 2:3], coarse_in

    def ray_features(self, h, o, d, tcap, ch, return_aux=False):
        """Per-ray decoder output for rays of ONE state (o [N, 3] per ray): H [N, 64] (+ aux), chunked."""
        F0, F1, _, cin = h
        mem = Memory(F0, F1, cin)
        outs, auxs = [], []
        for s in range(0, o.shape[0], self.chunk):
            sl = slice(s, s + self.chunk)
            args = (o[sl], d[sl], tcap[sl], ch[sl])
            if self.training and torch.is_grad_enabled() and self.dec_ckpt and not return_aux:
                H = torch.utils.checkpoint.checkpoint(lambda *a: self.dec(mem, *a), *args, use_reentrant=False)
                outs.append(H)
            else:
                r = self.dec(mem, *args, return_aux=return_aux)
                if return_aux:
                    outs.append(r[0]); auxs.append(r[1])
                else:
                    outs.append(r)
        H = torch.cat(outs)
        if return_aux:
            aux = {k: torch.cat([a[k] for a in auxs]) for k in auxs[0]}
            return H, aux, mem
        return H

    def view_cond(self, h, o, d, tcap, ch, return_aux=False):
        """V4-compatible: o [V, 3], d [V, R, 3], tcap [V, R], ch [V, 11, R] -> cond [V, 76, 64, 128] (+ aux)."""
        V, R = d.shape[:2]
        oo = o[:, None].expand(V, R, 3).reshape(-1, 3).float()
        r = self.ray_features(h, oo, d.reshape(-1, 3).float(), tcap.reshape(-1).float(), ch.transpose(1, 2).reshape(-1, ch.shape[1]).float(), return_aux)
        H = r[0] if return_aux else r
        c = torch.cat([ch.to(H.dtype), H.reshape(V, R, -1).transpose(1, 2)], 1).reshape(V, -1, GH, GW)
        c = F.pad(c, (0, GW_PAD - GW))
        valid = torch.zeros(V, 1, GH, GW_PAD, device=c.device, dtype=c.dtype)
        valid[..., :GW] = 1
        cond = torch.cat([c, valid], 1)
        return (cond, r[1], r[2]) if return_aux else cond

    def velocity(self, x, tau, cond, return_features=False):
        return self.unet(torch.cat([x, cond.to(x.dtype)], 1), tau, return_features)


def build(cfg):
    return V5Model(cfg)


def load_v4_warmstart(model, v4_state):
    """Strict name + shape copy of the V4 encoder / unet weights into a V5 model. Returns the report."""
    sd = model.state_dict()
    copied, skipped = [], []
    for k, v in v4_state.items():
        if k.startswith("encoder.") or k.startswith("unet."):
            assert k in sd and sd[k].shape == v.shape, ("V4 -> V5 mismatch", k)
            sd[k] = v.clone()
            copied.append(k)
        else:
            skipped.append(k)
    missing = [k for k in sd if (k.startswith("encoder.") or k.startswith("unet.")) and k not in v4_state]
    assert not missing, ("V5 inherited weights missing in V4", missing[:5])
    model.load_state_dict(sd)
    return {"copied": len(copied), "copied_numel": int(sum(v4_state[k].numel() for k in copied)), "not_used_from_v4": sorted({k.split(".")[0] for k in skipped}),
            "new_modules_numel": int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("dec.")))}
