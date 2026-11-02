"""V5.2-M (EXP6 Scratch70k): Idea-2 Geometry Belief Scorer, trained from scratch (no V4 / V5 / V5.2 weights).

D100 coarse P1 [1, 4, X', Y', 32] -> Encoder3D (V2/V4/V5 architecture, random init) -> F0 / F1 scene memory
camera rays (o, d, tcap, 11 analytic projection channels of the D100 map) -> RayQuery (the V5 geometry-guided
local / coarse retrieval + cross-attention, latent z only) -> ray map [V, 64 + 11 + 1, 64, 128]
-> SpatialUNet (deterministic 2D context; GN / SiLU / 3x3 ResBlocks, 4 scales, skip-add; no time, no noise)
-> per-ray context c_r (64)
-> DepthDecoder (per ray, ordered depth axis of N intervals; width 192, depthwise-conv + FFN residual blocks; inputs
   z_r, c_r, depth encodings, the analytic D100 ray structure and the coarse observation state at each interval)
-> surface marginal logit b(r, i) and raw conditional hazard (r, i)
-> D100 analytic constraints BEFORE transmittance:
     FREE: the part of every interval lying in [t_near, t_free_end) and inside the scene domain, with
           t_free_end = min(d_lo, t_known, t_max) (first UNKNOWN in-domain sample, first known TSDF zero crossing,
           grid exit; the planner's own cast), contributes NO hazard: a_i = rho_i * dt_i * (1 - free_frac_i) / S0
     BAND: only the trusted first visible surface (the TSDF zero crossing t_known) gets a learned, non-negative hazard
           bias in its interval (soft constraint; thick BAND without a crossing is not forced to hit)
     UNKNOWN: free network prediction
-> p_first / p_nohit in log space (fp32), sum = 1 (v52_model.firsthit).
Intervals: v52_model.intervals (uniform in log1p(t / 16 VS) on [tcap * 0.5 / 70, tcap]).
There is no flow, no K-world sampling, no depth image and no DistP1 in this model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from seen2scene.futureview import model as M
from seen2scene.futureview import v5_model as V5
from seen2scene.futureview import v52_model as V52

VS, S0 = M.VS, V5.S0
GH, GW, GW_PAD = M.GH, M.GW, M.GW_PAD


# ----------------------------------------------------------------------------------------------------------- ray query
class RayQuery(V5.V5RayDecoder):
    """V5RayDecoder without its surface / summary / adapter heads: latent z_r only (random init)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for nme in ("s_z", "s_t", "s_out", "sum_mlp", "adapter"):
            delattr(self, nme)
        del self.out_scale

    def latent(self, mem, o, d, tcap, ch):
        N = o.shape[0]
        q = self.query(o, d, tcap, ch, mem.ext)
        toks, geos, oks = [], [], []
        for lv, (Fg, n) in enumerate(((mem.F0, self.n0), (mem.F1, self.n1))):
            idx, ok, tau, delta = V5.retrieve(mem, lv, o, d, tcap, n)
            s = S0 * (1 if lv == 0 else 2)
            flatF = Fg[0].flatten(1).t()
            obs = (mem.obs0 if lv == 0 else mem.obs1)[0].flatten(1).t().to(flatF.dtype)
            f = flatF[idx]
            ob = obs[idx].float()
            e = self.geo(tau, delta, tcap, s, lv, ob, d[:, None].expand(N, n, 3))
            toks.append(self.f_proj[lv](f.to(self.f_proj[lv].weight.dtype)))
            geos.append(e); oks.append(ok)
        rel = mem.coarse_pos[None] - o[:, None]
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
        bias = torch.cat([torch.zeros(N, 1, self.h, device=o.device, dtype=geo.dtype), self.geo_bias(geo)], 1)
        hd = self.d // self.h
        qh = self.wq(self.ln_q(q)).reshape(N, self.h, 1, hd)
        kh = self.wk(kv).reshape(N, -1, self.h, hd).transpose(1, 2)
        vh = self.wv(kv).reshape(N, -1, self.h, hd).transpose(1, 2)
        logit = (qh.float() @ kh.float().transpose(-1, -2))[:, :, 0] / math.sqrt(hd) + bias.float().transpose(1, 2)
        logit = logit.masked_fill(~ok[:, None], float("-inf"))
        w = torch.softmax(logit, -1)
        att = (w.to(vh.dtype)[:, :, None] @ vh)[:, :, 0].reshape(N, self.d)
        z = q + self.wo(att)
        return z + self.ffn(z)


# ----------------------------------------------------------------------------------------------------------- 2D context
def gn(c):
    return nn.GroupNorm(min(32, c // 4), c)


class ResBlock2D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SpatialUNet(nn.Module):
    """Deterministic 2D U-Net over the camera ray grid [V, cin, 64, 128] -> [V, cout, 64, 128]."""

    def __init__(self, cin, cout, ch=(96, 192, 384, 384), blocks=2):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.down.append(nn.Sequential(*[ResBlock2D(c if j == 0 else co, co) for j in range(blocks)]))
            c = co
            self.pool.append(nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity())
        self.up_proj, self.up = nn.ModuleList(), nn.ModuleList()
        for i in range(len(ch) - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(ch[i + 1], ch[i], 3, padding=1))
            self.up.append(nn.Sequential(*[ResBlock2D(ch[i], ch[i]) for _ in range(blocks)]))
        self.out = nn.Sequential(gn(ch[0]), nn.SiLU(), nn.Conv2d(ch[0], cout, 3, padding=1))
        self.use_ckpt = False

    def _run(self, m, x):
        return ckpt.checkpoint(m, x, use_reentrant=False) if (self.use_ckpt and self.training and torch.is_grad_enabled()) else m(x)

    def forward(self, x):
        x = self.inp(x)
        skips = []
        for blk, pool in zip(self.down, self.pool):
            x = self._run(blk, x)
            skips.append(x)
            x = pool(x)
        for k, (proj, blk) in enumerate(zip(self.up_proj, self.up)):
            s = skips[len(skips) - 2 - k]
            x = proj(F.interpolate(x, size=s.shape[-2:], mode="nearest")) + s
            x = self._run(blk, x)
        return self.out(x)


# ----------------------------------------------------------------------------------------------------------- depth decoder
class DepthBlock(nn.Module):
    def __init__(self, w, k=5, mult=4):
        super().__init__()
        self.n1 = nn.LayerNorm(w)
        self.dw = nn.Conv1d(w, w, k, padding=k // 2, groups=w)
        self.n2 = nn.LayerNorm(w)
        self.ffn = nn.Sequential(nn.Linear(w, mult * w), nn.SiLU(), nn.Linear(mult * w, w))

    def forward(self, x):                                   # x [R, N, w]
        x = x + self.dw(self.n1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ffn(self.n2(x))


N_FREQ = 6
F_BIN = 2 * N_FREQ + 2 + 8 + 4 + 3                         # depth PE, t / tcap, log dt; analytic (8); coarse obs (4); class one-hot (3)


class DepthDecoder(nn.Module):
    def __init__(self, d_ray, d_ctx, w=192, n_blocks=6, k=5, mult=4, rho0=0.05, known_bias0=3.0):
        super().__init__()
        self.ray_in = nn.Linear(d_ray + d_ctx, w)
        self.bin_in = nn.Linear(F_BIN, w)
        self.blocks = nn.ModuleList([DepthBlock(w, k, mult) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(w)
        self.head = nn.Linear(w, 2)                          # surface logit, raw hazard
        self.known_bias = nn.Parameter(torch.tensor(math.log(math.expm1(known_bias0))))
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.copy_(torch.tensor([-2.0, math.log(math.expm1(rho0))]))
        self.use_ckpt = False

    def forward(self, z, c, fb):
        """z [R, d_ray], c [R, d_ctx], fb [R, N, F_BIN] -> logit_b [R, N], raw [R, N]."""
        x = self.ray_in(torch.cat([z, c.to(z.dtype)], -1))[:, None] + self.bin_in(fb.to(self.bin_in.weight.dtype))
        for blk in self.blocks:
            x = ckpt.checkpoint(blk, x, use_reentrant=False) if (self.use_ckpt and self.training and torch.is_grad_enabled()) else blk(x)
        out = self.head(self.norm(x)).float()
        return out[..., 0], out[..., 1]


# ----------------------------------------------------------------------------------------------------------- analytic ray structure
def ray_structure(edges, tcap, stats, n_sub=4):
    """D100 analytic structure of each interval from the planner's per-ray projection statistics.
    edges [R, N + 1]; tcap [R]; stats dict: tk, dlo, dhi (float16 statistics relative to tcap, 1 = none), cls [R],
    tmax [R] (grid exit, <= tcap), dom_fn(points [R, N, S, 3]) -> in-domain fraction of the coarse cells (or None).
    Returns free_frac [R, N] (part of the interval forced FREE), feats [R, N, 8], known_here [R, N] (interval holds
    the TSDF zero crossing t_known)."""
    tk = stats["tk"].float() * tcap
    dlo = stats["dlo"].float() * tcap
    dhi = stats["dhi"].float() * tcap
    has_k = stats["tk"].float() < 0.999
    has_u = stats["dlo"].float() < 0.999
    inf = torch.full_like(tcap, float("inf"))
    t_k = torch.where(has_k, tk, inf)
    t_lo = torch.where(has_u, dlo, inf)
    t_hi = torch.where(has_u, dhi, inf)
    tmax = stats["tmax"].float()
    t_free_end = torch.minimum(torch.minimum(t_lo, t_k), tmax)
    a, b = edges[:, :-1], edges[:, 1:]
    dt = (b - a).clamp(min=1e-9)
    fr = (torch.minimum(b, t_free_end[:, None]) - a).clamp(min=0) / dt
    if stats.get("dom") is not None:                       # only in-domain parts are forced FREE
        fr = fr * stats["dom"]
    unk = ((torch.minimum(b, t_hi[:, None] + 2 * VS) - torch.maximum(a, t_lo[:, None])).clamp(min=0) / dt).clamp(max=1)
    beyond = ((b - torch.maximum(a, tmax[:, None])).clamp(min=0) / dt).clamp(max=1)
    known_here = (a <= t_k[:, None]) & (b > t_k[:, None])
    after_k = (a >= t_k[:, None]).float()
    mid = 0.5 * (a + b)

    def slog(x):
        x = torch.where(torch.isfinite(x), x, torch.full_like(x, 1e4))
        return torch.sign(x) * torch.log1p(x.abs() / S0) / 5.0
    feats = torch.stack([fr, unk, beyond, known_here.float(), after_k, slog(mid - t_k[:, None]), slog(mid - t_lo[:, None]), slog(mid - t_hi[:, None])], -1)
    return fr.clamp(0, 1), feats, known_here


def ray_stats_from_view(view):
    """Planner view (v2_plan_core.Geometry.view) -> stats of ray_structure (float16-rounded exactly as the training data)."""
    from seen2scene.futureview import geometry as G
    pr, tcap = view["pr"], view["tcap"]
    st = G.stats_fp16(pr, tcap)
    return {"tk": st[0], "dlo": st[1], "dhi": st[2], "cls": pr["cls"], "tmax": torch.minimum(pr["tmax"].to(tcap.dtype), tcap)}


def tmax_from_dims(o, d, tcap, dims):
    """Grid exit (model units) of rays o + t d inside [0, dims * VS], capped at tcap (geometry.march_range)."""
    from seen2scene.futureview import geometry as G
    ext = torch.as_tensor(dims, device=d.device, dtype=d.dtype) * VS
    _, tmax = G.entry_exit(o, d, ext)
    return torch.minimum(tmax, tcap)


# ----------------------------------------------------------------------------------------------------------- model
class V52MModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, rq, sp, dd = cfg["encoder"], cfg["ray_query"], cfg["spatial"], cfg["depth"]
        self.cfg = cfg
        self.encoder = M.Encoder3D(e["in_ch"], tuple(e["ch"]), e["blocks_per_level"], e["dilation_L2"], e["groups"])
        c0, c1 = self.encoder.out_ch
        self.proj_ch = cfg["proj_ch"]
        self.rays = RayQuery(c0, c1, self.proj_ch, rq["d"], rq["heads"], rq["n0"], rq["n1"], rq["n_freq"], 32, 64, 16)
        self.ray_map = nn.Linear(rq["d"], sp["ray_ch"])
        self.spatial = SpatialUNet(sp["ray_ch"] + self.proj_ch + 1, sp["out_ch"], tuple(sp["ch"]), sp["blocks"])
        self.depth = DepthDecoder(rq["d"], sp["out_ch"], dd["width"], dd["blocks"], dd["kernel"], dd["mult"], dd["rho0"], dd["known_bias0"])
        self.n_bins, self.c_bins = dd["n_bins"], dd["c_bins_vox"] * VS
        self.n_sub_dom = dd.get("n_sub_dom", 4)
        self.ray_chunk = rq.get("chunk", 16384)
        self.dec_chunk = dd.get("chunk", 4096)

    # -------------------------------------------------------------- scene / rays
    def encode(self, coarse_in):
        F0, F1 = self.encoder(coarse_in)
        return F0, F1, coarse_in[:, 2:3], coarse_in

    def memory(self, h):
        return V5.Memory(h[0], h[1], h[3])

    def ray_latents(self, mem, o, d, tcap, ch, use_ckpt=False):
        out = []
        for s in range(0, o.shape[0], self.ray_chunk):
            sl = slice(s, s + self.ray_chunk)
            args = (o[sl], d[sl], tcap[sl], ch[sl])
            if use_ckpt and torch.is_grad_enabled():
                out.append(ckpt.checkpoint(lambda *a: self.rays.latent(mem, *a), *args, use_reentrant=False))
            else:
                out.append(self.rays.latent(mem, *args))
        return torch.cat(out)

    def context(self, z, ch):
        """z [V, R, d_ray], ch [V, 11, R] -> context [V, R, d_ctx] (2D spatial U-Net over the 64 x 114 ray grid)."""
        V, R = z.shape[:2]
        m = torch.cat([self.ray_map(z).transpose(1, 2), ch.to(z.dtype)], 1).reshape(V, -1, GH, GW)
        m = F.pad(m, (0, GW_PAD - GW))
        valid = torch.zeros(V, 1, GH, GW_PAD, device=m.device, dtype=m.dtype)
        valid[..., :GW] = 1
        c = self.spatial(torch.cat([m, valid], 1))[..., :GW]                    # [V, d_ctx, 64, 114]
        return c.reshape(V, c.shape[1], R).transpose(1, 2)

    # -------------------------------------------------------------- depth axis
    def coarse_obs(self, mem, p):
        return V52.obs_at(mem, p)

    def decode(self, mem, o, d, tcap, z, c, stats):
        """Rays (any subset): o [R, 3] d [R, 3] tcap [R] z [R, d_ray] c [R, d_ctx], stats (ray_structure) -> dict."""
        outs = []
        R = o.shape[0]
        for s in range(0, R, self.dec_chunk):
            sl = slice(s, min(R, s + self.dec_chunk))
            outs.append(self._decode(mem, o[sl], d[sl], tcap[sl], z[sl], c[sl], {k: (v[sl] if torch.is_tensor(v) else v) for k, v in stats.items()}))
        return {k: torch.cat([x[k] for x in outs]) for k in outs[0]}

    def _decode(self, mem, o, d, tcap, z, c, stats):
        tcap = tcap.float()
        edges = V52.intervals(tcap, self.n_bins, self.c_bins)
        t_mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
        dt = (edges[:, 1:] - edges[:, :-1]).clamp(min=1e-6)
        p = o[:, None] + d[:, None] * t_mid[..., None]
        obs = self.coarse_obs(mem, p)                                            # [R, N, 4]
        # in-domain fraction of each interval from n_sub sub-points of the coarse grid (FREE is forced in-domain only)
        fr_s = (torch.arange(self.n_sub_dom, device=o.device).float() + 0.5) / self.n_sub_dom
        ps = o[:, None, None] + d[:, None, None] * (edges[:, :-1, None] + dt[..., None] * fr_s)[..., None]
        dom = self.coarse_obs(mem, ps.reshape(o.shape[0], -1, 3))[..., 2].reshape(o.shape[0], self.n_bins, self.n_sub_dom)
        dom = (dom > 0.5).float().mean(-1)
        st = dict(stats, dom=dom)
        free, af, known_here = ray_structure(edges, tcap, st)
        oh = F.one_hot(stats["cls"].long(), 3).float()[:, None].expand(-1, self.n_bins, 3)
        lt = torch.log1p(t_mid / S0)[..., None] / 5.0
        fb = torch.cat([V5.fourier(lt, N_FREQ), (t_mid / tcap[:, None])[..., None], torch.log(dt / S0).clamp(-10, 10)[..., None] / 5.0, af, obs, oh], -1)
        logit_b, raw = self.depth(z, c, fb)
        raw = raw + F.softplus(self.depth.known_bias) * known_here.float()
        rho = F.softplus(raw)
        a = rho * dt * (1.0 - free) / S0                                         # FREE removed BEFORE transmittance
        out = V52.firsthit(a)
        out.update({"edges": edges, "t_mid": t_mid, "dt": dt, "logit_b": logit_b, "rho": rho, "free_frac": free, "known_here": known_here})
        return out

    # -------------------------------------------------------------- inference
    def view_belief(self, h, view, mem=None):
        """All rays of one planner view -> decode() outputs (+ z, c)."""
        from seen2scene.futureview import geometry as G
        mem = mem or self.memory(h)
        o, d, tcap = view["o"].float(), view["d"].float(), view["tcap"].float()
        ch = G.channels(view["pr"], view["tcap"], view["d"], view["o"]).float()          # [11, R]
        R = d.shape[0]
        z = self.ray_latents(mem, o[None].expand(R, 3), d, tcap, ch.t())
        c = self.context(z[None], ch[None])[0]
        st = ray_stats_from_view(view)
        out = self.decode(mem, o[None].expand(R, 3), d, tcap, z, c, st)
        out.update({"z": z, "c": c})
        return out


def param_table(m):
    rows = {}
    for nme, mod in (("encoder (3D U-Net)", m.encoder), ("ray query", m.rays), ("ray map", m.ray_map), ("spatial 2D U-Net", m.spatial), ("depth decoder", m.depth)):
        rows[nme] = sum(p.numel() for p in mod.parameters())
    rows["total"] = sum(p.numel() for p in m.parameters())
    rows["trainable"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return rows
