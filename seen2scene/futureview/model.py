"""V2 network (EXP5 Task 1, architecture §1-§4): shared 3D geometry encoder, ray-context encoder, 2D conditional
rectified-flow U-Net over the (b, y) map of the 64 x 114 scoring rays.

Tensor layout: coarse grids [B, 4, X', Y', 32] (EMPTY, BAND, in-domain fractions, z / 32); scene / model frame z-up,
cell (i, j, k) of F0 covers [8 i, 8 i + 8) voxels; rays in model units. Views: [V, C, 64, 114] padded to [V, C, 64, 128].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

VS = 0.011
BLOCK = 8
GH, GW, GW_PAD = 64, 114, 128


def gn(c, groups):
    g = min(groups, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


# ---------------------------------------------------------------------------------------------------- 3D encoder
class Res3D(nn.Module):
    def __init__(self, c, groups=8, dilation=1):
        super().__init__()
        self.n1, self.c1 = gn(c, groups), nn.Conv3d(c, c, 3, padding=dilation, dilation=dilation)
        self.n2, self.c2 = gn(c, groups), nn.Conv3d(c, c, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class Encoder3D(nn.Module):
    """[B, 4, X', Y', 32] -> F0 [B, c0, X', Y', 32], F1 [B, c1, X'/2, Y'/2, 16] (X', Y' multiples of 4)."""

    def __init__(self, in_ch=4, ch=(32, 64, 96), blocks=2, dilation_L2=2, groups=8):
        super().__init__()
        c0, c1, c2 = ch
        self.stem = nn.Conv3d(in_ch, c0, 3, padding=1)
        self.l0 = nn.Sequential(*[Res3D(c0, groups) for _ in range(blocks)])
        self.d1 = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.l1 = nn.Sequential(*[Res3D(c1, groups) for _ in range(blocks)])
        self.d2 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.l2 = nn.Sequential(*[Res3D(c2, groups, dilation_L2) for _ in range(blocks)])
        self.u1 = nn.Conv3d(c2 + c1, c1, 3, padding=1)
        self.r1 = Res3D(c1, groups)
        self.u0 = nn.Conv3d(c1 + c0, c0, 3, padding=1)
        self.r0 = Res3D(c0, groups)
        self.out_ch = (c0, c1)

    def forward(self, x):
        x0 = self.l0(self.stem(x))
        x1 = self.l1(self.d1(F.silu(x0)))
        x2 = self.l2(self.d2(F.silu(x1)))
        y1 = self.r1(self.u1(F.silu(torch.cat([F.interpolate(x2, size=x1.shape[2:], mode="nearest"), x1], 1))))
        y0 = self.r0(self.u0(F.silu(torch.cat([F.interpolate(y1, size=x0.shape[2:], mode="nearest"), x0], 1))))
        return y0, y1


def encoder_input(coarse_u8):
    """uint8 [3, X', Y', Z'] (t11 coarse grid) -> float [1, 4, X', Y', Z'] with the z channel."""
    x = coarse_u8.float()[None] / 255.0
    z = (torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) + 0.5) / x.shape[-1]
    return torch.cat([x, z.view(1, 1, 1, 1, -1).expand(1, 1, *x.shape[2:])], 1)


# ---------------------------------------------------------------------------------------------------- ray context
class RayEncoder(nn.Module):
    """Features of N points t_j = (j + 0.5) / N * t_cap along every ray, sampled from (F0, F1) -> [R, out_ch]."""

    def __init__(self, n_points=32, feat_ch=96, hidden=64, out_ch=64):
        super().__init__()
        self.n = n_points
        self.c1 = nn.Conv1d(feat_ch + 1, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
        self.c3 = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
        self.fc = nn.Linear(hidden * (n_points // 4), out_ch)

    def points(self, o, d, tcap):
        t = (torch.arange(self.n, device=d.device, dtype=d.dtype) + 0.5) / self.n
        o = o[None, None] if o.dim() == 1 else o[:, None]
        return o + d[:, None] * (tcap[:, None] * t[None])[..., None]                       # [R, N, 3]

    def point_features(self, F0, F1, dom, o, d, tcap):
        """Per-point features [R, C + 1, N] (F0, F1 trilinear, in-domain flag x inside-grid) and the inside-grid mask
        [R, N] of the N sample points of every ray (the first half of the V2 forward, unchanged)."""
        p = self.points(o, d, tcap)
        ext = torch.tensor([s * BLOCK * VS for s in F0.shape[2:]], device=p.device, dtype=p.dtype)
        g = 2.0 * p / ext - 1.0
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)[None, :, :, None].float()     # [1, R, N, 1, 3]
        with torch.autocast("cuda", enabled=False):          # float32 sampling (bf16 atomics in backward are slow)
            f0 = F.grid_sample(F0.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            f1 = F.grid_sample(F1.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            fd = F.grid_sample(dom.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        inside_pt = ((g > -1) & (g < 1)).all(-1)                                                # [R, N]
        inside = inside_pt[None, None, ..., None].float()
        f = torch.cat([f0, f1, (fd > 0).float() * inside], 1)[0, ..., 0].permute(1, 0, 2)       # [R, C, N]
        return f, inside_pt

    def head(self, f):
        """[R, C + 1, N] -> ray context [R, out_ch] (the Conv1d stack + fc of V2, unchanged)."""
        h = F.silu(self.c1(f))
        h = F.silu(self.c2(h))
        h = F.silu(self.c3(h))
        return self.fc(h.flatten(1))

    def forward(self, F0, F1, dom, o, d, tcap):
        """F0 [1, c0, X', Y', Z'], F1 [1, c1, ...], dom [1, 1, X', Y', Z'] (in-domain fraction); o [3] or [R, 3]
        (rays of several views of one state concatenated), d [R, 3], tcap [R]."""
        f, _ = self.point_features(F0, F1, dom, o, d, tcap)
        return self.head(f)


# ---------------------------------------------------------------------------------------------------- V4 ray-aligned cross-attention
def fourier_feats(x, n_freq):
    """x [..., D] (roughly in [-1, 1]) -> [..., D * 2 * n_freq] (sin / cos of 2^k pi x)."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class RayXAttn(nn.Module):
    """V4 (EXP6): ray-aligned cross-attention over the N sample points of EACH ray (no [R, R] attention).

    query  q = W_q [ray context (V2 ray encoder output), Fourier(ray direction), Fourier(camera position / extent)]
    keys / values from the per-point tokens [point features (F0, F1, in-domain flag), s_j = (j + 0.5) / N,
           Fourier(2 s_j - 1), t_j / ext_max] (LayerNorm on the token before W_k / W_v)
    masked softmax over the points inside the grid (UNKNOWN points are legal tokens; rays with no inside point give 0)
    output W_o(context), W_o initialised small (normal, std out_init_std, zero bias): the module starts as a small
    perturbation of the V2 ray context and every projection receives a non-zero gradient from step 0.
    Returns the residual [R, out_ch] that view_cond adds to the V2 ray context."""

    def __init__(self, feat_ch, n_points, ray_ch, dim=64, heads=4, n_freq=4, out_init_std=0.005):
        super().__init__()
        assert dim % heads == 0
        self.n, self.h, self.dim, self.n_freq = n_points, heads, dim, n_freq
        self.geom_ch = 1 + 2 * n_freq + 1
        self.ln_k = nn.LayerNorm(feat_ch + self.geom_ch)
        self.wq = nn.Linear(ray_ch + 2 * 3 * 2 * n_freq, dim)
        self.wk = nn.Linear(feat_ch + self.geom_ch, dim)
        self.wv = nn.Linear(feat_ch + self.geom_ch, dim)
        self.wo = nn.Linear(dim, ray_ch)
        nn.init.normal_(self.wo.weight, std=out_init_std)
        nn.init.zeros_(self.wo.bias)
        self.last_attn = None

    def forward(self, f, inside, rf, o, d, tcap, ext, keep_attn=False):
        """f [R, C, N] point features, inside [R, N] bool, rf [R, ray_ch] V2 ray context, o [R, 3], d [R, 3] (unit),
        tcap [R], ext [3] grid extent (model units) -> residual [R, ray_ch]."""
        R, C, N = f.shape
        dt = rf.dtype
        s = (torch.arange(N, device=f.device, dtype=torch.float32) + 0.5) / N                     # [N]
        ext_max = ext.max().float()
        t_rel = (tcap.float()[:, None] * s[None]) / ext_max                                         # [R, N]
        geom = torch.cat([s[None].expand(R, N)[..., None], fourier_feats((2.0 * s - 1.0)[:, None], self.n_freq)[None].expand(R, N, -1),
                          t_rel[..., None]], -1)                                                     # [R, N, geom_ch]
        tok = torch.cat([f.permute(0, 2, 1).float(), geom], -1)                                      # [R, N, C + geom_ch]
        tok = self.ln_k(tok.to(dt))
        qin = torch.cat([rf, fourier_feats(d.float(), self.n_freq).to(dt),
                         fourier_feats((2.0 * o.float() / ext.float() - 1.0).clamp(-1.0, 1.0), self.n_freq).to(dt)], -1)
        hd = self.dim // self.h
        q = self.wq(qin).reshape(R, self.h, 1, hd)
        k = self.wk(tok).reshape(R, N, self.h, hd).transpose(1, 2)                                   # [R, h, N, hd]
        v = self.wv(tok).reshape(R, N, self.h, hd).transpose(1, 2)
        empty = ~inside.any(1)                                                                       # rays without inside points
        am = (inside | empty[:, None])[:, None, None, :]                                             # [R, 1, 1, N] (no all-masked rows)
        # explicit per-ray attention (N = 32 keys per query): float32 softmax, deterministic, no fused-kernel batch limits
        logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)                           # [R, h, 1, N]
        w = torch.softmax(logits.masked_fill(~am, float("-inf")), -1)
        w = torch.where(empty[:, None, None, None], torch.zeros_like(w), w)                          # all-masked rays -> 0
        if keep_attn:
            self.last_attn = w[:, :, 0].detach()                                                     # [R, h, N]
        out = (w.to(v.dtype) @ v).reshape(R, self.dim)
        return self.wo(out)


# ---------------------------------------------------------------------------------------------------- 2D flow U-Net
def timestep_embedding(tau, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
    a = (tau.float() * 1000.0)[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class Res2D(nn.Module):
    def __init__(self, cin, cout, temb, groups):
        super().__init__()
        self.n1, self.c1 = gn(cin, groups), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(temb, 2 * cout)
        self.n2, self.c2 = gn(cout, groups), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(te)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        return self.skip(x) + self.c2(F.silu(h))


class Attn2D(nn.Module):
    def __init__(self, c, heads, groups):
        super().__init__()
        self.n, self.h = gn(c, groups), heads
        self.qkv, self.proj = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch=2, ch=(64, 128, 192, 256), blocks=2, attn_levels=(3,), heads=4, t_emb=256, groups=16):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, t_emb), nn.SiLU(), nn.Linear(t_emb, t_emb))
        self.inp = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.down, self.skips_ch = nn.ModuleList(), [ch[0]]
        c = ch[0]
        for lv, co in enumerate(ch):
            for _ in range(blocks):
                m = nn.ModuleList([Res2D(c, co, t_emb, groups)])
                if lv in attn_levels:
                    m.append(Attn2D(co, heads, groups))
                self.down.append(m)
                c = co
                self.skips_ch.append(c)
            if lv < len(ch) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                self.skips_ch.append(c)
        self.mid = nn.ModuleList([Res2D(c, c, t_emb, groups), Attn2D(c, heads, groups), Res2D(c, c, t_emb, groups)])
        self.up = nn.ModuleList()
        skips = list(self.skips_ch)
        for lv in reversed(range(len(ch))):
            co = ch[lv]
            for _ in range(blocks + 1):
                m = nn.ModuleList([Res2D(c + skips.pop(), co, t_emb, groups)])
                if lv in attn_levels:
                    m.append(Attn2D(co, heads, groups))
                self.up.append(m)
                c = co
            if lv > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        self.out = nn.Sequential(gn(c, groups), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, tau, return_features=False):
        te = self.temb(timestep_embedding(tau))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            if isinstance(m[0], Res2D):
                h = m[0](h, te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[0](h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for m in self.up:
            if isinstance(m[0], Res2D):
                h = m[0](torch.cat([h, hs.pop()], 1), te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[1](m[0](h))
        out = self.out(h)
        return (out, h) if return_features else out


# ---------------------------------------------------------------------------------------------------- V2 model
class V2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        e, r, u = cfg["encoder"], cfg["ray"], cfg["unet"]
        self.encoder = Encoder3D(e["in_ch"], tuple(e["ch"]), e["blocks_per_level"], e["dilation_L2"], e["groups"])
        assert sum(self.encoder.out_ch) == r["feat_ch"], (self.encoder.out_ch, r["feat_ch"])
        self.rays = RayEncoder(r["n_points"], r["feat_ch"], r["hidden"], r["out_ch"])
        self.x_ch, self.proj_ch, self.ray_ch = cfg["flow"]["x_ch"], cfg["proj_ch"], r["out_ch"]
        self.cond_ch = self.proj_ch + self.ray_ch + 1
        self.unet = UNet2D(self.x_ch + self.cond_ch, self.x_ch, tuple(u["ch"]), u["blocks_per_level"],
                           tuple(u["attn_levels"]), u["heads"], u["t_emb"], u["groups"])
        xa = cfg.get("ray_xattn", {"enabled": False})            # V4 (EXP6): created last -> V2 parameters keep their init order
        self.ray_xattn = RayXAttn(r["feat_ch"] + 1, r["n_points"], r["out_ch"], xa.get("dim", 64), xa.get("heads", 4),
                                  xa.get("n_freq", 4), xa.get("out_init_std", 0.005)) if xa.get("enabled") else None

    def encode(self, coarse_in):
        """coarse_in [1, 4, X', Y', Z'] (encoder_input) -> (F0, F1, dom)."""
        F0, F1 = self.encoder(coarse_in)
        return F0, F1, coarse_in[:, 2:3]

    def view_cond(self, h, o, d, tcap, ch):
        """Conditioning maps [V, C, 64, 128] of V views of ONE state: o [V, 3], d [V, R, 3], tcap [V, R], ch [V, 11, R]
        (projection channels) -> projection channels + ray context + valid-column flag."""
        F0, F1, dom = h
        V, R = d.shape[:2]
        args = (F0, F1, dom, o[:, None].expand(V, R, 3).reshape(-1, 3), d.reshape(-1, 3), tcap.reshape(-1))
        fn = self.rays if self.ray_xattn is None else self._ray_context_v4
        if self.training and torch.is_grad_enabled() and getattr(self, "rays_ckpt", True):   # recompute in backward (memory)
            rf = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        else:
            rf = fn(*args)
        c = torch.cat([ch.to(rf.dtype), rf.reshape(V, R, -1).transpose(1, 2)], 1).reshape(V, -1, GH, GW)
        c = F.pad(c, (0, GW_PAD - GW))
        valid = torch.zeros(V, 1, GH, GW_PAD, device=c.device, dtype=c.dtype)
        valid[..., :GW] = 1
        return torch.cat([c, valid], 1)

    def _ray_context_v4(self, F0, F1, dom, o, d, tcap, keep_attn=False):
        """V4: V2 ray context + ray-aligned cross-attention residual over the same sample points."""
        f, inside = self.rays.point_features(F0, F1, dom, o, d, tcap)
        rf = self.rays.head(f)
        ext = torch.tensor([s * BLOCK * VS for s in F0.shape[2:]], device=f.device, dtype=torch.float32)
        return rf + self.ray_xattn(f, inside, rf, o, d, tcap, ext, keep_attn=keep_attn).to(rf.dtype)

    def velocity(self, x, tau, cond, return_features=False):
        """x [V, 2, 64, 128] (already masked), tau [V], cond [V, C, 64, 128] -> velocity [V, 2, 64, 128]."""
        return self.unet(torch.cat([x, cond.to(x.dtype)], 1), tau, return_features)


def pad_rays(x):
    """[..., R=7296] -> [..., 64, 128] (zero-padded columns)."""
    return F.pad(x.reshape(*x.shape[:-1], GH, GW), (0, GW_PAD - GW))


def unpad_rays(x):
    return x[..., :GW].reshape(*x.shape[:-2], GH * GW)


@torch.no_grad()
def sample(model, cond, m_unk, eps, steps=10, autocast=True, return_features=False):
    """Euler integration of the rectified flow. cond [V, C, 64, 128]; m_unk [V, 64, 128] bool (M_unknown rays);
    eps [V, 2, 64, 128]. x is 0 outside m_unk at every step (as in training). Returns x1 [V, 2, 64, 128] float32."""
    x = eps * m_unk[:, None]
    feats = None
    for i in range(steps):
        tau = torch.full((x.shape[0],), i / steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model.velocity(x, tau, cond, return_features=return_features and i == steps - 1)
        if return_features and i == steps - 1:
            out, feats = out
        x = (x + out.float() / steps) * m_unk[:, None]
    return (x, feats) if return_features else x
