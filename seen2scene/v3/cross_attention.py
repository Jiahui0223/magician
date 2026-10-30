"""Camera-ray cross-attention onto the VAE scene memory (EXP6 V3 model T1 only).

Scene memory: one token per latent cell (mu at 8x fine voxels) that historical observation supports (band or free
fraction of the cell > 0; UNKNOWN / out-of-domain cells are never tokens): features [mu (C), band8, free8] + Fourier
positional encoding of the cell centre in the model (region / scene) frame, normalised by the grid extent. The memory
is a function of the map only, so it is shared by every candidate, Euler step and flow sample of one state.
Per candidate view: deterministic frustum selection (cell centre within the camera's field of view + margin and within
the ray cap), at most `budget` tokens (uniform stride over the distance-sorted set), padded with a key mask.
Queries: the flow U-Net bottleneck tokens (8 x 16 cells of the 64 x 128 map) + Fourier encoding of the mean ray
direction of the cell and of the camera position, so Q depends on the candidate camera and on the generation state.
Block: H' = H + W_o Attn(W_q LN(H + q_pe), W_k LN(M), W_v M), W_o initialised to zero (T1 starts as T0).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FOV_DEG = 60.0


def fourier(x, n_freq):
    """x [..., D] in roughly [-1, 1] -> [..., D * 2 * n_freq]."""
    f = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


class SceneMemory:
    """Tokens of one state: feat [N, C + 2], pos [N, 3] (model units), pos_n [N, 3] in [0, 1]."""

    def __init__(self, mu, valid8, band8, free8, ext):
        _, C, X, Y, Z = mu.shape
        obs = ((band8 > 0) | (free8 > 0)) & (valid8 > 0)                      # [1, 1, X, Y, Z]
        idx = torch.nonzero(obs[0, 0], as_tuple=False)                         # [N, 3]
        self.feat = torch.cat([mu[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].t(),
                               band8[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].t().to(mu.dtype),
                               free8[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].t().to(mu.dtype)], 1)   # [N, C + 2]
        cell = ext.to(mu.device).float() / torch.tensor([X, Y, Z], device=mu.device, dtype=torch.float32)
        self.pos = (idx.float() + 0.5) * cell                                  # model units
        self.pos_n = self.pos / ext.to(mu.device).float()
        self.n = idx.shape[0]

    def select(self, o, d, tcap, budget, margin_deg):
        """Frustum tokens of one view: o [3], d [R, 3] (unit), tcap [R] -> indices [<= budget] (int64)."""
        if self.n == 0:
            return torch.zeros(0, dtype=torch.long, device=o.device)
        axis = F.normalize(d.float().mean(0), dim=0)
        rel = self.pos - o.float()[None]
        dist = rel.norm(dim=1)
        cosang = (rel @ axis) / dist.clamp(min=1e-6)
        half = math.radians(FOV_DEG / 2.0 * math.sqrt(2.0) + margin_deg)          # corner rays of the 456 x 256 image
        keep = (cosang >= math.cos(half)) & (dist <= float(tcap.max()))
        idx = torch.nonzero(keep, as_tuple=False)[:, 0]
        if idx.numel() > budget:
            order = idx[torch.argsort(dist[idx])]
            pick = torch.linspace(0, order.numel() - 1, budget, device=o.device).round().long()
            idx = order[pick]
        return idx

    def batch(self, views, budget, margin_deg, n_freq):
        """views: [(o [3], d [R, 3], tcap [R])] -> tokens [V, budget, F], key mask [V, budget] (True = valid)."""
        V = len(views)
        Fdim = self.feat.shape[1] + 6 * n_freq
        tok = torch.zeros(V, budget, Fdim, device=self.feat.device, dtype=self.feat.dtype)
        mask = torch.zeros(V, budget, dtype=torch.bool, device=self.feat.device)
        for i, (o, d, tcap) in enumerate(views):
            idx = self.select(o, d, tcap, budget, margin_deg)
            n = idx.numel()
            if n:
                tok[i, :n] = torch.cat([self.feat[idx], fourier(2.0 * self.pos_n[idx] - 1.0, n_freq).to(self.feat.dtype)], 1)
                mask[i, :n] = True
        return tok, mask


def query_pe(o, d, ext, n_freq, gh=8, gw=16, GH=64, GW=114, GW_PAD=128):
    """Per bottleneck cell: Fourier(mean ray direction) + Fourier(camera position / extent) -> [gh * gw, 12 * n_freq]."""
    dd = F.pad(d.float().reshape(GH, GW, 3).permute(2, 0, 1), (0, GW_PAD - GW))                 # [3, 64, 128]
    pooled = F.normalize(F.avg_pool2d(dd[None], (GH // gh, GW_PAD // gw))[0].permute(1, 2, 0).reshape(-1, 3), dim=1)
    pos = (2.0 * o.float() / ext.to(o.device).float() - 1.0)[None].expand(gh * gw, 3)
    return torch.cat([fourier(pooled, n_freq), fourier(pos, n_freq)], 1)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, mem_dim, q_pe_dim, heads=4):
        super().__init__()
        self.h = heads
        self.ln_q, self.ln_k = nn.LayerNorm(dim), nn.LayerNorm(mem_dim)
        self.q_pe = nn.Linear(q_pe_dim, dim)
        self.wq, self.wk, self.wv = nn.Linear(dim, dim), nn.Linear(mem_dim, dim), nn.Linear(mem_dim, dim)
        self.wo = nn.Linear(dim, dim)
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.wo.bias)

    def forward(self, hmap, mem_tok, mem_mask, qpe):
        """hmap [B, C, gh, gw]; mem_tok [B, N, F]; mem_mask [B, N] bool; qpe [B, gh * gw, P] -> [B, C, gh, gw]."""
        B, C, gh, gw = hmap.shape
        htok = hmap.flatten(2).transpose(1, 2)                                     # [B, L, C]
        q = self.wq(self.ln_q(htok + self.q_pe(qpe.to(htok.dtype))))
        kin = self.ln_k(mem_tok.to(htok.dtype))
        k, v = self.wk(kin), self.wv(mem_tok.to(htok.dtype))
        hd = C // self.h
        q = q.reshape(B, -1, self.h, hd).transpose(1, 2)
        k = k.reshape(B, -1, self.h, hd).transpose(1, 2)
        v = v.reshape(B, -1, self.h, hd).transpose(1, 2)
        am = mem_mask[:, None, None, :]                                            # [B, 1, 1, N]
        empty = ~mem_mask.any(1)                                                   # views without any token
        am = am | empty[:, None, None, None]                                       # avoid all-masked rows (NaN)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        out = out.transpose(1, 2).reshape(B, -1, C)
        out = torch.where(empty[:, None, None], torch.zeros_like(out), out)
        return hmap + self.wo(out).transpose(1, 2).reshape(B, C, gh, gw)
