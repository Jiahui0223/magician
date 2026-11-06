"""Scene keys / values of Experiment XA: observation-supported cells of one scale as tokens, the T1 frustum selection, and the
global token tensor of a batch of views.

Token row (float32): [feature (C), band, free, obs, vfrac, Fourier(2 pos / ext - 1) (6 n_freq), scale id, mask].
F8 tokens are the T1 scene memory (seen2scene.v3.cross_attention.SceneMemory: cells with band or free support inside the
domain, features [mu, band8, free8]); their obs / vfrac entries are 1. F16 tokens come from vae_pyramid.pool16 (cells with
obs16 > 0). Positions of every scale are normalised by the extent of the 4x grid, the frame of the camera encodings.
"""
import math

import torch
import torch.nn.functional as F

from seen2scene.v3.cross_attention import FOV_DEG, fourier


class ScaleTokens:
    def __init__(self, feat, band, free, obs, vfrac, support, cell, ext):
        """feat [1, C, X, Y, Z]; band / free / obs / vfrac [1, 1, X, Y, Z]; support bool [1, 1, X, Y, Z]; cell: model-unit
        size of one cell; ext [3]: extent of the 4x grid (model units)."""
        idx = torch.nonzero(support[0, 0], as_tuple=False)
        g = lambda t: t[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].t().float()  # noqa: E731
        self.feat = torch.cat([g(feat), g(band), g(free), g(obs), g(vfrac)], 1)             # [N, C + 4]
        self.pos = (idx.float() + 0.5) * float(cell)
        self.pos_n = self.pos / ext.to(self.pos.device).float()
        self.n = idx.shape[0]

    def select(self, o, d, tcap, budget, margin_deg):
        """The T1 rule (cross_attention.SceneMemory.select): cell centre inside the view cone (half angle = half diagonal
        field of view + margin) and within the largest ray cap; more than `budget` -> uniform stride over the
        distance-sorted set."""
        if self.n == 0:
            return torch.zeros(0, dtype=torch.long, device=o.device)
        axis = F.normalize(d.float().mean(0), dim=0)
        rel = self.pos - o.float()[None]
        dist = rel.norm(dim=1)
        cosang = (rel @ axis) / dist.clamp(min=1e-6)
        half = math.radians(FOV_DEG / 2.0 * math.sqrt(2.0) + margin_deg)
        keep = (cosang >= math.cos(half)) & (dist <= float(tcap.max()))
        idx = torch.nonzero(keep, as_tuple=False)[:, 0]
        if idx.numel() > budget:
            order = idx[torch.argsort(dist[idx])]
            pick = torch.linspace(0, order.numel() - 1, budget, device=o.device).round().long()
            idx = order[pick]
        return idx


def f8_tokens(h):
    one = torch.ones_like(h["band8"], dtype=torch.float32)
    sup = ((h["band8"] > 0) | (h["free8"] > 0)) & (h["valid8"] > 0)
    cell = float(h["ext"][0]) / h["mu"].shape[2]
    return ScaleTokens(h["mu"], h["band8"], h["free8"], one, one, sup, cell, h["ext"])


def f16_tokens(h):
    from seen2scene.v3.xa.vae_pyramid import pool16
    mu16, band16, free16, obs16, vfrac16 = pool16(h["mu"], h["valid8"], h["band8"], h["free8"])
    cell = 2.0 * float(h["ext"][0]) / h["mu"].shape[2]
    return ScaleTokens(mu16, band16, free16, obs16, vfrac16, obs16 > 0, cell, h["ext"])


def global_tokens(scales, views, margin_deg, n_freq):
    """scales: [(ScaleTokens, budget, scale id)]; views: [(o [3], d [R, 3], tcap [R])] -> G [V, sum budget, C + 4 +
    6 n_freq + 2] float32 (zero rows are masked: last entry 0)."""
    V = len(views)
    width = scales[0][0].feat.shape[1] + 6 * n_freq + 2
    dev = scales[0][0].feat.device
    G = torch.zeros(V, sum(b for _, b, _ in scales), width, device=dev, dtype=torch.float32)
    for i, (o, d, tcap) in enumerate(views):
        s0 = 0
        for tok, budget, sid in scales:
            idx = tok.select(o, d, tcap, budget, margin_deg)
            n = idx.numel()
            if n:
                row = torch.cat([tok.feat[idx], fourier(2.0 * tok.pos_n[idx] - 1.0, n_freq),
                                 torch.full((n, 1), float(sid), device=dev), torch.ones(n, 1, device=dev)], 1)
                G[i, s0:s0 + n] = row
            s0 += budget
    return G
