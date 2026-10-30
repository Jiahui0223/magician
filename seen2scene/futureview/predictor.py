"""Batched V2 inference for one observed state (EXP5 Task 1, architecture §4-§7): encode once, condition every candidate
view, draw K samples with shared noise across candidates (common random numbers), return (b, y) per ray.
"""
import torch

from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M


class V2Predictor:
    def __init__(self, model, device, steps=10, max_views=64, autocast=True):
        self.model, self.dev, self.steps, self.max_views, self.autocast = model.eval(), device, steps, max_views, autocast

    @torch.no_grad()
    def encode(self, coarse_u8):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.model.encode(M.encoder_input(coarse_u8.to(self.dev)))

    @torch.no_grad()
    def cond(self, h, views):
        """views: [(o [3], d [R, 3], tcap [R], pr)] -> cond [V, C, 64, 128], m_unk [V, 64, 128]."""
        o = torch.stack([v[0] for v in views]).float()
        d = torch.stack([v[1] for v in views]).float()
        tc = torch.stack([v[2] for v in views]).float()
        ch = torch.stack([G.channels(v[3], v[2], v[1], v[0]) for v in views])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            c = self.model.view_cond(h, o, d, tc, ch)
        m_unk = M.pad_rays(torch.stack([v[3]["cls"] == G.CLS_UNKNOWN for v in views]))
        return c, m_unk

    @torch.no_grad()
    def sample(self, cond, m_unk, eps, return_features=False):
        """cond [V, ...], eps [K, 2, 64, 128] shared by all views -> b, y [V, K, R] float32 (+ features [V, K, C, 64, 128])."""
        V, K = cond.shape[0], eps.shape[0]
        idx = torch.arange(V * K, device=cond.device)
        vi, ki = idx // K, idx % K
        xs, fs = [], []
        for s in range(0, V * K, self.max_views):
            sl = slice(s, min(V * K, s + self.max_views))
            r = M.sample(self.model, cond[vi[sl]], m_unk[vi[sl]], eps[ki[sl]], self.steps, self.autocast, return_features)
            if return_features:
                xs.append(r[0]); fs.append(r[1].float())
            else:
                xs.append(r)
        x = M.unpad_rays(torch.cat(xs)).reshape(V, K, 2, -1)
        if return_features:
            return x[:, :, 0], x[:, :, 1], torch.cat(fs).reshape(V, K, *fs[0].shape[1:])
        return x[:, :, 0], x[:, :, 1]


def noise(device, K, seed_fn):
    """eps [K, 2, 64, 128] from per-k seeds seed_fn(k)."""
    out = []
    for k in range(K):
        g = torch.Generator(device=device)
        g.manual_seed(seed_fn(k) % (2 ** 63))
        out.append(torch.randn(2, M.GH, M.GW_PAD, generator=g, device=device))
    return torch.stack(out)
