"""Planner-side inference of V3 (EXP6): the V2Predictor interface (seen2scene.futureview.predictor) for a V3 model.

encode_fusion(fu)        live P1 fusion (sum / cnt / empty / dims / dom on the GPU) -> 4x aggregation -> h (mu, ...)
cond(h, views)           views [(o, d, tcap, pr)] -> (cond pack, m_unk); the pack carries the T1 memory tokens
sample(pack, m_unk, eps) K samples with shared noise across candidates -> (b, y) per ray  (futureview.predictor rules)
The 4x aggregation is the same function the training cache used (partial_tsdf.aggregate), so train and deploy inputs
are identical by construction; mu is deterministic for a given map (the decoder is not instantiated here).
"""
import torch

from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.v3 import model as V3
from seen2scene.v3 import partial_tsdf as PT


class V3Predictor:
    def __init__(self, model, device, steps=10, max_views=64, autocast=True):
        self.model, self.dev, self.steps, self.max_views, self.autocast = model.eval(), device, steps, max_views, autocast

    @torch.no_grad()
    def encode_grid(self, grid):
        """grid: dict (sdf / band / free / dom, numpy or torch) -> h."""
        x, valid = PT.unpack(grid, self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.model.encode(x, valid)

    @torch.no_grad()
    def encode(self, packed):
        """Planner interface: the uint8 [4, X4, Y4, Z4] tensor of partial_tsdf.pack_grid (Geometry.coarse() of the V3
        planner patch) -> h."""
        x, valid = PT.unpack_packed(packed.to(self.dev), self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.model.encode(x, valid)

    @torch.no_grad()
    def encode_fusion(self, fu):
        return self.encode_grid(PT.aggregate_fusion(fu))

    @torch.no_grad()
    def cond(self, h, views):
        """views: [(o [3], d [R, 3], tcap [R], pr)] -> (pack, m_unk [V, 64, 128]); pack = (cond [V, C, 64, 128], mem or None)."""
        o = torch.stack([v[0] for v in views]).float()
        d = torch.stack([v[1] for v in views]).float()
        tc = torch.stack([v[2] for v in views]).float()
        ch = torch.stack([G.channels(v[3], v[2], v[1], v[0]) for v in views])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            out = self.model.view_cond(h, o, d, tc, ch)
        pack = out if self.model.xa else (out, None)
        m_unk = M.pad_rays(torch.stack([v[3]["cls"] == G.CLS_UNKNOWN for v in views]))
        return pack, m_unk

    @torch.no_grad()
    def sample(self, pack, m_unk, eps, return_features=False):
        """pack from cond(); eps [K, 2, 64, 128] shared by all views -> b, y [V, K, R] float32 (+ features)."""
        cond, mem = pack
        V, K = cond.shape[0], eps.shape[0]
        idx = torch.arange(V * K, device=cond.device)
        vi, ki = idx // K, idx % K
        xs, fs = [], []
        for s in range(0, V * K, self.max_views):
            sl = slice(s, min(V * K, s + self.max_views))
            r = V3.sample(self.model, cond[vi[sl]], m_unk[vi[sl]], eps[ki[sl]], V3.index_mem(mem, vi[sl]), self.steps, self.autocast, return_features)
            if return_features:
                xs.append(r[0]); fs.append(r[1].float())
            else:
                xs.append(r)
        x = M.unpad_rays(torch.cat(xs)).reshape(V, K, 2, -1)
        if return_features:
            return x[:, :, 0], x[:, :, 1], torch.cat(fs).reshape(V, K, *fs[0].shape[1:])
        return x[:, :, 0], x[:, :, 1]


def load_core(ckpt_path, device, steps=10, expect_sha=None):
    """EMA checkpoint of the trainer -> (model, predictor, sha256)."""
    import hashlib
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(f"checkpoint sha {sha} != expected {expect_sha}")
    ck = torch.load(ckpt_path, map_location=device)
    model = V3.V3Model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, V3Predictor(model, device, steps), sha
