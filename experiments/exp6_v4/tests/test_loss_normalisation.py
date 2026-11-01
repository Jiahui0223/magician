"""G0-3 (DDP / accumulation loss normalisation): the V2 rule 'sum of squared velocity errors over the GLOBAL labelled
rays / (2 x global ray count)' gives the same loss (and gradient) whether a 64-view batch is processed by one process,
by 4 ranks (sum of per-rank error sums / global count) or as accumulated micro-batches. CPU, tiny synthetic batch."""
import json
import os
import sys

import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import model as M

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task1", "scripts"))
import v2_data as VD  # noqa: E402

CFG = json.load(open(os.path.join(REPO, "experiments", "exp6_v4", "configs", "model_v4.json")))


def batch(V, seed):
    g = torch.Generator().manual_seed(seed)
    m_unk = torch.rand(V, M.GH, M.GW_PAD, generator=g) > 0.4
    b = torch.where(torch.rand(V, M.GH, M.GW_PAD, generator=g) > 0.3, torch.sign(torch.randn(V, M.GH, M.GW_PAD, generator=g)), torch.zeros(1))
    return {"cond": torch.randn(V, 76, M.GH, M.GW_PAD, generator=g), "m_unk": m_unk, "lmask": m_unk & (b != 0),
            "x1": torch.stack([torch.where(b == 0, torch.full_like(b, -1.0), b), torch.rand(V, M.GH, M.GW_PAD, generator=g) * 2 - 1], 1)}


def test_global_normalisation_equals_sharded_and_accumulated():
    torch.manual_seed(0)
    m = M.V2Model(CFG).train()
    with torch.no_grad():
        m.unet.out[-1].weight.normal_(0, 0.02)
    bt = batch(8, 1)
    eps, tau = torch.randn(8, 2, M.GH, M.GW_PAD), torch.rand(8)
    esum, nray = VD.fm_sum(m, bt, eps, tau, autocast=False)
    whole = esum / (2 * nray)
    m.zero_grad(); whole.backward()
    g_whole = torch.cat([p.grad.flatten() for p in m.unet.parameters()]).clone()
    parts = []
    for r in range(4):
        sl = slice(2 * r, 2 * r + 2)
        e, n = VD.fm_sum(m, {k: v[sl] for k, v in bt.items()}, eps[sl], tau[sl], autocast=False)
        parts.append((e, n))
    n_g = sum(n for _, n in parts)
    assert float(n_g) == float(nray)
    sharded = sum(e for e, _ in parts) / (2 * n_g)
    assert torch.allclose(sharded, whole, rtol=1e-5, atol=1e-7)
    m.zero_grad()
    for e, _ in parts:                                   # per-rank loss e / (2 n_g); gradients summed over ranks
        (e / (2 * n_g)).backward()
    g_sharded = torch.cat([p.grad.flatten() for p in m.unet.parameters()])
    assert torch.allclose(g_sharded, g_whole, rtol=1e-4, atol=1e-6)
