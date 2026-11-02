"""G0-5 / G0-7: V5 forward never calls grid_sample (no old 32-point sampling), produces the V4 cond layout, is per-ray
(chunk invariant, subset z = full-pass z), and every new module plus the flow receives gradient; the belief summary
reaches the flow (changing the surface head changes the cond)."""
import sys

import torch
import torch.nn.functional as F

from _common import M, V5, run, tiny_state, v5_cfg


def setup(seed=0):
    torch.manual_seed(seed)
    m = V5.V5Model(v5_cfg())
    coarse = tiny_state()
    cin = M.encoder_input(coarse)
    V, R = 2, M.GH * M.GW
    ext = torch.tensor(coarse.shape[1:], dtype=torch.float32) * V5.S0
    o = torch.stack([ext * 0.3, ext * 0.55])
    d = F.normalize(torch.randn(V, R, 3), dim=-1)
    return m, cin, o, d, torch.full((V, R), 1.6), torch.rand(V, 11, R)


def test_no_grid_sample_and_layout():
    m, cin, o, d, tc, ch = setup()
    orig = F.grid_sample
    calls = []
    F.grid_sample = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        m.eval()
        with torch.no_grad():
            cond = m.view_cond(m.encode(cin), o, d, tc, ch)
    finally:
        F.grid_sample = orig
    assert not calls, "grid_sample called in V5 inference"
    assert not hasattr(m, "rays") and not hasattr(m, "ray_xattn")
    assert cond.shape == (2, 76, M.GH, M.GW_PAD)
    assert torch.equal(cond[:, :11, :, :M.GW].reshape(2, 11, -1), ch)
    assert bool((cond[:, -1, :, :M.GW] == 1).all()) and bool((cond[:, :, :, M.GW:] == 0).all())


def test_chunk_and_subset_invariance():
    m, cin, o, d, tc, ch = setup()
    m.eval()
    with torch.no_grad():
        h = m.encode(cin)
        a = m.view_cond(h, o, d, tc, ch)
        m.chunk = 1000
        b = m.view_cond(h, o, d, tc, ch)
        oo = o[:, None].expand(2, d.shape[1], 3).reshape(-1, 3)
        H, aux, mem = m.ray_features(h, oo, d.reshape(-1, 3), tc.reshape(-1), ch.transpose(1, 2).reshape(-1, 11), return_aux=True)
        sel = torch.tensor([3, 5000, 9000, 14000])
        Hs, auxs = m.dec(mem, oo[sel], d.reshape(-1, 3)[sel], tc.reshape(-1)[sel], ch.transpose(1, 2).reshape(-1, 11)[sel], return_aux=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(Hs, H[sel], atol=1e-5) and torch.allclose(auxs["z"], aux["z"][sel], atol=1e-5)


def test_all_paths_get_gradient():
    m, cin, o, d, tc, ch = setup()
    m.train(); m.dec_ckpt = True
    with torch.no_grad():
        m.unet.out[-1].weight.normal_(0, 0.02)
    h = m.encode(cin)
    cond = m.view_cond(h, o, d, tc, ch)
    v = m.velocity(torch.randn(2, 2, M.GH, M.GW_PAD), torch.rand(2), cond)
    mem = V5.Memory(h[0], h[1], h[3])
    _, aux = m.dec(mem, o[:1].expand(50, 3), d[0, :50], tc[0, :50], ch[0, :, :50].t(), return_aux=True)
    ls = m.dec.surface_logits(aux["z"], torch.rand(50, 4) * 1.5, tc[0, :50])
    (v.pow(2).mean() + torch.sigmoid(ls).mean()).backward()
    none = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0]
    assert not none, none[:10]


def test_belief_reaches_flow_condition():
    m, cin, o, d, tc, ch = setup()
    m.eval()
    with torch.no_grad():
        h = m.encode(cin)
        a = m.view_cond(h, o, d, tc, ch)
        m.dec.s_out[-1].bias += 3.0                     # change ONLY the surface head
        b = m.view_cond(h, o, d, tc, ch)
    diff = (a - b)[:, 11:75].abs().mean()
    assert float(diff) > 1e-4, float(diff)
    assert torch.equal(a[:, :11], b[:, :11])


if __name__ == "__main__":
    sys.exit(run([test_no_grid_sample_and_layout, test_chunk_and_subset_invariance, test_all_paths_get_gradient, test_belief_reaches_flow_condition]))
