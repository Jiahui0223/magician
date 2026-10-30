"""G0 model tests: shapes of encode / view_cond / velocity / decode, T0 vs T1 common-module initialisation under the
same seed, zero-initialised cross-attention output, gradient flow of FM / recon / KL into the encoder (CPU, tiny)."""
import json
import os

import pytest
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.v3 import model as V3
from seen2scene.v3 import partial_tsdf as PT
from seen2scene.v3 import vae as VAE

CFG = json.load(open(os.path.join(PROJECT_ROOT, "experiments", "exp6_v3", "configs", "model_v3.json")))


def cfg(enabled):
    c = json.loads(json.dumps(CFG))
    c["xattn"]["enabled"] = enabled
    c["vae"]["act_ckpt"] = False
    return c


def make(enabled, seed=0):
    torch.manual_seed(seed)
    return V3.V3Model(cfg(enabled))


def synth_state(dev, shape4=(16, 12, 8)):
    torch.manual_seed(1)
    x = torch.rand(1, 3, *shape4, device=dev)
    x[:, 0] = x[:, 0] * 2 - 1
    x[:, 1] = (x[:, 1] > 0.7).float() * x[:, 1]
    valid = torch.ones(1, 1, *shape4, device=dev)
    valid[..., -2:, :, :] = 0
    return x * valid, valid


def synth_views(V, dev, ext):
    R = V3.GH * V3.GW
    o = torch.rand(V, 3, device=dev) * ext.to(dev) * 0.5 + 0.25 * ext.to(dev)
    d = torch.nn.functional.normalize(torch.randn(V, R, 3, device=dev) * torch.tensor([1.0, 1.0, 0.3], device=dev), dim=-1)
    tcap = torch.full((V, R), float(ext.max()) * 0.8, device=dev)
    ch = torch.rand(V, 11, R, device=dev)
    return o, d, tcap, ch


@pytest.mark.parametrize("enabled", [False, True])
def test_shapes_and_grad(enabled):
    dev = torch.device("cpu")
    m = make(enabled).train()
    x, valid = synth_state(dev)
    h = m.encode(x, valid)
    assert h["mu"].shape == (1, 16, 8, 6, 4) and h["logvar"].shape == h["mu"].shape and h["valid8"].shape == (1, 1, 8, 6, 4)
    o, d, tcap, ch = synth_views(3, dev, h["ext"])
    out = m.view_cond(h, o, d, tcap, ch)
    cond, mem = (out, None) if not enabled else out
    assert cond.shape == (3, m.cond_ch, 64, 128)
    if enabled:
        tok, mask, qpe = mem
        assert tok.shape[:2] == (3, 512) and mask.shape == (3, 512) and qpe.shape == (3, 128, 12 * 8)
        assert mask.any(), "frustum selection found no observed tokens"
    xt = torch.randn(3, 2, 64, 128, device=dev)
    v = m.velocity(xt, torch.rand(3, device=dev), cond, mem)
    assert v.shape == (3, 2, 64, 128)
    rec = m.decode(VAE.reparameterize(h["mu"], h["logvar"]))
    assert rec.shape == (1, 4, 16, 12, 8)
    kl, per_ch = VAE.kl_per_unit(h["mu"], h["logvar"], h["valid8"])
    assert float(kl) >= -1e-6 and per_ch.shape == (16,)
    # gradients: FM -> encoder (through mu), recon -> encoder (mu and logvar), KL -> encoder (mu and logvar).
    # The flow U-Net's output conv is zero-initialised (velocity = 0 at init, as in V2), so the FM path is tested with
    # a perturbed output conv (what one optimizer step produces).
    torch.manual_seed(5)
    with torch.no_grad():
        m.unet.out[-1].weight.normal_(0, 0.02)
    if enabled:                                          # the zero W_o of T1 gets the first gradient; check it at init
        m.zero_grad()
        m.velocity(xt, torch.rand(3, device=dev), cond, mem).pow(2).mean().backward(retain_graph=True)
        assert float(m.unet.xattn.wo.weight.grad.abs().sum()) > 0 and float(m.unet.xattn.wq.weight.grad.abs().sum()) == 0
        with torch.no_grad():                            # after a step W_o != 0 and Q / K / V receive gradients
            m.unet.xattn.wo.weight.normal_(0, 0.02)
    v = m.velocity(xt, torch.rand(3, device=dev), cond, mem)
    for name, loss in (("fm", (v ** 2).mean()), ("rec", rec.abs().mean()), ("kl", kl)):
        m.zero_grad()
        loss.backward(retain_graph=True)
        gw = m.vae_enc.head[-1].weight.grad
        assert gw is not None and float(gw.abs().sum()) > 0, name
        g_mu, g_lv = gw[:16].abs().sum(), gw[16:].abs().sum()
        if name == "fm":
            assert float(g_mu) > 0 and float(g_lv) == 0, "FM must reach mu only"
        else:
            assert float(g_mu) > 0 and float(g_lv) > 0, name
        if enabled and name == "fm":
            assert m.unet.xattn.wq.weight.grad is not None and float(m.unet.xattn.wq.weight.grad.abs().sum()) > 0


def test_common_init_identical_and_xattn_zero_out():
    t0, t1 = make(False, 7), make(True, 7)
    assert V3.state_hash(V3.common_state(t0)) == V3.state_hash(V3.common_state(t1))
    assert V3.state_hash(V3.common_state(make(False, 8))) != V3.state_hash(V3.common_state(t0))
    assert float(t1.unet.xattn.wo.weight.abs().sum()) == 0 and float(t1.unet.xattn.wq.weight.abs().sum()) > 0
    pc0, pc1 = V3.param_counts(t0), V3.param_counts(t1)
    assert pc0["xattn"] == 0 and pc1["xattn"] > 0 and pc1["total"] - pc0["total"] == pc1["xattn"]
    # T1 at initialisation = T0 (zero output projection): same velocity on the same input
    dev = torch.device("cpu")
    x, valid = synth_state(dev)
    with torch.no_grad():                               # perturb the zero output conv identically (non-trivial velocities)
        w = torch.randn_like(t0.unet.out[-1].weight) * 0.02
        t0.unet.out[-1].weight.copy_(w); t1.unet.out[-1].weight.copy_(w)
        h0, h1 = t0.eval().encode(x, valid), t1.eval().encode(x, valid)
        o, d, tcap, ch = synth_views(2, dev, h0["ext"])
        c0 = t0.view_cond(h0, o, d, tcap, ch)
        c1, mem = t1.view_cond(h1, o, d, tcap, ch)
        xt = torch.randn(2, 2, 64, 128)
        tau = torch.rand(2)
        assert torch.allclose(t0.velocity(xt, tau, c0), t1.velocity(xt, tau, c1, mem), atol=1e-6)


def test_query_depends_on_camera():
    t1 = make(True, 3).eval()
    x, valid = synth_state(torch.device("cpu"))
    with torch.no_grad():
        h = t1.encode(x, valid)
        o, d, tcap, ch = synth_views(2, torch.device("cpu"), h["ext"])
        _, (tok, mask, qpe) = t1.view_cond(h, o, d, tcap, ch)
    assert not torch.allclose(qpe[0], qpe[1]) and not torch.equal(mask[0], mask[1]) or not torch.allclose(tok[0], tok[1])
    assert tok.shape[-1] == 16 + 2 + 6 * 8
