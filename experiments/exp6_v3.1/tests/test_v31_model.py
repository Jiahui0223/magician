"""G0-8 of EXP6 V3.1 (CPU, tiny): the V3.1 model is the V3-T0 architecture (no cross-attention; parameter count and the
seed-0 common-module initialisation hash equal to the T0 run manifest), mu / logvar finite, reparameterisation, KL and
reconstruction finite and back-propagating into the encoder, the flow conditioned on mu."""
import json
import os

import torch

from seen2scene import PROJECT_ROOT
from seen2scene.v3 import losses as L
from seen2scene.v3 import model as V3
from seen2scene.v3 import vae as VAE

REPO = str(PROJECT_ROOT)
CFG = json.load(open(os.path.join(REPO, "experiments", "exp6_v3.1", "configs", "model_v3_1.json")))
T0_RUN = os.path.join(REPO, "work_dirs", "exp6_v3", "task0_vae", "runs", "main", "run_none.json")


def test_g0_8_architecture_equals_t0_and_vae_terms():
    assert not CFG["xattn"]["enabled"]
    torch.manual_seed(0)
    m = V3.V3Model(CFG)
    assert m.xa is None and m.unet.xattn is None
    counts = V3.param_counts(m)
    assert counts["total"] == 23539622 and counts["xattn"] == 0, counts
    if os.path.exists(T0_RUN):
        r = json.load(open(T0_RUN))
        assert r["params"] == counts
        assert V3.state_hash(V3.common_state(m)) == r["common_init_sha256"]     # same construction as the formal T0 run
    cfg = json.loads(json.dumps(CFG)); cfg["vae"]["act_ckpt"] = False
    torch.manual_seed(0)
    m = V3.V3Model(cfg).train()
    torch.manual_seed(1)
    x = torch.rand(1, 3, 16, 12, 8); x[:, 0] = x[:, 0] * 2 - 1; x[:, 1] = (x[:, 1] > 0.7).float() * x[:, 1]
    valid = torch.ones(1, 1, 16, 12, 8); valid[..., -2:, :, :] = 0
    x = x * valid
    h = m.encode(x, valid)
    assert torch.isfinite(h["mu"]).all() and torch.isfinite(h["logvar"]).all() and h["mu"].shape == (1, 16, 8, 6, 4)
    st = {"mu": h["mu"], "logvar": h["logvar"], "valid8": h["valid8"], "x4": x, "valid4": valid}
    rec, diag = L.recon_terms(m, [st], autocast=False)
    kl, per_ch = L.kl_terms([st])
    assert torch.isfinite(rec) and torch.isfinite(kl) and float(kl) >= 0 and per_ch.shape == (16,)
    (rec + 0.02 * kl).backward()
    g = sum(float(p.grad.abs().sum()) for p in m.vae_enc.parameters() if p.grad is not None)
    assert g > 0 and all(torch.isfinite(p.grad).all() for p in m.vae_enc.parameters() if p.grad is not None)
    z = VAE.reparameterize(h["mu"].detach(), h["logvar"].detach())
    assert torch.isfinite(z).all() and z.shape == h["mu"].shape
    with torch.no_grad():
        R = V3.GH * V3.GW
        o = torch.rand(2, 3) * h["ext"] * 0.5 + 0.25 * h["ext"]
        d = torch.nn.functional.normalize(torch.randn(2, R, 3), dim=-1)
        cond = m.view_cond(h, o, d, torch.full((2, R), float(h["ext"].max()) * 0.8), torch.rand(2, 11, R))
        assert cond.shape == (2, 76, 64, 128) and torch.isfinite(cond).all()
