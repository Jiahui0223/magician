"""G0-8: warm start = exact copy of the V4 EMA@40k encoder and flow U-Net (sha-checked checkpoint, never written); the
old RayEncoder / RayXAttn are not part of V5 (teacher only); the teacher's H_old equals the 64 ray channels the V4
model gives the flow."""
import hashlib
import os
import sys

import torch
import torch.nn.functional as F

from _common import M, V4CK, V5, run, tiny_state, v5_cfg

V4SHA = "1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_warm_start_exact_copy():
    mt = os.path.getmtime(V4CK)
    assert sha(V4CK) == V4SHA
    ck = torch.load(V4CK, map_location="cpu", weights_only=False)
    m = V5.V5Model(v5_cfg())
    rep = V5.load_v4_warmstart(m, ck["ema"])
    sd = m.state_dict()
    for k, v in ck["ema"].items():
        if k.startswith(("encoder.", "unet.")):
            assert torch.equal(sd[k], v), k
    assert set(rep["not_used_from_v4"]) == {"rays", "ray_xattn"}
    assert all(not k.startswith(("rays.", "ray_xattn.")) for k in sd)
    assert os.path.getmtime(V4CK) == mt and sha(V4CK) == V4SHA


def test_teacher_ray_context_equals_v4_cond_channels():
    ck = torch.load(V4CK, map_location="cpu", weights_only=False)
    t = M.V2Model(ck["model_cfg"]); t.load_state_dict(ck["ema"]); t.eval()
    coarse = tiny_state()
    cin = M.encoder_input(coarse)
    V, R = 1, M.GH * M.GW
    ext = torch.tensor(coarse.shape[1:], dtype=torch.float32) * V5.S0
    o = (ext * 0.4)[None]; d = F.normalize(torch.randn(V, R, 3), dim=-1); tc = torch.full((V, R), 1.5); ch = torch.rand(V, 11, R)
    with torch.no_grad():
        h = t.encode(cin)
        cond = t.view_cond(h, o, d, tc, ch)
        Hold = t._ray_context_v4(h[0], h[1], h[2], o[:, None].expand(V, R, 3).reshape(-1, 3), d.reshape(-1, 3), tc.reshape(-1))
    assert torch.allclose(cond[0, 11:75, :, :M.GW].reshape(64, -1).t(), Hold, atol=1e-5)


if __name__ == "__main__":
    sys.exit(run([test_warm_start_exact_copy, test_teacher_ray_context_equals_v4_cond_channels]))
