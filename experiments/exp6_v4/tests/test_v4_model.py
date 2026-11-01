"""G0-4 / G0-5 / G0-6 / G0-7 / G0-11a of EXP6 V4 (CPU, tiny grids): shapes and finiteness of the V4 forward, non-zero
finite gradients of every ray_xattn projection, camera conditioning (same P1, different legal camera -> different
sample coordinates, attention and condition maps), visibility mask (UNKNOWN points are legal tokens; rays without any
in-grid point give a zero residual and no NaN) and the bitwise V2 replay with the flag off (parameter names equal to
the V2 Original checkpoint; cond / sample equal to the pre-V4 module from git HEAD)."""
import importlib.util
import json
import os
import subprocess
import tempfile

import pytest
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import model as M

REPO = str(PROJECT_ROOT)
CFG = json.load(open(os.path.join(REPO, "experiments", "exp6_v4", "configs", "model_v4.json")))
V2CFG = json.load(open(os.path.join(REPO, "experiments", "exp5_future_view", "task1", "configs", "model_v1.json")))
V2CKPT = os.path.join(REPO, "work_dirs", "weights", "W20_exp5_v2", "W20-1_V2_original_ema60k", "ema_060000.pt")


def make(cfg, seed=0):
    torch.manual_seed(seed)
    return M.V2Model(cfg)


def synth(dev, V=2, shape=(12, 16, 32), seed=1):
    torch.manual_seed(seed)
    coarse = (torch.rand(3, *shape, device=dev) * 255).to(torch.uint8)
    cin = M.encoder_input(coarse)
    R = M.GH * M.GW
    ext = torch.tensor([s * M.BLOCK * M.VS for s in shape])
    o = torch.rand(V, 3, device=dev) * 0.4 * ext + 0.3 * ext
    d = torch.nn.functional.normalize(torch.randn(V, R, 3, device=dev), dim=-1)
    tcap = torch.full((V, R), float(ext.max()) * 0.8, device=dev)
    ch = torch.rand(V, 11, R, device=dev)
    return cin, o, d, tcap, ch, ext


def test_g0_4_shapes_finite_and_params():
    m = make(CFG).eval()
    cin, o, d, tcap, ch, _ = synth("cpu")
    with torch.no_grad():
        h = m.encode(cin)
        cond = m.view_cond(h, o, d, tcap, ch)
        v = m.velocity(torch.randn(2, 2, M.GH, M.GW_PAD), torch.rand(2), cond)
    assert cond.shape == (2, m.cond_ch, M.GH, M.GW_PAD) and m.cond_ch == 11 + 64 + 1
    assert v.shape == (2, 2, M.GH, M.GW_PAD) and torch.isfinite(cond).all() and torch.isfinite(v).all()
    n = {k: sum(p.numel() for p in getattr(m, k).parameters()) for k in ("encoder", "rays", "unet", "ray_xattn")}
    assert n["encoder"] == 2412000 and n["rays"] == 76224 and n["unet"] == 22333890, n      # V2 modules unchanged
    assert 0 < n["ray_xattn"] < 50000, n                                                   # lightweight


def test_g0_5_xattn_gradients_nonzero_finite():
    m = make(CFG).train()
    m.rays_ckpt = False
    cin, o, d, tcap, ch, _ = synth("cpu")
    with torch.no_grad():                               # the V2 output conv is zero-initialised: perturb it (one optimizer step)
        m.unet.out[-1].weight.normal_(0, 0.02)
    m.zero_grad()
    v = m.velocity(torch.randn(2, 2, M.GH, M.GW_PAD), torch.rand(2), m.view_cond(m.encode(cin), o, d, tcap, ch))
    v.pow(2).mean().backward()
    for k, p in m.ray_xattn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0, k
    for k in ("encoder", "rays", "unet"):
        g = sum(float(p.grad.abs().sum()) for p in getattr(m, k).parameters() if p.grad is not None)
        assert g > 0, k


def test_g0_6_camera_conditioning():
    m = make(CFG).eval()
    cin, o, d, tcap, ch, ext = synth("cpu", V=2)
    R = M.GH * M.GW
    with torch.no_grad():
        h = m.encode(cin)
        F0, F1, dom = h
        # sample coordinates follow the extrinsics
        p0 = m.rays.points(o[0].expand(R, 3), d[0], tcap[0])
        p1 = m.rays.points(o[1].expand(R, 3), d[0], tcap[0])
        assert torch.allclose(p1 - p0, (o[1] - o[0]).expand(R, 32, 3))
        att = {}
        for i in range(2):
            f, inside = m.rays.point_features(F0, F1, dom, o[i].expand(R, 3), d[i], tcap[i])
            rf = m.rays.head(f)
            m.ray_xattn(f, inside, rf, o[i].expand(R, 3), d[i], tcap[i], ext, keep_attn=True)
            att[i] = m.ray_xattn.last_attn.clone()
        assert torch.allclose(att[0].sum(-1)[inside.any(1)], torch.ones(1), atol=1e-5)
        assert float((att[0] - att[1]).abs().mean()) > 1e-4                      # attention depends on the camera
        c = m.view_cond(h, o, d, tcap, ch)                                       # same P1, two cameras -> different cond
        assert float((c[0, 11:75] - c[1, 11:75]).abs().mean()) > 1e-4
        # same rays, origin shifted: the X-Attn residual changes (the query encodes the origin and the tokens move)
        f, inside = m.rays.point_features(F0, F1, dom, o[0].expand(R, 3), d[0], tcap[0])
        r0 = m.ray_xattn(f, inside, m.rays.head(f), o[0].expand(R, 3), d[0], tcap[0], ext)
        oo = o[0].expand(R, 3) + 0.05 * ext
        f2, inside2 = m.rays.point_features(F0, F1, dom, oo, d[0], tcap[0])
        r1 = m.ray_xattn(f2, inside2, m.rays.head(f2), oo, d[0], tcap[0], ext)
        assert float((r0 - r1).abs().mean()) > 0


def test_g0_7_visibility_mask_and_unknown_tokens():
    m = make(CFG).eval()
    cin, o, d, tcap, ch, ext = synth("cpu", V=1)
    R = M.GH * M.GW
    with torch.no_grad():
        F0, F1, dom = m.encode(cin)
        far = torch.full((R, 3), 100.0)                                          # every sample point outside the grid
        f, inside = m.rays.point_features(F0, F1, dom, far, d[0], tcap[0])
        assert not inside.any()
        r = m.ray_xattn(f, inside, m.rays.head(f), far, d[0], tcap[0], ext, keep_attn=True)
        assert torch.isfinite(r).all() and float(r.abs().max()) == 0 and float(m.ray_xattn.last_attn.abs().max()) == 0
        # the mask is geometric only: a grid that is entirely UNKNOWN / out-of-domain keeps every in-grid point as a token
        cin0 = M.encoder_input(torch.zeros(3, 12, 16, 32, dtype=torch.uint8))
        F0z, F1z, domz = m.encode(cin0)
        fz, insz = m.rays.point_features(F0z, F1z, domz, o[0].expand(R, 3), d[0], tcap[0])
        assert insz.any() and float(fz[:, -1].abs().sum()) == 0                  # in-domain flag channel is 0, points still inside
        rz = m.ray_xattn(fz, insz, m.rays.head(fz), o[0].expand(R, 3), d[0], tcap[0], ext)
        assert torch.isfinite(rz).all()
        # whole-model forward with partly masked rays is finite
        cond = m.view_cond((F0, F1, dom), far[None, 0][:, None].expand(1, 1, 3)[:, 0], d[:1], tcap[:1], ch[:1])
        v = m.velocity(torch.randn(1, 2, M.GH, M.GW_PAD), torch.rand(1), cond)
        assert torch.isfinite(v).all()


@pytest.mark.skipif(not os.path.exists(V2CKPT), reason="V2 Original checkpoint not present")
def test_g0_11a_v2_bitwise_replay_with_flag_off():
    ck = torch.load(V2CKPT, map_location="cpu", weights_only=False)
    new = M.V2Model(ck["model_cfg"]).eval()
    assert new.ray_xattn is None and list(new.state_dict()) == list(ck["ema"])
    new.load_state_dict(ck["ema"])
    src = subprocess.check_output(["git", "show", "HEAD:seen2scene/futureview/model.py"], cwd=REPO).decode()
    if "class RayXAttn" in src:                                                     # HEAD already carries V4: compare with the
        src = subprocess.check_output(["git", "show", "ba60f50:seen2scene/futureview/model.py"], cwd=REPO).decode()  # pre-V4 commit
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model_old.py")
        open(p, "w").write(src)
        spec = importlib.util.spec_from_file_location("model_old", p)
        MO = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MO)
    old = MO.V2Model(ck["model_cfg"]).eval()
    old.load_state_dict(ck["ema"])
    cin, o, d, tcap, ch, _ = synth("cpu", V=2)
    m_unk = torch.rand(2, M.GH, M.GW_PAD) > 0.5
    eps = torch.randn(2, 2, M.GH, M.GW_PAD)
    with torch.no_grad():
        co, cn = old.view_cond(old.encode(cin), o, d, tcap, ch), new.view_cond(new.encode(cin), o, d, tcap, ch)
        assert torch.equal(co, cn)
        assert torch.equal(MO.sample(old, co, m_unk, eps, autocast=False), M.sample(new, cn, m_unk, eps, autocast=False))
