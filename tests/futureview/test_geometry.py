"""EXP5 Task 1.1 unit tests of futureview.geometry / rays on synthetic grids (protocol §2.6 T-P2, T-L1, T-L2 parts that
need no frozen experiment code; the bitwise comparisons with SceneCaster / pose_rays run in task1/scripts/t11_checks.py)."""
import math

import numpy as np
import pytest
import torch

from seen2scene.futureview import geometry as G
from seen2scene.futureview import rays as RY

VS, h = G.VS, G.H_STEP
DEV = "cpu"


def grid(dims, fill=G.UNKNOWN):
    return torch.full(dims, fill, dtype=torch.int8), torch.ones(dims, dtype=torch.bool)


def one_ray(o_vox, d=(1.0, 0.0, 0.0)):
    o = torch.tensor([x * VS for x in o_vox], dtype=torch.float32)
    d = torch.tensor([d], dtype=torch.float32)
    d = d / d.norm(dim=1, keepdim=True)
    return o, d


def run(struct, dom, o, d, tcap_vox=1e3, t_known=float("inf")):
    ext = torch.tensor([n * VS for n in struct.shape], dtype=torch.float32)
    tcap = torch.tensor([tcap_vox * VS], dtype=torch.float32)
    tk = torch.tensor([t_known], dtype=torch.float32)
    return G.project(o, d, tcap, ext, tk, struct, dom), tcap, ext


def expected_unknown(pr, o, d, pred):
    """Reference: the same sample positions, unknown iff pred(voxel index) (float64 voxel index of the float32 point)."""
    ts = G.sample_ts(pr["tmin"], pr["M"])[0]
    keep = (ts <= pr["tmax"][0]) & pr["valid"][0] & (ts < pr["t_known"][0])
    p = o[None] + d[0][None] * ts[:, None]
    ijk = torch.floor(p / VS).long()
    return [bool(k and pred(tuple(q.tolist()))) for k, q in zip(keep.tolist(), ijk)], ts


def test_constants_match_frozen_sensor():
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "experiments", "exp3b_macaronspp_transfer"))
    import m_common as MC
    assert (G.VS, G.TRUNC, G.EMPTY, G.UNKNOWN, G.BAND, G.NEAR_CLIP) == (MC.VS, MC.TRUNC, MC.EMPTY, MC.UNKNOWN, MC.BAND, MC.NEAR_CLIP)
    assert (RY.H, RY.W, RY.M_SHORT, RY.FOV_DEG, RY.S_PROJ) == (MC.H, MC.W, MC.M_SHORT, MC.FOV_DEG, MC.S_PROJ)
    u0, v0 = RY.grid_uv((0.0, 0.0))
    u1, v1 = MC.scoring_pixel_grid()
    assert np.array_equal(u0, u1) and np.array_equal(v0, v1)
    assert G.PATCH_H == MC.PATCH * MC.VS


def test_wall_hides_unknown_behind_known_surface():
    struct, dom = grid((40, 6, 6))
    struct[:10] = G.EMPTY
    struct[20:22] = G.BAND
    o, d = one_ray((0.3, 3.5, 3.5))
    pr, _, _ = run(struct, dom, o, d, t_known=20.7 * VS)
    ref, ts = expected_unknown(pr, o, d, lambda q: 10 <= q[0] < 20)
    assert int(pr["n_u"][0]) == sum(ref) and int(pr["n_seg"][0]) == 1
    first = ref.index(True)
    last = len(ref) - 1 - ref[::-1].index(True)
    assert float(pr["d_lo"][0]) == float(ts[first]) and float(pr["d_hi"][0]) == float(ts[last])
    assert int(pr["cls"][0]) == G.CLS_UNKNOWN
    assert float(pr["d_hi"][0]) < 20 * VS                 # nothing behind the wall


def test_multi_segment_domain_and_range():
    struct, dom = grid((40, 6, 6))
    struct[8:12] = G.EMPTY
    struct[20:] = G.EMPTY
    dom[16:18] = False                                      # out-of-domain strip inside the second UNKNOWN run
    o, d = one_ray((0.0, 3.5, 3.5))
    pr, _, _ = run(struct, dom, o, d)
    ref, _ = expected_unknown(pr, o, d, lambda q: (0 <= q[0] < 8) or (12 <= q[0] < 16) or (18 <= q[0] < 20))
    assert int(pr["n_u"][0]) == sum(ref) and int(pr["n_seg"][0]) == 3
    pr2, _, _ = run(struct, dom, o, d, tcap_vox=5.0)       # range cap inside the first run
    ref2, _ = expected_unknown(pr2, o, d, lambda q: q[0] < 8)
    assert int(pr2["n_u"][0]) == sum(ref2) and float(pr2["d_hi"][0]) <= 5 * VS


def test_camera_outside_box_and_ray_missing_box():
    struct, dom = grid((20, 6, 6))
    o, d = one_ray((-5.0, 3.5, 3.5))
    pr, _, _ = run(struct, dom, o, d)
    assert abs(float(pr["tmin"][0]) - 5 * VS) < 1e-6 and int(pr["n_u"][0]) > 0
    o2, d2 = one_ray((-5.0, 30.0, 3.5))                    # passes beside the box
    pr2, _, _ = run(struct, dom, o2, d2)
    assert not bool(pr2["valid"][0]) and int(pr2["n_u"][0]) == 0 and int(pr2["cls"][0]) == G.CLS_EMPTY


def test_classes_known_and_empty():
    struct, dom = grid((20, 6, 6), G.EMPTY)
    o, d = one_ray((0.5, 3.5, 3.5))
    pr, _, _ = run(struct, dom, o, d, t_known=10 * VS)
    assert int(pr["cls"][0]) == G.CLS_KNOWN
    pr, _, _ = run(struct, dom, o, d)
    assert int(pr["cls"][0]) == G.CLS_EMPTY


def labels_for(struct, dom, o, d, tstar_vox, t_known=float("inf"), k_M=10.0, dz=1.0, tcap_vox=1e3):
    pr, tcap, ext = run(struct, dom, o, d, tcap_vox=tcap_vox, t_known=t_known)
    ts = torch.tensor([tstar_vox * VS if math.isfinite(tstar_vox) else float("inf")], dtype=torch.float32)
    lab = G.encode_labels(o, d, pr, ts, torch.tensor([dz], dtype=torch.float64), k_M, struct, dom)
    return pr, lab


def test_label_categories_and_roundtrip():
    struct, dom = grid((40, 6, 6))
    struct[8:12] = G.EMPTY
    struct[25:27] = G.BAND
    o, d = one_ray((0.0, 3.5, 3.5))
    tk = 25.4 * VS
    # HIT in the second UNKNOWN run -> decode returns the nearest UNKNOWN sample
    pr, lab = labels_for(struct, dom, o, d, 13.26, t_known=tk)
    assert int(lab["cat"][0]) == G.CAT["HIT"] and int(lab["b"][0]) == 1
    t_dec, hu = G.decode(lab["b"].float()[None], lab["y"].float()[None], o, d, pr, struct, dom)
    assert bool(hu[0, 0]) and abs(float(t_dec[0, 0]) - 13.26 * VS) <= h / 2 + 1e-7
    # no surface -> NOHIT, decodes to the projection outcome (known surface here)
    pr, lab = labels_for(struct, dom, o, d, float("inf"), t_known=tk)
    assert int(lab["cat"][0]) == G.CAT["NOHIT"] and int(lab["b"][0]) == -1 and float(lab["y"][0]) == 1.0
    t_dec, hu = G.decode(lab["b"].float()[None], lab["y"].float()[None], o, d, pr, struct, dom)
    assert not bool(hu[0, 0]) and float(t_dec[0, 0]) == float(pr["t_known"][0])
    # surface at / behind the known surface -> KNOWN
    _, lab = labels_for(struct, dom, o, d, 25.0, t_known=tk)
    assert int(lab["cat"][0]) == G.CAT["KNOWN"] and int(lab["b"][0]) == -1
    # surface inside observed EMPTY -> infeasible, no target
    _, lab = labels_for(struct, dom, o, d, 9.6, t_known=tk)
    assert int(lab["cat"][0]) == G.CAT["INF_EMPTY"] and int(lab["b"][0]) == 0
    # surface before the box entry -> BLOCKED; z < 0.5 -> NEAR
    o2, d2 = one_ray((-8.0, 3.5, 3.5))
    _, lab = labels_for(struct, dom, o2, d2, 3.0, t_known=tk + 8 * VS)
    assert int(lab["cat"][0]) == G.CAT["BLOCKED"] and int(lab["b"][0]) == 0
    _, lab = labels_for(struct, dom, o, d, 2.0, t_known=tk, k_M=10.0, dz=1.0)      # z = 0.022 * 10 = 0.22 < 0.5
    assert int(lab["cat"][0]) == G.CAT["NEAR"] and int(lab["b"][0]) == 0
    # sky beyond the range cap -> NOHIT
    _, lab = labels_for(struct, dom, o, d, 30.0, tcap_vox=20.0)
    assert int(lab["cat"][0]) == G.CAT["NOHIT"]


def test_decode_rules_never_violate_known_geometry():
    torch.manual_seed(0)
    dims = (60, 50, 40)
    struct = torch.randint(0, 3, dims, dtype=torch.int8)
    dom = torch.rand(dims) > 0.1
    R = 512
    o = torch.tensor([30.5 * VS, 25.5 * VS, 20.5 * VS])
    d = torch.randn(R, 3)
    d = d / d.norm(dim=1, keepdim=True)
    ext = torch.tensor([n * VS for n in dims], dtype=torch.float32)
    tcap = torch.full((R,), 25 * VS)
    tk = torch.where(torch.rand(R) < 0.5, torch.rand(R) * 25 * VS, torch.full((R,), float("inf")))
    pr = G.project(o, d, tcap, ext, tk, struct, dom, chunk=100)
    K = 16
    b = torch.randn(K, R)
    y = torch.rand(K, R) * 2.2 - 1.1
    t, hu = G.decode(b, y, o, d, pr, struct, dom, chunk=77)
    assert bool((hu <= ((b > 0) & (y < 1))).all())
    p = o[None, None] + d[None] * torch.where(hu, t, torch.zeros_like(t))[..., None]
    ijk = torch.floor(p / VS).long()[hu]
    assert len(ijk) > 0
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    assert bool((struct.reshape(-1)[flat] == G.UNKNOWN).all()) and bool(dom.reshape(-1)[flat].all())
    assert bool((t[hu] < tk[None].expand(K, R)[hu]).all())
    nohit = ~hu
    assert bool((t[nohit] == tk[None].expand(K, R)[nohit]).all())
    # encode -> decode round trip on every HIT label: the nearest UNKNOWN sample (within one voxel) is returned
    tstar = torch.rand(R) * 25 * VS
    lab = G.encode_labels(o, d, pr, tstar, torch.ones(R, dtype=torch.float64), 1e3, struct, dom, chunk=64)
    hit = lab["cat"] == G.CAT["HIT"]
    assert int(hit.sum()) > 20
    t2, hu2 = G.decode(lab["b"].float()[None], lab["y"].float()[None], o, d, pr, struct, dom)
    assert bool(hu2[0][hit].all()) and bool(((t2[0][hit] - tstar[hit]).abs() <= VS + 1e-6).all())
    assert bool((lab["b"][pr["n_u"] == 0] == 0).all())


def test_coarse_grid_fractions():
    struct, dom = grid((16, 10, 16), G.UNKNOWN)
    struct[:8, :8, :8] = G.EMPTY
    struct[8:, :8, 8:] = G.BAND
    dom[:, 8:, :] = False
    c = G.coarse_grid(struct, dom)
    assert list(c.shape) == [3, 4, 4, 2]
    assert int(c[0, 0, 0, 0]) == 255 and int(c[1, 1, 0, 1]) == 255 and int(c[2, 0, 1, 0]) == 0
    assert int(c[2, 0, 0, 0]) == 255


def test_channels_shape_and_range():
    struct, dom = grid((40, 6, 6))
    o, d = one_ray((0.0, 3.5, 3.5))
    pr, tcap, _ = run(struct, dom, o, d, tcap_vox=30.0, t_known=20 * VS)
    ch = G.channels(pr, tcap, d, o)
    assert ch.shape == (G.N_CH, 1)
    assert float(ch[:7].min()) >= 0.0 and float(ch[:7].max()) <= 1.0


def test_label_intrinsics_affine_solution():
    for jit in [(0.0, 0.0), (0.5, 0.5), (-1.7, 1.3)]:
        fx, fy, px, py = RY.label_intrinsics(jit)
        u, v = RY.grid_uv(jit)
        nx = RY.W / RY.M_SHORT - 2 * u / (RY.M_SHORT - 1)
        ny = RY.H / RY.M_SHORT - 2 * v / (RY.M_SHORT - 1)
        j = np.tile(np.arange(RY.GW), RY.GH)
        i = np.repeat(np.arange(RY.GH), RY.GW)
        assert np.allclose(fx * nx / RY.S_PROJ + px, (RY.GW - 1 - 2 * j) / RY.GH, atol=1e-12)
        assert np.allclose(fy * ny / RY.S_PROJ + py, (RY.GH - 1 - 2 * i) / RY.GH, atol=1e-12)


def test_jitter_range_and_determinism():
    js = [RY.jitter_of("pisa", "cg-A-0000", t, a) for t in range(20) for a in range(8)]
    arr = np.asarray(js)
    assert arr.min() >= -2.0 and arr.max() < 2.0
    assert js[3] == RY.jitter_of("pisa", "cg-A-0000", 0, 3)


def test_model_rays_torch_matches_numpy():
    import sys, os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "experiments", "exp3b_macaronspp_transfer"))
    import m_common as MC
    rng = np.random.default_rng(1)
    for _ in range(3):
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        T = rng.normal(size=3) * 5
        jit = tuple(rng.uniform(-2, 2, 2))
        origin, k = rng.normal(size=3), 13.7
        o, d, tc, dz = RY.model_rays_torch(torch.as_tensor(A[None]), torch.as_tensor(T[None]),
                                           torch.as_tensor([jit]), torch.as_tensor(origin), k)
        u, v = RY.grid_uv(jit)
        d_y, dz_n, c = MC.pixel_rays(A, T, u, v)
        assert np.allclose(o[0].numpy(), (MC.to_z_up(c) - origin) / k, atol=1e-12)
        assert np.allclose(d[0].numpy(), MC.to_z_up(d_y), atol=1e-12)
        assert np.allclose(dz[0].numpy(), dz_n, rtol=1e-12)
        assert np.allclose(tc[0].numpy(), MC.SENSOR_RANGE / dz_n / k, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="model test runs on the GPU")
def test_model_shapes_and_mask():
    from seen2scene.futureview import model as M
    cfg = {"encoder": {"in_ch": 4, "ch": [32, 64, 96], "blocks_per_level": 2, "dilation_L2": 2, "groups": 8},
           "ray": {"n_points": 32, "feat_ch": 96, "hidden": 64, "out_ch": 64}, "proj_ch": 11,
           "unet": {"ch": [64, 128, 192, 256], "blocks_per_level": 2, "attn_levels": [3], "heads": 4, "t_emb": 256, "groups": 16},
           "flow": {"x_ch": 2}}
    dev = torch.device("cuda")
    net = M.V2Model(cfg).to(dev)
    coarse = torch.randint(0, 256, (3, 8, 12, 32), dtype=torch.uint8, device=dev)
    h = net.encode(M.encoder_input(coarse))
    assert h[0].shape == (1, 32, 8, 12, 32) and h[1].shape == (1, 64, 4, 6, 16)
    V, R = 2, 7296
    o = torch.tensor([[0.3, 0.4, 1.0], [0.2, 0.5, 1.2]], device=dev)
    d = torch.nn.functional.normalize(torch.randn(V, R, 3, device=dev), dim=-1)
    cond = net.view_cond(h, o, d, torch.full((V, R), 2.0, device=dev), torch.rand(V, 11, R, device=dev))
    assert cond.shape == (V, net.cond_ch, 64, 128)
    v = net.velocity(torch.zeros(V, 2, 64, 128, device=dev), torch.rand(V, device=dev), cond)
    assert v.shape == (V, 2, 64, 128)
    print("params", {k: sum(p.numel() for p in getattr(net, k).parameters()) for k in ("encoder", "rays", "unet")})
