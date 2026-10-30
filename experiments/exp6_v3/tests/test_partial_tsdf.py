"""G0 data tests of seen2scene.v3.partial_tsdf: three-state semantics, padding, blob round trip, reconstruction
targets (synthetic fused arrays; CPU)."""
import numpy as np
import torch

from seen2scene.v3 import partial_tsdf as PT


def synth(dims=(12, 8, 16)):
    n = int(np.prod(dims))
    sum_ = torch.zeros(n, dtype=torch.int64)
    cnt = torch.zeros(n, dtype=torch.int32)
    empty = torch.zeros(n, dtype=torch.bool)
    dom = [np.ones(d, bool) for d in dims]
    dom[0][-2:] = False                                   # last two x slabs outside the domain
    idx = lambda i, j, k: (i * dims[1] + j) * dims[2] + k  # noqa: E731
    # wall at x = 4 (band voxels x = 3, 4, 5 with sdf -VS, 0, +VS), free space x < 3, unknown x > 5
    for j in range(dims[1]):
        for k in range(dims[2]):
            for i, s in ((3, -PT.VS), (4, 0.0), (5, PT.VS)):
                sum_[idx(i, j, k)] += int(round(s * PT.FIXED)) * 2
                cnt[idx(i, j, k)] += 2
            for i in range(0, 3):
                empty[idx(i, j, k)] = True
            empty[idx(4, j, k)] = True                   # carved AND band: band has priority
    return sum_, cnt, empty, dims, dom


def test_three_states_and_padding():
    sum_, cnt, empty, dims, dom = synth()
    g = PT.aggregate(sum_, cnt, empty, dims, dom)
    assert tuple(g["sdf"].shape) == (4, 2, 4), g["sdf"].shape          # 12 -> 16 (pad to 8), 8, 16 -> /4
    band, free, domf, sdf = (g[k].float() for k in ("band", "free", "dom", "sdf"))
    # block x in [0,4): voxels x=0,1,2 free, x=3 band -> band 16/64, free 48/64, dom full
    assert torch.allclose(band[0], torch.full_like(band[0], round(16 / 64 * 255))) and torch.allclose(free[0], torch.full_like(free[0], round(48 / 64 * 255)))
    # block x in [4,8): x=4,5 band (x=4 also carved -> band priority), x=6,7 unknown -> band 32/64, free 0
    assert torch.allclose(band[1], torch.full_like(band[1], round(32 / 64 * 255))) and float(free[1].max()) == 0
    # sdf: block 0 band voxels all -VS -> -1/3 TRUNC; block 1: (0 + VS)/2 -> +1/6 TRUNC
    assert torch.allclose(sdf[0] / PT.SDF_Q, torch.full_like(sdf[0], -1 / 3), atol=0.01)
    assert torch.allclose(sdf[1] / PT.SDF_Q, torch.full_like(sdf[1], 1 / 6), atol=0.01)
    # block x in [8,12): unknown, in domain for x = 8, 9 only -> dom 32/64, band = free = 0; padded block [12,16): dom 0
    assert torch.allclose(domf[2], torch.full_like(domf[2], round(32 / 64 * 255))) and float(band[2].max()) == 0 and float(free[2].max()) == 0
    assert float(domf[3].max()) == 0 and float(sdf[2].abs().max()) == 0


def test_blob_roundtrip_and_targets():
    sum_, cnt, empty, dims, dom = synth()
    g = PT.aggregate(sum_, cnt, empty, dims, dom)
    blob, shape = PT.to_blob(g)
    r = PT.from_blob(blob, shape)
    for k in ("sdf", "band", "free", "dom"):
        assert np.array_equal(r[k], g[k].numpy()), k
    x, valid = PT.unpack(r, torch.device("cpu"))
    assert x.shape == (1, 3, 4, 2, 4) and valid.shape == (1, 1, 4, 2, 4)
    assert float(valid[0, 0, 3].max()) == 0 and float(x[0, :, 3].abs().max()) == 0     # padded cells are zero / invalid
    cls, sdf_t, band_m, vmask = PT.recon_targets(x, valid)
    assert (cls[0, 0] == PT.BAND).all() and (cls[0, 1] == PT.BAND).all()
    assert (cls[0, 2] == PT.UNKNOWN).all() and vmask[0, 2].all() and not vmask[0, 3].any()
    assert float(band_m[0, 0, 3].max()) == 0 and float(band_m[0, 0, 0].min()) == 1
    assert abs(float(PT.extent(shape)[0]) - 16 * PT.VS) < 1e-6
