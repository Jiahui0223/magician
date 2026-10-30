"""State-aware 4x aggregation of the fused partial TSDF (P1) into the VAE input.

The fused map (experiments/exp3b m_p1.P1Fusion, D100 variant p1_full.P1FusionFull) holds per fine voxel (VS = 0.011
model units): sum / cnt (fixed-point signed-distance sums of the truncation band, |sdf| < TRUNC = 3 VS), empty (free
space carved from the full depth) and a separable domain mask. Fine-voxel states follow m_p1.input_volume: BAND where
cnt > 0 (priority), EMPTY where empty and not BAND, UNKNOWN otherwise; the UNKNOWN sentinel -TRUNC of input_volume is
NOT a physical distance and is never used here.

Each FACTOR^3 block of fine voxels (FACTOR = 4) becomes one input cell with
    sdf   mean of clamp(band value, +-TRUNC) / TRUNC over the BAND voxels of the block (0 if none)   int8  x 127
    band  BAND voxels / in-domain voxels of the block                                               uint8 x 255
    free  EMPTY voxels / in-domain voxels                                                            uint8 x 255
    dom   in-domain voxels / FACTOR^3                                                                uint8 x 255
Grids are padded at the high end of every axis with out-of-domain voxels to a multiple of PAD_TO = 8 fine voxels (even 4x
dims, so the stride-2 latent of the VAE covers the same extent), and the 4x grid covers
[0, X4 * FACTOR * VS) in model units: the same frame as the fine grid (cell (i, j, k) of the fine grid spans
[i VS, (i + 1) VS)), which the ray encoder samples with align_corners=False.

Reconstruction targets (train-time VAE decoder) per valid cell: class BAND if band > 0, else EMPTY if free >= 0.5,
else UNKNOWN (codes of m_common: EMPTY = 0, UNKNOWN = 1, BAND = 2); sdf regression only on BAND cells.
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F

VS = 0.011
TRUNC_VOX = 3
TRUNC = TRUNC_VOX * VS
FACTOR = 4
SDF_Q = 127.0
EMPTY, UNKNOWN, BAND = 0, 1, 2
FIXED = float(2 ** 40)
CHANNELS = ("sdf", "band", "free")


PAD_TO = 8          # fine voxels: the 4x grid gets even dims, so the stride-2 latent covers exactly the same extent


def padded_dims(dims, factor=FACTOR, pad_to=PAD_TO):
    """4x grid dims of a fine grid padded (high end) to a multiple of pad_to fine voxels."""
    return [int(-(-int(n) // pad_to) * pad_to // factor) for n in dims]


def _pool(v, factor):
    return F.avg_pool3d(v[None, None], factor)[0, 0]


@torch.no_grad()
def aggregate(sum_, cnt, empty, dims, dom1d, factor=FACTOR, trunc=TRUNC, fixed=FIXED):
    """Fused arrays (flat, on one device) -> dict of 4x grids (sdf int8, band / free / dom uint8), torch on the same
    device. dom1d: three bool arrays / tensors (separable domain of the fine grid). Memory-lean: one pooled quantity
    at a time; the indicator pools run in half precision (k / factor^3 is exact in half)."""
    dev = sum_.device
    dims = [int(x) for x in dims]
    d4 = padded_dims(dims, factor)
    pad = [0, d4[2] * factor - dims[2], 0, d4[1] * factor - dims[1], 0, d4[0] * factor - dims[0]]
    dom = [torch.as_tensor(np.asarray(m, bool), device=dev) for m in dom1d]
    dom3 = dom[0][:, None, None] & dom[1][None, :, None] & dom[2][None, None, :]
    has_band = (cnt > 0).reshape(dims)

    ind = torch.half if dev.type == "cuda" else torch.float32          # CPU avg_pool3d has no half kernel

    def P(x):
        return _pool(F.pad(x, pad) if any(pad) else x, factor).float()
    n_dom = P(dom3.to(ind))
    n_band = P((has_band & dom3).to(ind))
    n_free = P((empty.reshape(dims) & ~has_band & dom3).to(ind))
    val = torch.where(has_band, sum_.reshape(dims).float() / fixed, torch.zeros(dims, dtype=torch.float32, device=dev))
    val = (val / cnt.reshape(dims).clamp(min=1).float()).clamp(-trunc, trunc) / trunc
    val = torch.where(has_band & dom3, val, torch.zeros_like(val))
    s_sum = P(val)                                    # mean of val over the block = sum / factor^3
    del val, dom3, has_band
    safe = n_dom.clamp(min=1.0 / factor ** 3)
    out = {"sdf": torch.round((s_sum / n_band.clamp(min=1.0 / factor ** 3)) * SDF_Q).clamp(-SDF_Q, SDF_Q).to(torch.int8),
           "band": torch.round(n_band / safe * 255.0).to(torch.uint8),
           "free": torch.round(n_free / safe * 255.0).to(torch.uint8),
           "dom": torch.round(n_dom * 255.0).to(torch.uint8)}
    out["sdf"] = torch.where(n_band > 0, out["sdf"], torch.zeros_like(out["sdf"]))
    return out


def structure_lean(fu, known_ratio=0.999):
    """int8 structure (EMPTY / UNKNOWN / BAND codes of m_common), bitwise equal to m_p1.P1Fusion.structure() but
    without its float64 full-grid input_volume: the band value is evaluated in float64 only on the cnt > 0 voxels
    (BAND if |v| < 0.999 TRUNC, EMPTY if v >= 0.999 TRUNC, UNKNOWN otherwise: the frozen rule, under which a band
    voxel at the edge of the truncation is not BAND), EMPTY where carved and cnt == 0, UNKNOWN elsewhere."""
    dims = [int(x) for x in fu.dims]
    thr = TRUNC * known_ratio
    s = torch.full((int(np.prod(dims)),), UNKNOWN, dtype=torch.int8, device=fu.sum.device)
    s[fu.empty] = EMPTY
    idx = torch.nonzero(fu.cnt > 0)[:, 0]
    v = (fu.sum[idx].double() / FIXED / fu.cnt[idx].double()).clamp(-TRUNC, TRUNC).float()
    cls = torch.where(v >= thr, torch.full_like(idx, EMPTY, dtype=torch.int8),
                      torch.where(v.abs() < thr, torch.full_like(idx, BAND, dtype=torch.int8), torch.full_like(idx, UNKNOWN, dtype=torch.int8)))
    s[idx] = cls
    return s.reshape(dims)


def aggregate_fusion(fu, factor=FACTOR):
    """P1Fusion-like object (sum, cnt, empty, dims, dom) -> aggregate()."""
    return aggregate(fu.sum, fu.cnt, fu.empty, fu.dims, [m.cpu().numpy() if torch.is_tensor(m) else m for m in fu.dom], factor)


def to_blob(grid, level=1):
    """dict of torch / numpy grids -> (zlib bytes, shape). Order: sdf (int8 as uint8), band, free, dom."""
    arrs = [np.ascontiguousarray(grid[k].cpu().numpy() if torch.is_tensor(grid[k]) else grid[k]) for k in ("sdf", "band", "free", "dom")]
    shape = arrs[0].shape
    raw = np.concatenate([a.reshape(-1).view(np.uint8) for a in arrs])
    return zlib.compress(raw.tobytes(), level), shape


def from_blob(blob, shape):
    """-> dict of numpy grids (sdf int8, band / free / dom uint8) of the given 4x shape."""
    n = int(np.prod(shape))
    raw = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    assert raw.size == 4 * n, (raw.size, shape)
    return {"sdf": raw[:n].view(np.int8).reshape(shape), "band": raw[n:2 * n].reshape(shape),
            "free": raw[2 * n:3 * n].reshape(shape), "dom": raw[3 * n:].reshape(shape)}


def unpack(grid, device, dtype=torch.float32):
    """numpy / torch grids -> (x [1, 3, X4, Y4, Z4] in [-1, 1] / [0, 1], valid [1, 1, X4, Y4, Z4] in {0, 1})."""
    t = lambda k: torch.as_tensor(np.asarray(grid[k]) if not torch.is_tensor(grid[k]) else grid[k], device=device)  # noqa: E731
    sdf = t("sdf").to(dtype) / SDF_Q
    band = t("band").to(dtype) / 255.0
    free = t("free").to(dtype) / 255.0
    valid = (t("dom") > 0).to(dtype)
    return torch.stack([sdf, band, free])[None] * valid[None, None], valid[None, None]


def recon_targets(x, valid):
    """x [B, 3, ...], valid [B, 1, ...] -> class [B, ...] (EMPTY / UNKNOWN / BAND), sdf target [B, 1, ...],
    band mask [B, 1, ...] (float), valid mask [B, ...] (bool). Cells outside the domain get UNKNOWN but are masked."""
    sdf, band, free = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    is_band = band > 0
    cls = torch.full(band.shape, UNKNOWN, dtype=torch.long, device=x.device)
    cls = torch.where(free >= 0.5, torch.full_like(cls, EMPTY), cls)
    cls = torch.where(is_band, torch.full_like(cls, BAND), cls)[:, 0]
    return cls, sdf, (is_band & (valid > 0)).to(x.dtype), valid[:, 0] > 0


def pack_grid(grid):
    """dict of 4x grids -> one uint8 tensor [4, X4, Y4, Z4] (sdf int8 bits, band, free, dom): a hashable planner-side
    'coarse' object (the V2 planners hash and encode the object returned by Geometry.coarse())."""
    t = lambda k: grid[k] if torch.is_tensor(grid[k]) else torch.as_tensor(np.asarray(grid[k]))  # noqa: E731
    return torch.stack([t("sdf").view(torch.uint8), t("band"), t("free"), t("dom")])


def unpack_packed(packed, device, dtype=torch.float32):
    """Inverse of pack_grid -> (x, valid) as unpack()."""
    g = {"sdf": packed[0].view(torch.int8), "band": packed[1], "free": packed[2], "dom": packed[3]}
    return unpack(g, device, dtype)


def extent(shape4, factor=FACTOR):
    """Model-unit extent [3] of a 4x grid (its padded fine grid)."""
    return torch.tensor([s * factor * VS for s in shape4], dtype=torch.float32)
