"""Camera rays of the frozen sensor (EXP3b m_common: MAGICIAN back-projection convention, 456 x 256, fov 60) for the
114 x 64 scoring grid shifted by a sub-pixel offset, and the PyTorch3D intrinsics that put the pixel centres of a
64 x 114 rasterisation exactly on those rays (training labels = first mesh intersection along the planner's rays).

numpy only; the constants are asserted equal to m_common in tests/futureview.
"""
import hashlib

import numpy as np

H, W = 256, 456
M_SHORT = 256
FOV_DEG = 60.0
S_PROJ = 1.0 / np.tan(np.radians(FOV_DEG / 2.0))
BLOCK = 4
GH, GW = H // BLOCK, W // BLOCK            # 64 x 114 scoring grid
JITTER_RANGE = (-2.0, 2.0)


def grid_uv(jitter=(0.0, 0.0)):
    """Continuous pixel positions (u, v) of the scoring grid (block centres 4 i + 1.5) shifted by jitter = (du, dv);
    flat order row-major over (64, 114), as m_common.scoring_pixel_grid."""
    v, u = np.meshgrid(BLOCK * np.arange(GH) + (BLOCK - 1) / 2.0 + float(jitter[1]),
                       BLOCK * np.arange(GW) + (BLOCK - 1) / 2.0 + float(jitter[0]), indexing="ij")
    return u.reshape(-1), v.reshape(-1)


def jitter_of(scene: str, traj: str, t: int, a: int):
    """Seeded training shift (du, dv) in [-2, 2)^2 of one candidate view (protocol §2.2)."""
    h = hashlib.sha256(f"exp5-V2|jitter|{scene}|{traj}|{int(t)}|{int(a)}".encode()).hexdigest()
    r = np.random.default_rng(int(h[:15], 16)).uniform(JITTER_RANGE[0], JITTER_RANGE[1], size=2)
    return float(r[0]), float(r[1])


def label_intrinsics(jitter=(0.0, 0.0)):
    """(fx, fy, px, py) of a PyTorch3D PerspectiveCameras(in_ndc=True) whose 64 x 114 pixel centres are the rays of
    grid_uv(jitter) under the MAGICIAN convention.

    MAGICIAN ray of pixel (u, v): X/Z = ndc_x(u) / s, Y/Z = ndc_y(v) / s with ndc_x(u) = W/m - 2u/(m-1),
    ndc_y(v) = H/m - 2v/(m-1). PyTorch3D pixel centre (i, j) of an H' x W' image (short side H' spans [-1, 1], +X left,
    +Y up): x = (W' - 1 - 2j) / H', y = (H' - 1 - 2i) / H'. With u_j = 4j + 1.5 + du, X/Z = ax - b j is affine in j
    (b = 2 BLOCK / ((m - 1) s)), so fx (X/Z) + px = x_j for all j gives fx = (2 / H') / b, px = (W' - 1) / H' - fx ax;
    the same for rows (fy = fx)."""
    m = M_SHORT
    b = 2.0 * BLOCK / ((m - 1) * S_PROJ)                      # d(X/Z)/dj
    ax = (W / m - 2.0 * ((BLOCK - 1) / 2.0 + jitter[0]) / (m - 1)) / S_PROJ
    ay = (H / m - 2.0 * ((BLOCK - 1) / 2.0 + jitter[1]) / (m - 1)) / S_PROJ
    f = (2.0 / GH) / b
    px = (GW - 1) / GH - f * ax
    py = (GH - 1) / GH - f * ay
    return f, f, px, py


def sha_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def model_rays_torch(R, T, jit, origin_z, k_M, sensor_range=70.0):
    """Batched torch version of m_common.pixel_rays + SceneFrame.runtime_to_model + SceneCaster.rays for V views:
    R [V, 3, 3], T [V, 3], jit [V, 2] (float64 tensors), origin_z [3], k_M float -> o [V, 3], d [V, 7296, 3] (model
    frame, unit), tcap [V, 7296] (model units), dz [V, 7296] (z-depth per runtime ray length). float64."""
    import torch
    dev = R.device
    j = torch.arange(GW, device=dev, dtype=torch.float64)
    i = torch.arange(GH, device=dev, dtype=torch.float64)
    v, u = torch.meshgrid(BLOCK * i + (BLOCK - 1) / 2.0, BLOCK * j + (BLOCK - 1) / 2.0, indexing="ij")
    u = u.reshape(1, -1) + jit[:, 0:1]
    v = v.reshape(1, -1) + jit[:, 1:2]
    nx = W / M_SHORT - 2.0 * u / (M_SHORT - 1)
    ny = H / M_SHORT - 2.0 * v / (M_SHORT - 1)
    view = torch.stack([nx / S_PROJ, ny / S_PROJ, torch.ones_like(nx)], -1)          # z = 1
    p1 = (view - T[:, None]) @ R.transpose(1, 2)
    c = -(T[:, None] @ R.transpose(1, 2))[:, 0]
    dd = p1 - c[:, None]
    n = dd.norm(dim=-1, keepdim=True)
    d_y = dd / n
    dz = 1.0 / n[..., 0]
    zu = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64, device=dev)
    o = (c @ zu.T - origin_z) / k_M
    d = d_y @ zu.T
    return o, d, sensor_range / dz / k_M, dz
