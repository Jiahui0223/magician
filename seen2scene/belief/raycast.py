"""First-surface ray casting of dense TSDF worlds on an arbitrary GridFrame, for arbitrary rays.

Same marching rule as active.raycast.PatchRaycaster.cast (exp1-task4), generalised from the 256^3 patch and
the square-FOV camera rays to any grid shape and any ray set (here: the pixels of an observed depth image):
- rays are clipped to the grid box [origin, origin + shape * voxel_size] (and to t >= 0, t <= t_far) and sampled
  every `step_vox` voxels; values are trilinear on voxel centres (grid_sample, align_corners=True); samples
  outside the box count as +truncation;
- the hit is the FIRST + -> - sign change (free space -> inside), refined linearly; leaving an object never counts.
PatchRaycaster stays the implementation of the frozen planners (S1/S2/VHD); `tests/belief/test_raycast.py` checks
that both give the same hits on the 256^3 patch.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from seen2scene.belief.state import GridFrame


def box_interval(frame: GridFrame, origins: torch.Tensor, dirs: torch.Tensor):
    """Entry / exit distance of every ray in the grid box (valid where t_out > t_in); t_in >= 0."""
    lo = torch.as_tensor(frame.origin, dtype=dirs.dtype, device=dirs.device)
    size = torch.as_tensor(frame.extent, dtype=dirs.dtype, device=dirs.device)
    o = origins.expand_as(dirs) - lo
    inv = 1.0 / torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)
    t0 = (0.0 - o) * inv
    t1 = (size - o) * inv
    t_in = torch.minimum(t0, t1).amax(-1).clamp(min=0.0)
    t_out = torch.maximum(t0, t1).amin(-1)
    return t_in, t_out


@torch.no_grad()
def march_first_hit(volumes: torch.Tensor, frame: GridFrame, origins: torch.Tensor, dirs: torch.Tensor,
                    t_far: Optional[torch.Tensor] = None, step_vox: float = 0.5, chunk: int = 1 << 21
                    ) -> Dict[str, torch.Tensor]:
    """volumes [K, D, H, W] (dense TSDF), origins [3] or [R, 3], unit dirs [R, 3] (world metres, same device).

    Returns t [K, R] first-hit distance (inf = no hit), t_in / t_out [R] (box interval, after t_far) and
    valid [R] (the ray crosses the box)."""
    K = volumes.shape[0]
    dev = volumes.device
    dirs = dirs.to(dev, torch.float32)
    origins = origins.to(dev, torch.float32)
    vs, trunc, step = frame.voxel_size, frame.truncation, step_vox * frame.voxel_size
    shape = torch.as_tensor(frame.shape, dtype=torch.float32, device=dev)
    lo = torch.as_tensor(frame.origin, dtype=torch.float32, device=dev)
    o = origins.expand_as(dirs) - lo
    t_in, t_out = box_interval(frame, origins, dirs)
    if t_far is not None:
        t_out = torch.minimum(t_out, t_far.to(dev, torch.float32))
    valid = t_out > t_in
    R = dirs.shape[0]
    t_hit = torch.full((K, R), float("inf"), device=dev)
    if not valid.any():
        return {"t": t_hit, "t_in": t_in, "t_out": t_out, "valid": valid}
    M = int(torch.ceil(((t_out - t_in).clamp(min=0).max()) / step).item()) + 2
    vol5 = volumes[:, None].float()  # [K, 1, D, H, W]; grid_sample: x -> W (k), y -> H (j), z -> D (i)
    rays_per_chunk = max(1, chunk // (M * K))
    for s in range(0, R, rays_per_chunk):
        sl = slice(s, min(R, s + rays_per_chunk))
        ts = t_in[sl, None] + step * torch.arange(M, device=dev)[None]  # [r, M]
        inside = (ts <= t_out[sl, None]) & valid[sl, None]
        p = o[sl, None] + dirs[sl, None] * ts[..., None]  # [r, M, 3] grid-local metres
        u = p / vs - 0.5  # continuous voxel-centre index
        g = 2 * u / (shape - 1) - 1
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)
        grid = grid[None, :, :, None].expand(K, -1, -1, -1, -1)  # [K, r, M, 1, 3]
        v = F.grid_sample(vol5, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, :, :, 0]
        v = torch.where(inside[None], v, torch.full_like(v, trunc))
        cross = (v[..., :-1] > 0) & (v[..., 1:] <= 0)  # free -> inside
        has = cross.any(-1)
        j = torch.argmax(cross.int(), dim=-1)
        v0 = torch.gather(v, 2, j[..., None])[..., 0]
        v1 = torch.gather(v, 2, (j + 1)[..., None])[..., 0]
        frac = (v0 / (v0 - v1).clamp(min=1e-9)).clamp(0, 1)
        th = t_in[sl][None] + step * (j.float() + frac)
        t_hit[:, sl] = torch.where(has, th, torch.full_like(th, float("inf")))
    return {"t": t_hit, "t_in": t_in, "t_out": t_out, "valid": valid}
