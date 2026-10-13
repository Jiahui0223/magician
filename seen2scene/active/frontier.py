"""Frontier score for the classical geometric baseline B-Frontier (exp2, definition frozen 2026-09-27).

- Frontier voxel: observed EMPTY and at least one 6-neighbour is UNKNOWN (neighbours outside the patch do not
  count: the scene outside the patch is known in the regional-inspection setting).
- Score of a candidate view: number of unique frontier voxels it can see directly. A ray enters the patch
  (outside-patch occlusion applied through t_max, as for G_cov), travels through observed EMPTY voxels and
  stops at the first observed surface (BAND) or UNKNOWN voxel; the frontier voxels it passes before stopping
  are visible. Samples every 0.5 voxel along the ray, as the G_cov ray marcher.
It uses only the current observation, never a generative model.
"""
import numpy as np
import torch

from seen2scene.active.raycast import camera_rays

EMPTY, UNKNOWN, BAND = 0, 1, 2


def frontier_mask(obs: torch.Tensor) -> torch.Tensor:
    """[S,S,S] bool: observed EMPTY voxels with at least one UNKNOWN 6-neighbour inside the patch."""
    unk = obs == UNKNOWN
    nb = torch.zeros_like(unk)
    nb[1:] |= unk[:-1]; nb[:-1] |= unk[1:]
    nb[:, 1:] |= unk[:, :-1]; nb[:, :-1] |= unk[:, 1:]
    nb[:, :, 1:] |= unk[:, :, :-1]; nb[:, :, :-1] |= unk[:, :, 1:]
    return (obs == EMPTY) & nb


@torch.no_grad()
def frontier_score(rc, obs: torch.Tensor, frontier: torch.Tensor, c2w: np.ndarray, res: int = 128,
                   fov_deg: float = 60.0, t_max=None, step_vox: float = 0.5, ray_chunk: int = 4096) -> int:
    """Unique frontier voxels directly visible from c2w. rc: the episode's PatchRaycaster (patch frame)."""
    S, vs, dev = obs.shape[0], rc.vs, rc.dev
    dirs, org, _ = camera_rays(c2w, res, fov_deg)
    d = torch.as_tensor(dirs, device=dev)
    o = (torch.as_tensor(org, device=dev) - rc.bmin).expand_as(d)
    tmin, tmax = rc._entry_exit(o, d)
    if t_max is not None:
        tmax = torch.minimum(tmax, t_max.to(dev))
    obs_f, fr_f = obs.reshape(-1), frontier.reshape(-1)
    n_samples = int(np.ceil(S * np.sqrt(3) / step_vox)) + 2
    steps = torch.arange(n_samples, device=dev, dtype=d.dtype) * (step_vox * vs)
    seen = torch.zeros(S ** 3, dtype=torch.bool, device=dev)
    for s in range(0, d.shape[0], ray_chunk):
        sl = slice(s, s + ray_chunk)
        ts = tmin[sl, None] + 1e-6 + steps[None]
        valid = ts < tmax[sl, None]
        q = torch.floor((o[sl, None] + d[sl, None] * ts[..., None]) / vs).long()
        inb = ((q >= 0) & (q < S)).all(-1)
        flat = ((q[..., 0].clamp(0, S - 1) * S + q[..., 1].clamp(0, S - 1)) * S + q[..., 2].clamp(0, S - 1))
        st = obs_f[flat]
        stop = (st != EMPTY) | ~valid | ~inb
        free = torch.cumsum(stop.int(), dim=1) == 0  # before the first stop
        hit = free & fr_f[flat]
        seen[flat[hit]] = True
    return int(seen.sum().item())
