"""Observations of the belief update: new depth rays, and the fused observation state of the grid.

- RayBundle: rays of one or more new depth frames in the grid's world frame: origins, unit directions, observed range
  t_obs (inf = no return) and the frame index of every ray. Any sensor model can produce it (DepthObservation below
  for pinhole images; an experiment adapter for the MAGICIAN / PyTorch3D cameras of MACARONS++).
- DepthObservation: one depth image of a pinhole camera in the convention of active.render.SceneRenderer and
  active.fusion.PatchFusion: cam2world in the OpenGL / Blender convention (looks along local -z, +y up), z-depth in
  metres, 0 = no return, pixel centres (u + 0.5, v + 0.5). The ray of pixel (u, v) has camera direction
  ((u+0.5-cx)/fx, -(v+0.5-cy)/fy, -1); t_obs = z * |direction|. `from_fov` = the square-FOV pinhole of the official
  renders (f = (w/2) / tan(fov/2), principal point at the image centre).
- ObservationState: dense TSDF of everything observed so far (unobserved = -truncation, the input of
  `WorldModel.make_batch`) and its three-state structure EMPTY / UNKNOWN / BAND with the thresholds of
  active.episode.structure_torch. The belief package never fuses observations itself; it reads this state.
"""
import dataclasses
import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from seen2scene.belief.state import GridFrame

EMPTY, UNKNOWN, BAND = 0, 1, 2
BAND_RATIO = 0.999  # active.episode.KR: |tsdf| < 0.999 truncation is BAND, tsdf >= 0.999 truncation is EMPTY


def classify_structure(tsdf: torch.Tensor, truncation: float) -> torch.Tensor:
    """EMPTY / UNKNOWN / BAND (int8) of a fused TSDF; same thresholds as active.episode.structure_torch."""
    s = torch.full_like(tsdf, UNKNOWN, dtype=torch.int8)
    s[tsdf >= truncation * BAND_RATIO] = EMPTY
    s[tsdf.abs() < truncation * BAND_RATIO] = BAND
    return s


def structure_fractions(structure: torch.Tensor, factor: int) -> torch.Tensor:
    """Fractions of EMPTY / UNKNOWN / BAND voxels per factor^3 cell -> [3, X, Y, Z] float32 (the latent grid).
    Voxels beyond the grid (padding up to a multiple of factor) count as UNKNOWN."""
    s = structure.long()
    pad = [(-n) % factor for n in s.shape]
    if any(pad):
        s = F.pad(s, (0, pad[2], 0, pad[1], 0, pad[0]), value=UNKNOWN)
    one_hot = F.one_hot(s, 3).permute(3, 0, 1, 2).float()[None]
    return F.avg_pool3d(one_hot, factor, factor)[0]


@dataclasses.dataclass
class RayBundle:
    origins: torch.Tensor  # [R, 3] (world frame of the grid)
    dirs: torch.Tensor  # [R, 3] unit
    t_obs: torch.Tensor  # [R] observed range along the ray, inf = no return
    frame_ids: torch.Tensor  # [R] int64, index of the depth frame a ray belongs to

    def __post_init__(self):
        R = self.dirs.shape[0]
        if self.origins.dim() == 1:
            self.origins = self.origins[None].expand(R, 3)
        if self.origins.shape != (R, 3) or self.t_obs.shape != (R,) or self.frame_ids.shape != (R,):
            raise ValueError("RayBundle: origins [R, 3], dirs [R, 3], t_obs [R], frame_ids [R]")

    @property
    def n_rays(self) -> int:
        return int(self.dirs.shape[0])

    @property
    def n_frames(self) -> int:
        return int(torch.unique(self.frame_ids).numel()) if self.n_rays else 0

    def to(self, device) -> "RayBundle":
        return RayBundle(self.origins.to(device), self.dirs.to(device), self.t_obs.to(device), self.frame_ids.to(device))

    @staticmethod
    def cat(bundles: Sequence["RayBundle"]) -> "RayBundle":
        """Concatenate frames; frame ids are renumbered so that every input bundle keeps distinct frames."""
        out, offset = [], 0
        for b in bundles:
            ids = torch.unique(b.frame_ids, return_inverse=True)[1] + offset
            offset = int(ids.max()) + 1 if ids.numel() else offset
            out.append((b.origins, b.dirs, b.t_obs, ids))
        if not out:
            raise ValueError("no rays")
        return RayBundle(*[torch.cat([o[i] for o in out]) for i in range(4)])


@dataclasses.dataclass
class DepthObservation:
    depth: torch.Tensor  # [H, W] z-depth (m), 0 = no return
    c2w: torch.Tensor  # [4, 4] OpenGL cam2world
    fx: float
    fy: float
    cx: float
    cy: float
    pixel_center: bool = True
    zfar: Optional[float] = None  # ranges >= zfar count as no return

    @classmethod
    def from_fov(cls, depth, c2w, fov_deg: float = 60.0, pixel_center: bool = True,
                 zfar: Optional[float] = None) -> "DepthObservation":
        depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
        h, w = depth.shape
        f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
        return cls(depth, torch.as_tensor(np.asarray(c2w), dtype=torch.float64), f, f, w / 2, h / 2, pixel_center,
                   zfar)

    def to_rays(self, stride: int = 1, max_rays: Optional[int] = None, generator: Optional[torch.Generator] = None,
                frame_id: int = 0, device="cpu") -> RayBundle:
        """Pixels on a stride grid, then a random subset of max_rays (deterministic with `generator`)."""
        h, w = self.depth.shape
        v, u = torch.meshgrid(torch.arange(0, h, stride), torch.arange(0, w, stride), indexing="ij")
        v, u = v.reshape(-1), u.reshape(-1)
        if max_rays is not None and v.numel() > max_rays:
            sel = torch.randperm(v.numel(), generator=generator)[:max_rays].sort().values
            v, u = v[sel], u[sel]
        off = 0.5 if self.pixel_center else 0.0
        x = (u.double() + off - self.cx) / self.fx
        y = -(v.double() + off - self.cy) / self.fy
        d_cam = torch.stack([x, y, -torch.ones_like(x)], -1)
        norm = d_cam.norm(dim=-1)
        c2w = self.c2w.to(torch.float64)
        dirs = (d_cam / norm[:, None]) @ c2w[:3, :3].T
        z = self.depth[v, u].double()
        no_return = z <= 0
        if self.zfar is not None:
            no_return |= z >= self.zfar
        t_obs = torch.where(no_return, torch.full_like(z, math.inf), z * norm)
        R = dirs.shape[0]
        return RayBundle(c2w[:3, 3].float().to(device), dirs.float().to(device), t_obs.float().to(device),
                         torch.full((R,), frame_id, dtype=torch.long, device=device))


@dataclasses.dataclass
class ObservationState:
    """Fused observation of the grid after `version` observations (TSDF, unobserved = -truncation)."""

    frame: GridFrame
    tsdf: torch.Tensor
    version: int = 0
    _structure: Optional[torch.Tensor] = dataclasses.field(default=None, repr=False)

    @property
    def structure(self) -> torch.Tensor:
        if self._structure is None:
            self._structure = classify_structure(self.tsdf, self.frame.truncation)
        return self._structure

    @classmethod
    def from_structure(cls, frame: GridFrame, tsdf: torch.Tensor, structure: torch.Tensor,
                       version: int = 0) -> "ObservationState":
        """When the caller's fusion defines the structure itself (e.g. the P1 layers of EXP3b)."""
        return cls(frame, tsdf, version, structure)

    @classmethod
    def unobserved(cls, frame: GridFrame, device="cpu") -> "ObservationState":
        return cls(frame, torch.full(frame.shape, -frame.truncation, dtype=torch.float32, device=device), 0)


def bundles_of(observations: Sequence[DepthObservation], stride: int = 1, max_rays: Optional[int] = None,
               generator: Optional[torch.Generator] = None, device="cpu") -> RayBundle:
    """One RayBundle of several depth images (frame ids 0..n-1)."""
    return RayBundle.cat([o.to_rays(stride, max_rays, generator, i, device) for i, o in enumerate(observations)])


__all__: List[str] = ["EMPTY", "UNKNOWN", "BAND", "classify_structure", "structure_fractions", "RayBundle",
                      "DepthObservation", "ObservationState", "bundles_of"]
