"""GPU TSDF fusion into a 256^3 patch, replicating the official Seen2Scene VDBFusion fork
(github.com/quan-meng/vdbfusion @ seen2scene-v0.1.6, src/vdbfusion/vdbfusion/VDBVolume.cpp)
restricted to the voxels of the patch (exp1-task6).

Official semantics we replicate (read from the fork's source):
- global grid: VDB index G covers world [G, G+1)*vs; its value is the SDF sampled at world
  (G + 0.5)*vs (GetVoxelCenter adds vs/2); background (unobserved) = -trunc;
- per depth point p with camera origin o: the ray o -> p is traversed with an exact voxel DDA
  from t0 = 0 (space carving) to t1 = |p - o| + trunc;
- per visited voxel: sdf = sign((v - o).(p - v)) * |p - v|; if sdf > -trunc:
  tsdf = min(trunc, sdf), weight 1, running weighted average (= mean over all updates);
- the dataset crops the VDB at worldToIndexCellCentered(bbox_min) = floor(bbox_min/vs + 0.5),
  so patch voxel a corresponds to global index I0 + a, where I0 = floor(bbox_min/vs + 0.5).

Official depth images (data/front3d/scripts.py:render_scene_w_pyrender): pyrender,
PerspectiveCamera(yfov=60 deg, znear=0.05, zfar=20), 512x512, cam2world = BlenderProc pose,
depth > 1e4 -> 0, stored as uint16(depth / 100 m * 65535) (floor). `quantize_depth` reproduces it.
The back-projection code (tools/reconstruction/integration.py) is NOT released. Its pixel
convention was calibrated against the official VDBs in exp1-task6 on 5 dev patches with the p_0.1 cameras.
Pixel CENTRES (u+0.5) reproduce the official fusion: three-state agreement 99.997%, BAND mean abs
error 0.0066 mm. Pixel corners do not: 99.34%, 2.27 mm. `pixel_center=True` is the default.
Render depth with SceneRenderer(mesh, 512, 512, 60.0, znear=0.05, zfar=20.0), then `quantize_depth`.
"""
from typing import Optional

import numpy as np
import torch

VOXEL_SIZE = 0.011
TRUNCATION = 3 * VOXEL_SIZE
MAX_DEPTH = 100.0


def quantize_depth(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """The official uint16 depth storage and its decoding (1.53 mm steps, background 0)."""
    d = depth.copy()
    d[d > 1e4] = 0.0
    q = (d / max_depth * np.iinfo(np.uint16).max).astype(np.uint16)
    return q.astype(np.float32) / np.iinfo(np.uint16).max * max_depth


def grid_origin_index(bbox_min, vs: float = VOXEL_SIZE) -> np.ndarray:
    """Global VDB index of patch voxel (0,0,0): OpenVDB worldToIndexCellCentered (round half up)."""
    return np.floor(np.asarray(bbox_min, dtype=np.float64) / vs + 0.5).astype(np.int64)


class PatchFusion:
    def __init__(self, bbox_min, size: int = 256, vs: float = VOXEL_SIZE, trunc: float = TRUNCATION,
                 device: str = "cuda"):
        self.dev = torch.device(device)
        self.vs, self.trunc, self.size = vs, trunc, size
        self.I0 = grid_origin_index(bbox_min, vs)
        self.corner = torch.as_tensor(self.I0 * vs, dtype=torch.float64, device=self.dev)  # world of cell (0,0,0) lower corner
        n = size ** 3
        self.sum = torch.zeros(n, dtype=torch.float64, device=self.dev)
        self.cnt = torch.zeros(n, dtype=torch.float64, device=self.dev)

    @torch.no_grad()
    def integrate(self, depth: np.ndarray, c2w: np.ndarray, fov_deg: float = 60.0, pixel_center: bool = True,
                  ray_chunk: int = 1 << 16):
        """Fuse one depth image (z-depth, metres, 0 = no return) seen from c2w (OpenGL/Blender)."""
        h, w = depth.shape
        f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
        off = 0.5 if pixel_center else 0.0
        v, u = np.nonzero(depth > 0)
        z = depth[v, u].astype(np.float64)
        x = (u + off - w / 2) / f * z
        y = -(v + off - h / 2) / f * z
        p_cam = np.stack([x, y, -z], 1)
        pts = p_cam @ c2w[:3, :3].T + c2w[:3, 3]
        o = torch.as_tensor(c2w[:3, 3], dtype=torch.float64, device=self.dev)
        P = torch.as_tensor(pts, dtype=torch.float64, device=self.dev)
        for s in range(0, len(P), ray_chunk):
            self._integrate_rays(o, P[s:s + ray_chunk])

    def _integrate_rays(self, o: torch.Tensor, p: torch.Tensor):
        vs, S = self.vs, self.size
        d = p - o
        depth = d.norm(dim=1)
        d = d / depth[:, None]
        t1 = depth + self.trunc
        # ray in cell coordinates c(t) = (o - corner)/vs + d t / vs ; cell a spans [a, a+1)
        c0 = (o - self.corner) / vs
        dc = d / vs
        inv = 1.0 / torch.where(dc.abs() < 1e-12, torch.full_like(dc, 1e-12), dc)
        ta = (0.0 - c0) * inv
        tb = (S - c0) * inv
        t_in = torch.minimum(ta, tb).amax(1).clamp(min=0.0)
        t_out = torch.minimum(torch.maximum(ta, tb).amin(1), t1)
        alive = t_out > t_in
        if not alive.any():
            return
        idx = torch.nonzero(alive)[:, 0]
        p, d, dc, t_in, t_out = p[idx], d[idx], dc[idx], t_in[idx], t_out[idx]
        pos = c0[None] + dc * (t_in[:, None] + 1e-9)
        cell = torch.floor(pos).long().clamp(0, S - 1)
        step = torch.sign(dc).long()
        step = torch.where(step == 0, torch.ones_like(step), step)
        nxt = cell + (step > 0).long()  # next boundary in cell coords
        t_max = (nxt.double() - c0[None]) / torch.where(dc.abs() < 1e-12, torch.full_like(dc, 1e-12), dc)
        t_max = torch.where(dc.abs() < 1e-12, torch.full_like(t_max, float("inf")), t_max)
        t_delta = (1.0 / dc.abs()).clamp(max=1e30)
        t_cur = t_in.clone()
        active = torch.ones(len(idx), dtype=torch.bool, device=self.dev)
        while active.any():
            a = torch.nonzero(active)[:, 0]
            c = cell[a]
            centre = self.corner[None] + (c.double() + 0.5) * vs
            vo = centre - o[None]
            pv = p[a] - centre
            dist = pv.norm(dim=1)
            proj = (vo * pv).sum(1)
            sdf = torch.where(proj >= 0, dist, -dist)
            ok = sdf > -self.trunc
            if ok.any():
                flat = (c[ok, 0] * S + c[ok, 1]) * S + c[ok, 2]
                self.sum.index_add_(0, flat, torch.clamp(sdf[ok], max=self.trunc))
                self.cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
            # advance along the axis with the smallest t_max
            tm = t_max[a]
            ax = torch.argmin(tm, dim=1)
            t_next = tm.gather(1, ax[:, None])[:, 0]
            rows = a
            cell[rows, ax] += step[rows, ax]
            t_max[rows, ax] += t_delta[rows, ax]
            t_cur[rows] = t_next
            inside = ((cell[rows] >= 0) & (cell[rows] < S)).all(1)
            active[rows] = inside & (t_next <= t_out[rows])

    def tsdf(self) -> torch.Tensor:
        """Dense [S,S,S] float32 TSDF; unobserved voxels = -trunc (official background)."""
        vol = torch.where(self.cnt > 0, self.sum / self.cnt.clamp(min=1), torch.full_like(self.sum, -self.trunc))
        return vol.float().reshape(self.size, self.size, self.size)
