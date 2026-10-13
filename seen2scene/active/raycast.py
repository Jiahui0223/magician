"""GPU ray casting of TSDF worlds inside the target patch (exp1-task4).

Why not fvdb.GridBatch.ray_implicit_intersection: it is exact on dense grids, but on
sparse band-only grids (the form of decoded worlds) it returned wrong hits in a test on
an analytic sphere. Errors were about 2.7 voxels, a ray that misses was reported as a
hit, and a ray starting inside was wrong (exp1-task4). We therefore march rays
ourselves:

- a world is a dense 256^3 TSDF of the patch. Band values are kept and everything else
  is +trunc (the `TSDFPatch.to_dense()` convention of decoded worlds);
- rays are clipped to the patch box and sampled every `step_vox` voxels; values are
  trilinearly interpolated (grid_sample on voxel centres); points outside the patch count as +trunc;
- the hit is the FIRST +/- sign change (free space -> inside), refined by linear
  interpolation. Leaving an object (- to +) never counts as a hit;
- `t_max` (per ray) optionally stops rays early, e.g. at occluders outside the patch.

Coordinates: world metres. Rays use the same pinhole model as SceneRenderer
(OpenGL/Blender cam2world, square FOV, pixel centres). Depth is the ray distance t; use
`z_factor` to convert to z-depth.

Setting: regional active inspection within an otherwise mapped scene. Geometry OUTSIDE the
target patch is known and blocks rays during scoring (decided 2026-09-26, option A); the
patch content is unknown. `outside_block_tmax` builds the per-ray t_max for `cast`. A ray is
blocked iff the full scene mesh is hit before the ray enters the patch box. Only geometry in
front of the box decides this, so nothing inside the patch leaks.

Frozen planner coverage scores (used by S1/S2/VHD and every later experiment):
- G_cov (main): number of unique voxels hit whose 3x3x3 neighbourhood contains no observed
  BAND voxel, i.e. surface the view would newly see (same neighbourhood rule as the
  evaluation surface coverage in eval/metrics_known.py);
- G_free (auxiliary): number of unique voxels that are UNKNOWN in the current observation and
  that the ray passes through before its hit, i.e. free space the view would newly observe.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

VOXEL_SIZE = 0.011
TRUNCATION = 3 * VOXEL_SIZE
PATCH = 256
EMPTY, UNKNOWN, BAND = 0, 1, 2


def camera_rays(c2w: np.ndarray, res: int = 128, fov_deg: float = 60.0):
    """Unit ray directions [R,3] (world), origin [3], and z_factor [R] (z-depth = t * z_factor)."""
    t = np.tan(np.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(res), np.arange(res))
    x = (2 * (u + 0.5) / res - 1) * t
    y = (1 - 2 * (v + 0.5) / res) * t
    d_cam = np.stack([x, y, -np.ones_like(x)], -1).reshape(-1, 3)
    norm = np.linalg.norm(d_cam, axis=1, keepdims=True)
    d_cam = d_cam / norm
    dirs = d_cam @ c2w[:3, :3].T
    return dirs.astype(np.float32), c2w[:3, 3].astype(np.float32), (1.0 / norm[:, 0]).astype(np.float32)


def dense_from_band(band_ijk: np.ndarray, band_values: np.ndarray, trunc: float = TRUNCATION) -> np.ndarray:
    vol = np.full((PATCH,) * 3, trunc, dtype=np.float32)
    vol[band_ijk[:, 0], band_ijk[:, 1], band_ijk[:, 2]] = band_values
    return vol


class PatchRaycaster:
    def __init__(self, bbox_min, device: str = "cuda", step_vox: float = 0.5, voxel_size: float = VOXEL_SIZE,
                 trunc: float = TRUNCATION, chunk: int = 1 << 21):
        self.dev = torch.device(device)
        self.bmin = torch.as_tensor(np.asarray(bbox_min, dtype=np.float32), device=self.dev)
        self.vs, self.trunc, self.step = voxel_size, trunc, step_vox * voxel_size
        self.size = PATCH * voxel_size
        self.chunk = chunk  # max samples per grid_sample call

    @torch.no_grad()
    def outside_block_tmax(self, c2w: np.ndarray, scene_renderer, res: int = 128, fov_deg: float = 60.0):
        """Per-ray t_max for `cast`: 0 for rays blocked before entering the patch, else +inf.

        scene_renderer: SceneRenderer of the full scene mesh at resolution `res` with the same FOV.
        It uses the same pinhole model and pixel centres as `camera_rays`.
        Returns (t_max [R] on device, blocked fraction among rays that enter the patch).
        """
        dirs, org, zf = camera_rays(c2w, res, fov_deg)
        d = torch.as_tensor(dirs, device=self.dev)
        o = (torch.as_tensor(org, device=self.dev) - self.bmin).expand_as(d)
        tmin, tmax = self._entry_exit(o, d)
        enters = tmax > tmin
        z = torch.as_tensor(scene_renderer.render_depth(c2w).reshape(-1), device=self.dev)
        t_scene = torch.where(z > 0, z / torch.as_tensor(zf, device=self.dev), torch.full_like(z, float("inf")))
        blocked = enters & (t_scene < tmin - self.vs)
        t_max = torch.where(blocked, torch.zeros_like(tmin), torch.full_like(tmin, float("inf")))
        frac = float(blocked.sum() / enters.sum().clamp(min=1))
        return t_max, frac

    def _entry_exit(self, o: torch.Tensor, d: torch.Tensor):
        inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t0 = (0.0 - o) * inv
        t1 = (self.size - o) * inv
        tmin = torch.minimum(t0, t1).amax(-1).clamp(min=0.0)
        tmax = torch.maximum(t0, t1).amin(-1)
        return tmin, tmax  # valid where tmax > tmin

    @torch.no_grad()
    def cast(self, volumes: torch.Tensor, c2w: np.ndarray, res: int = 128, fov_deg: float = 60.0,
             t_max: Optional[torch.Tensor] = None, observed_structure: Optional[torch.Tensor] = None
             ) -> Dict[str, torch.Tensor]:
        """Cast one view into K worlds.

        volumes: [K, 256, 256, 256] dense TSDF on device. observed_structure: [256]^3 int8 of
        the current observation (optional; needed for the G_cov/G_free scores).
        Returns t [K,R] (inf = no hit), hit_ijk [K,R,3] (-1 if none), z_factor [R], and
        optionally new_surface [K,R] (bool), g_cov [K] and g_free [K].
        """
        K = volumes.shape[0]
        dirs, org, zf = camera_rays(c2w, res, fov_deg)
        d = torch.as_tensor(dirs, device=self.dev)
        o = torch.as_tensor(org, device=self.dev) - self.bmin  # patch-local
        o = o.expand_as(d)
        tmin, tmax = self._entry_exit(o, d)
        if t_max is not None:
            tmax = torch.minimum(tmax, t_max.to(self.dev))
        valid = tmax > tmin
        R = d.shape[0]
        M = int(torch.ceil(((tmax - tmin).clamp(min=0).max()) / self.step).item()) + 2
        t_hit = torch.full((K, R), float("inf"), device=self.dev)
        vol5 = volumes[:, None]  # [K,1,D,H,W]; grid_sample maps x->W, y->H, z->D
        rays_per_chunk = max(1, self.chunk // (M * K))
        for s in range(0, R, rays_per_chunk):
            sl = slice(s, min(R, s + rays_per_chunk))
            ts = tmin[sl, None] + self.step * torch.arange(M, device=self.dev)[None]  # [r,M]
            inside = (ts <= tmax[sl, None]) & valid[sl, None]
            p = o[sl, None] + d[sl, None] * ts[..., None]  # [r,M,3] patch-local metres
            u = p / self.vs - 0.5  # continuous voxel-centre index
            g = 2 * u / (PATCH - 1) - 1
            grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)  # (x=W=k, y=H=j, z=D=i)
            grid = grid[None, :, :, None].expand(K, -1, -1, -1, -1)  # [K,r,M,1,3]
            v = F.grid_sample(vol5, grid, mode="bilinear", padding_mode="border", align_corners=True)
            v = v[:, 0, :, :, 0]  # [K,r,M]
            v = torch.where(inside[None], v, torch.full_like(v, self.trunc))
            cross = (v[..., :-1] > 0) & (v[..., 1:] <= 0)  # free -> inside
            has = cross.any(-1)
            j = torch.argmax(cross.int(), dim=-1)  # first crossing index
            v0 = torch.gather(v, 2, j[..., None])[..., 0]
            v1 = torch.gather(v, 2, (j + 1)[..., None])[..., 0]
            frac = (v0 / (v0 - v1).clamp(min=1e-9)).clamp(0, 1)
            th = tmin[sl][None] + self.step * (j.float() + frac)
            t_hit[:, sl] = torch.where(has, th, torch.full_like(th, float("inf")))

        hit = torch.isfinite(t_hit)
        p_hit = o[None] + d[None] * torch.where(hit, t_hit, torch.zeros_like(t_hit))[..., None]
        ijk = torch.floor(p_hit / self.vs).long().clamp(0, PATCH - 1)
        ijk = torch.where(hit[..., None], ijk, torch.full_like(ijk, -1))
        out = {"t": t_hit, "hit_ijk": ijk, "z_factor": torch.as_tensor(zf, device=self.dev)}
        if observed_structure is not None:
            out.update(self._scores(observed_structure, o, d, tmin, t_hit, ijk, hit, valid))
        return out

    def _scores(self, obs, o, d, tmin, t_hit, ijk, hit, valid):
        obs = obs.to(self.dev)
        band_dil = F.max_pool3d((obs == BAND).float()[None, None], 3, 1, 1)[0, 0] > 0
        K = t_hit.shape[0]
        new = hit & ~band_dil[ijk[..., 0].clamp(min=0), ijk[..., 1].clamp(min=0), ijk[..., 2].clamp(min=0)]
        flat = (ijk[..., 0] * PATCH + ijk[..., 1]) * PATCH + ijk[..., 2]
        g_cov = torch.tensor([float(torch.unique(flat[k][new[k]]).numel()) for k in range(K)], device=self.dev)
        # G_free: unknown voxels traversed before the hit (sampled at 1 voxel steps)
        unknown = (obs == UNKNOWN).reshape(-1)
        g_free = []
        steps = torch.arange(0, int(np.ceil(self.size * np.sqrt(3) / self.vs)) + 1, device=self.dev) * self.vs
        _, tmax = self._entry_exit(o, d)
        for k in range(K):
            end = torch.where(hit[k], t_hit[k], tmax)  # rays without a hit traverse to the patch exit
            seen = torch.zeros(PATCH ** 3, dtype=torch.bool, device=self.dev)
            for s in range(0, d.shape[0], 4096):
                sl = slice(s, s + 4096)
                ts = tmin[sl, None] + steps[None]
                m = (ts < end[sl, None]) & valid[sl, None]
                q = torch.floor((o[sl, None] + d[sl, None] * ts[..., None]) / self.vs).long()
                ok = m & (q >= 0).all(-1) & (q < PATCH).all(-1)
                qf = ((q[..., 0] * PATCH + q[..., 1]) * PATCH + q[..., 2])[ok]
                qf = qf[unknown[qf]]
                seen[qf] = True
            g_free.append(seen.sum().float())
        return {"new_surface": new, "g_cov": g_cov, "g_free": torch.stack(g_free)}
