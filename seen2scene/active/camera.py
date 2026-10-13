"""Cameras for the NBV benchmark (exp1-task3).

1. `load_official_cameras`: recover the official BlenderProc cam2world matrices from
   the released `cameras_p_*.ply` glyphs. Each camera is `create_camera_primitive`
   (tools/mesh_utils.py) transformed by its pose: 23 vertices, where vertex 0 is the
   camera centre and vertices 1-4 are the pyramid base corners at local
   (-a,-a,-d), (a,-a,-d), (a,a,-d), (-a,a,-d). So x = v2-v1, y = v4-v1 and the camera
   looks along -z (OpenGL/Blender). The glyph does not encode the FOV. The FOV is the
   official setting: 60 deg, 512x512 (`scans_fov_60.0_d_8.0_r_512`).

2. `CandidateGenerator`: per-step candidate views that follow the official camera
   distribution (tools/camera_utils.py) plus one NBV-specific rule:
   - position: uniform over the floor boxes of the room(s) under the patch, 1.5-2.0 m
     above the floor, and actually above floor geometry (downward ray hits the floor),
     like the official `position_is_above_object`;
   - rotation: yaw U(0, 360), pitch U(-60, 60) (Blender X-rotation 30-150 deg), roll 0;
   - obstacle check as in the official `perform_obstacle_in_view_check(min=0.1,
     no_background=True)`: every probe ray hits geometry, and no hit is closer than 0.1 m;
   - NBV rule: the view frustum intersects the target patch;
   - determinism: rng = default_rng([2027, scene_hash, episode, step]); candidates are
     returned in acceptance order.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import trimesh

from seen2scene.active.render import SceneRenderer

GLOBAL_SEED = 2027


def load_official_cameras(ply_path: str) -> np.ndarray:
    """[N, 4, 4] cam2world (OpenGL/Blender convention) from a cameras_p_*.ply glyph file."""
    v = np.asarray(trimesh.load(ply_path, process=False).vertices, dtype=np.float64)
    assert len(v) % 23 == 0, f"unexpected glyph vertex count {len(v)}"
    v = v.reshape(-1, 23, 3)
    center = v[:, 0]
    x = v[:, 2] - v[:, 1]
    y = v[:, 4] - v[:, 1]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    c2w = np.tile(np.eye(4), (len(v), 1, 1))
    c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2], c2w[:, :3, 3] = x, y, z, center
    return c2w


def camera_angles(c2w: np.ndarray) -> Dict[str, np.ndarray]:
    """Yaw/pitch of the viewing direction (-z) and roll (tilt of the x axis), in degrees."""
    fwd = -c2w[..., :3, 2]
    pitch = np.degrees(np.arcsin(np.clip(fwd[..., 2], -1, 1)))
    yaw = np.degrees(np.arctan2(fwd[..., 1], fwd[..., 0])) % 360
    roll = np.degrees(np.arcsin(np.clip(c2w[..., 2, 0], -1, 1)))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def look_matrix(pos: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """cam2world with roll 0 looking along (yaw, pitch)."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    x = np.cross(fwd, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, -fwd, pos
    return c2w


def floor_boxes(meta: dict) -> np.ndarray:
    """[F, 5] (xmin, ymin, xmax, ymax, z) of the floor objects in meta.json."""
    out = []
    for name, box in zip(meta["object_names"], meta["object_bboxes"]):
        if name.split(".")[0].lower() == "floor":
            b = np.asarray(box)
            out.append([b[0, 0], b[0, 1], b[1, 0], b[1, 1], b[0, 2]])
    return np.asarray(out, dtype=np.float64).reshape(-1, 5)


def scene_hash(scene: str) -> int:
    return int(scene.replace("-", "")[:8], 16)


@dataclass
class CandidateConfig:
    n: int = 64
    height: Sequence[float] = (1.5, 2.0)
    pitch_deg: Sequence[float] = (-60.0, 60.0)
    fov_deg: float = 60.0
    clearance: float = 0.1
    probe_res: int = 10  # official perform_obstacle_in_view_check: sqrt_number_of_rays=10
    max_tries_factor: int = 200


class CandidateGenerator:
    def __init__(self, scene: str, mesh: trimesh.Trimesh, meta: dict, patch_bbox: Sequence[float],
                 cfg: Optional[CandidateConfig] = None):
        self.scene, self.mesh, self.cfg = scene, mesh, cfg or CandidateConfig()
        self.bbox = np.asarray(patch_bbox, dtype=np.float64)
        floors = floor_boxes(meta)
        over = ((floors[:, 0] < self.bbox[3]) & (floors[:, 2] > self.bbox[0])
                & (floors[:, 1] < self.bbox[4]) & (floors[:, 3] > self.bbox[1]))
        self.floors = floors[over] if over.any() else floors
        area = (self.floors[:, 2] - self.floors[:, 0]) * (self.floors[:, 3] - self.floors[:, 1])
        self.floor_p = area / area.sum()
        self.probe = SceneRenderer(mesh, self.cfg.probe_res, self.cfg.probe_res, self.cfg.fov_deg)
        g = np.linspace(0, 1, 9)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        self.patch_pts = self.bbox[:3] + np.stack([gx, gy, gz], -1).reshape(-1, 3) * (self.bbox[3:] - self.bbox[:3])

    def _above_floor(self, pos: np.ndarray, floor_z: float) -> bool:
        locs, _, _ = self.mesh.ray.intersects_location([pos], [[0.0, 0.0, -1.0]], multiple_hits=True)
        return len(locs) > 0 and bool(np.any(np.abs(locs[:, 2] - floor_z) < 0.03))

    def frustum_hits_patch(self, c2w: np.ndarray) -> bool:
        p = (self.patch_pts - c2w[:3, 3]) @ c2w[:3, :3]  # camera coords
        z = -p[:, 2]
        t = np.tan(np.radians(self.cfg.fov_deg) / 2)
        inside = (z > 1e-3) & (np.abs(p[:, 0]) <= t * z) & (np.abs(p[:, 1]) <= t * z)
        return bool(inside.any())

    def check(self, c2w: np.ndarray, floor_z: Optional[float] = None) -> Dict[str, object]:
        """All validity checks for one camera (also used on official cameras)."""
        depth = self.probe.render_depth(c2w)
        res = {"no_background": bool((depth > 0).all()),
               "clearance_ok": bool(depth[depth > 0].min() >= self.cfg.clearance) if (depth > 0).any() else False,
               "frustum_hits_patch": self.frustum_hits_patch(c2w)}
        if floor_z is not None:
            res["above_floor"] = self._above_floor(c2w[:3, 3], floor_z)
        hits = self.probe.backproject(depth, c2w)
        inb = np.all((hits >= self.bbox[:3]) & (hits <= self.bbox[3:]), axis=1)
        res["visible_frac"] = float(inb.mean()) if len(hits) else 0.0  # probe rays whose first hit is in the patch
        return res

    def generate(self, episode: int, step: int, n: Optional[int] = None,
                 extra_check: Optional[Callable[[np.ndarray], bool]] = None) -> List[Dict[str, object]]:
        """extra_check(c2w) -> bool is an optional extra validity rule applied last, e.g. the
        exp1-task7 rule "after outside-patch occlusion at least one ray still reaches the patch".
        With None the output is identical to the task3 generator."""
        n = n or self.cfg.n
        rng = np.random.default_rng([GLOBAL_SEED, scene_hash(self.scene), episode, step])
        out, tries = [], 0
        while len(out) < n and tries < self.cfg.max_tries_factor * n:
            tries += 1
            f = self.floors[rng.choice(len(self.floors), p=self.floor_p)]
            pos = np.array([rng.uniform(f[0], f[2]), rng.uniform(f[1], f[3]), f[4] + rng.uniform(*self.cfg.height)])
            yaw, pitch = rng.uniform(0, 360), rng.uniform(*self.cfg.pitch_deg)
            c2w = look_matrix(pos, yaw, pitch)
            if not self.frustum_hits_patch(c2w):
                continue
            if not self._above_floor(pos, f[4]):
                continue
            r = self.check(c2w)
            if r["no_background"] and r["clearance_ok"] and (extra_check is None or extra_check(c2w)):
                out.append({"c2w": c2w, "yaw": yaw, "pitch": pitch, "height": pos[2] - f[4],
                            "visible_frac": r["visible_frac"], "tries": tries})
        return out

    def close(self):
        self.probe.close()
