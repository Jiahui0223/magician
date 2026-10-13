"""Offscreen depth rendering of a scene mesh with pyrender/EGL (exp1-task3, reused by task6).

Camera convention: cam2world matrices in the OpenGL/Blender convention (camera looks
along its local -z, +y is up). This is the convention of the official BlenderProc
cameras recovered by `seen2scene.active.camera.load_official_cameras`.
Depth is the linear z-depth (distance along the viewing axis); 0 means background.
Back-face culling must be off: 3D-FRONT walls/ceilings are single-sided, while Blender
(official renders) and the official BVH checks are double-sided. pyrender's DEPTH_ONLY and SEG
paths always enable GL_CULL_FACE and ignore SKIP_CULL_FACES (pyrender/renderer.py:519-594).
So we render with FLAT | SKIP_CULL_FACES and read the depth buffer. With DEPTH_ONLY, the ceiling
disappeared and 56% of the official p_0.1 cameras wrongly saw "background" (exp1-task3).
"""
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402


_RENDERERS = {}


def _shared_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    """One OffscreenRenderer per resolution for the whole process, never deleted.

    All pyrender EGL renderers share one display. Deleting any of them, explicitly or
    through __del__ at garbage collection, calls eglTerminate and breaks every renderer
    still alive (EGL_NOT_INITIALIZED, seen in exp1-task4). One renderer can render any
    number of pyrender Scenes, so we share it.
    """
    key = (width, height)
    if key not in _RENDERERS:
        _RENDERERS[key] = pyrender.OffscreenRenderer(width, height)
    return _RENDERERS[key]


class SceneRenderer:
    def __init__(self, mesh: trimesh.Trimesh, width: int = 512, height: int = 512, fov_deg: float = 60.0,
                 znear: float = 0.01, zfar: float = 100.0):
        self.width, self.height, self.fov = width, height, np.deg2rad(fov_deg)
        self.scene = pyrender.Scene()
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        cam = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=width / height, znear=znear, zfar=zfar)
        self.cam_node = self.scene.add(cam, pose=np.eye(4))
        self.renderer = _shared_renderer(width, height)

    def render_depth(self, cam2world: np.ndarray) -> np.ndarray:
        self.scene.set_pose(self.cam_node, pose=cam2world)
        flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES
        _, depth = self.renderer.render(self.scene, flags=flags)
        return depth.astype(np.float32)

    def backproject(self, depth: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
        """World-space hit points of all pixels with depth > 0, shape [M, 3]."""
        h, w = depth.shape
        t = np.tan(self.fov / 2)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (2 * (u + 0.5) / w - 1) * t * (w / h)
        y = (1 - 2 * (v + 0.5) / h) * t
        m = depth > 0
        d = depth[m]
        pts_cam = np.stack([x[m] * d, y[m] * d, -d], axis=1)
        return pts_cam @ cam2world[:3, :3].T + cam2world[:3, 3]

    def close(self):
        # The OffscreenRenderer is shared and deliberately never deleted, see _shared_renderer.
        self.scene = None
