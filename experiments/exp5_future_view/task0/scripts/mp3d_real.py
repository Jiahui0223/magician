"""EXP5 Task 0 -- real-captured view protocol (R) helpers (magician env; simulator / evaluation side).

A panorama's 18 undistorted depth frames (3 cameras x 6 yaws; 16-bit PNG / 4000 = z-depth in m; camera-to-world with
the OpenGL camera and a bottom-left image origin, T0.3 F6) are back-projected to world points (stride 2) and
forward-projected with a z-buffer (minimum view depth per pixel) into
  - the frozen virtual sensor (456 x 256, FoVPerspectiveCameras(R, T)) at the virtual pose -> R-protocol zbuf / mask, or
  - the 64 x 114 scoring-ray grid (EXP5 T1.1 label cameras: pixel centres = scoring rays) -> real future-depth labels.
Pixels / rays without a real sample stay empty (zbuf -1 / no label). Points beyond 10 m (the sensor range) are kept;
the frozen pipeline applies its own range.
"""
import os
from collections import OrderedDict

import numpy as np

import mp3d_sim as S

STRIDE = 2


def read_conf(scan):
    """{panorama uuid: [(depth png, K 3x3, c2w 4x4)]} from the undistorted camera parameters."""
    out, K = {}, None
    for line in open(os.path.join(S.MP, scan, "undistorted_camera_parameters", f"{scan}.conf")):
        f = line.split()
        if not f:
            continue
        if f[0] == "intrinsics_matrix":
            K = np.array(list(map(float, f[1:10]))).reshape(3, 3)
        elif f[0] == "scan":
            out.setdefault(f[1].split("_d")[0], []).append((f[1], K.copy(), np.array(list(map(float, f[3:19]))).reshape(4, 4)))
    return out


class RealPanos:
    """World points (runtime y-up) of the real depth frames of a panorama, LRU-cached on the GPU."""

    def __init__(self, scan, device, cache=6):
        from PIL import Image
        self.scan, self.dev, self.cache, self.Image = scan, device, OrderedDict(), Image
        self.conf = read_conf(scan)
        self.n_cache = cache

    def points(self, pano_name):
        import torch
        if pano_name in self.cache:
            self.cache.move_to_end(pano_name)
            return self.cache[pano_name]
        pts = []
        for name, K, c2w in self.conf[pano_name]:
            d = np.asarray(self.Image.open(os.path.join(S.MP, self.scan, "undistorted_depth_images", name)), np.float64) / 4000.0
            H, Wd = d.shape
            r, c = np.mgrid[0:H:STRIDE, 0:Wd:STRIDE]
            z = d[r, c]
            ok = z > 0
            r, c, z = r[ok], c[ok], z[ok]
            v = H - 1 - r                                           # bottom-left image origin
            x = (c - K[0, 2]) / K[0, 0] * z
            y = (v - K[1, 2]) / K[1, 1] * z
            pc = np.stack([x, y, -z], 1)                           # OpenGL camera: looks along -z
            pts.append(pc @ c2w[:3, :3].T + c2w[:3, 3])
        P = torch.as_tensor(S.zup_m_to_rt(np.concatenate(pts)), dtype=torch.float32, device=self.dev)
        self.cache[pano_name] = P
        if len(self.cache) > self.n_cache:
            self.cache.popitem(last=False)
        return P


def zbuffer(cam, pts, H, Wd):
    """Minimum view depth per pixel of the points projected with a PyTorch3D camera (screen pixels, centre = +0.5)."""
    import torch
    sp = cam.transform_points_screen(pts[None], image_size=((H, Wd),))[0]
    x, y, z = sp[:, 0], sp[:, 1], cam.get_world_to_view_transform().transform_points(pts[None])[0][:, 2]
    j, i = torch.floor(x).long(), torch.floor(y).long()
    ok = (z > 0) & (i >= 0) & (i < H) & (j >= 0) & (j < Wd)
    zb = torch.full((H * Wd,), float("inf"), device=pts.device)
    zb.scatter_reduce_(0, (i[ok] * Wd + j[ok]), z[ok], reduce="amin", include_self=True)
    return zb.view(H, Wd)


def real_frame(sensor, real, pano_name, R, T):
    """R-protocol observation at a virtual pose: the frozen-format frame dict (zbuf -1 = no real sample)."""
    import torch
    fov = sensor.cam(R, T)
    zb = zbuffer(fov, real.points(pano_name), S.H_IMG, S.W_IMG)
    zbuf = torch.where(torch.isfinite(zb), zb, torch.full_like(zb, -1.0)).view(1, S.H_IMG, S.W_IMG, 1)
    return {"zbuf": zbuf, "mask": zbuf > -1, "fov": fov, "R": R, "T": T}
