"""Cross-check: PyTorch3D (CPU) z-buffer vs embree z-depth on the UAVScenes mesh, using the
Macarons++/MAGICIAN camera model (FoVPerspectiveCameras, fov 60 on the short side, 456x256, y-up).

python -I p3d_crosscheck.py <cache.npz> <proto_render_zdepth.npy> <prototype_grid.json> <out_dir> [scale]

The mesh (ENU z-up, metres) is converted to y-up (x, z, -y) and multiplied by `scale`
(runtime units per metre).  Faces farther than 400 m from the camera are culled for CPU speed.
Run with CUDA_VISIBLE_DEVICES="" (CPU only).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer, look_at_view_transform

cache, ref_npy, proto_json, out = sys.argv[1:5]
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
assert not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only"
d = np.load(cache)
V, F = d["V"], d["F"]
st = json.load(open(proto_json))
c0 = np.array(st["render_cam"])
el = np.radians(-60)
fwd = np.array([0, np.cos(el), np.sin(el)])
keep = np.linalg.norm(V[F].mean(1) - c0, axis=1) < 400
Fk = F[keep]
to_yup = lambda p: np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1) * scale
Vy = torch.tensor(to_yup(V), dtype=torch.float32)
mesh = Meshes([Vy], [torch.tensor(Fk, dtype=torch.int64)])
eye = torch.tensor(to_yup(c0)[None], dtype=torch.float32)
at = torch.tensor(to_yup(c0 + fwd)[None], dtype=torch.float32)
R, T = look_at_view_transform(eye=eye, at=at, up=((0, 1, 0),))
cams = FoVPerspectiveCameras(R=R, T=T, fov=60.0, znear=1.0 * scale, zfar=750.0)
rs = RasterizationSettings(image_size=(256, 456), blur_radius=0.0, faces_per_pixel=1, bin_size=None,
                           max_faces_per_bin=400000)
t0 = time.time()
frag = MeshRasterizer(cameras=cams, raster_settings=rs)(mesh)
dt = time.time() - t0
z = frag.zbuf[0, :, :, 0].numpy()
z = np.where(z > 0, z / scale, np.nan)
ref = np.load(ref_npy)
both = np.isfinite(z) & np.isfinite(ref)
diff = np.abs(z - ref)[both]
# left-right flip check (camera handedness)
diff_flip = np.abs(z[:, ::-1] - ref)[both]
res = {"scale_units_per_m": scale, "faces_kept": int(len(Fk)), "p3d_cpu_raster_s": dt,
       "p3d_hit_frac": float(np.isfinite(z).mean()), "embree_hit_frac": float(np.isfinite(ref).mean()),
       "absdiff_m_pct_50_95_99": np.percentile(diff, [50, 95, 99]).tolist(),
       "frac_absdiff_lt_0.1m": float((diff < 0.1).mean()),
       "absdiff_if_lr_flipped_median_m": float(np.median(diff_flip))}
json.dump(res, open(os.path.join(out, f"p3d_crosscheck_s{scale:g}.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
np.save(os.path.join(out, f"p3d_zbuf_s{scale:g}.npy"), z)
