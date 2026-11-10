"""CVPR Fig. 4 -- validity check of the candidate states (magician env, 1 GPU, read-only): for the EXECUTED view of every pool state
(pose of service record t + 1 of the stored V3-T0-D1 episode) the share of image pixels that show BACK faces of the mesh (camera
inside / under the photogrammetry mesh; the simulator does not cull them) and the share of sky. States with >= 5 % back faces are
not eligible as figure examples (same test as figs/fig2/scripts/s8_backface_check.py; found there for neuschwanstein starts 2 / 4).
Also stored (amendment of 2026-11-10 05:40, see fig4_plot.py): the back-face share of the CURRENT view (record t) and the largest
share over the already observed views at steps 0, 10, ..., t, so that states whose map P_t was built from inside the mesh can be excluded.
Usage: $MAGICIAN_PYTHON fig4_backface.py --scene S   ->  work_dirs/cvpr_results/overnight_20261110/intermediates/fig4/backface_<scene>.jsonl"""
import argparse
import glob
import json
import os
import sys

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_magician_replay as RP  # noqa: E402
import exp3b_paths as P  # noqa: E402

np, torch = RP.np, RP.torch
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings  # noqa: E402

RUN = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor", "V3-T0-D1", "V2M")
OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "overnight_20261110", "intermediates", "fig4")
STEPS = (0, 10, 20, 30, 40, 50)

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
a = ap.parse_args()
dev = torch.device("cuda")
params = RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
obj = sorted(glob.glob(os.path.join(P.scene_dir(a.scene), "*.obj")))[0]
mesh = RP.load_scene(obj, params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)
v, f = mesh.verts_packed(), mesh.faces_packed()
fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
fc = v[f].mean(1)
rs = RasterizationSettings(image_size=(256, 456), blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=3000000)
with open(os.path.join(OUT, f"backface_{a.scene}.jsonl"), "w") as fo:
    for k in range(5):
        for t in STEPS:
            res = {}
            for name, tt in (("next", t + 1), ("cur", t)):
                r = np.load(os.path.join(RUN, "svc", f"V2M_{a.scene}_s{k}_r0", f"svc_{a.scene}_s{k}_t{tt}.npz"))
                cam = FoVPerspectiveCameras(R=torch.as_tensor(r["R"][None], dtype=torch.float32, device=dev), T=torch.as_tensor(r["T"][None], dtype=torch.float32, device=dev), zfar=params.zfar, device=dev)
                with torch.no_grad():
                    p2f = MeshRasterizer(cameras=cam, raster_settings=rs)(mesh).pix_to_face[0, ..., 0]
                hit = p2f >= 0
                c = cam.get_camera_center()[0]
                res[name] = (((fn[p2f[hit]] * (fc[p2f[hit]] - c[None])).sum(-1) > 0).float().mean().item() if hit.any() else 0.0, 1.0 - hit.float().mean().item())
            # back_share_hist_max: the largest back-face share over the views already observed at the refresh steps 0, 10, ..., t (the views that built P_t, sampled every 10 steps)
            hist = []
            for tt in range(0, t + 1, 10):
                r = np.load(os.path.join(RUN, "svc", f"V2M_{a.scene}_s{k}_r0", f"svc_{a.scene}_s{k}_t{tt}.npz"))
                cam = FoVPerspectiveCameras(R=torch.as_tensor(r["R"][None], dtype=torch.float32, device=dev), T=torch.as_tensor(r["T"][None], dtype=torch.float32, device=dev), zfar=params.zfar, device=dev)
                with torch.no_grad():
                    p2f = MeshRasterizer(cameras=cam, raster_settings=rs)(mesh).pix_to_face[0, ..., 0]
                hit = p2f >= 0
                c = cam.get_camera_center()[0]
                hist.append(((fn[p2f[hit]] * (fc[p2f[hit]] - c[None])).sum(-1) > 0).float().mean().item() if hit.any() else 0.0)
            row = {"scene": a.scene, "start": k, "t": t, "back_share_next": res["next"][0], "sky_share_next": res["next"][1], "back_share_cur": res["cur"][0], "back_share_hist_max": max(hist)}
            fo.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
