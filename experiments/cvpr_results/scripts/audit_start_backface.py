"""Benchmark audit (magician env, 1 GPU, read-only): are the official start poses in free space?
For every runnable scene and official start k: the camera pose of service record t of the stored V3-T0-D1 episode, t in
{0, 10, 20, 50} (t = 0 is the official start pose, identical for every method; later steps show whether the episode stays there);
share of image pixels that show BACK faces of the mesh (camera inside / under the photogrammetry mesh; the simulator's renderer
does not cull them), share of sky, median depth. Same test as figs/fig2/scripts/s8_backface_check.py and fig4/fig4_backface.py.
Output: experiments/cvpr_results/manifest/start_pose_backface.csv.  Usage: $MAGICIAN_PYTHON audit_start_backface.py"""
import csv
import glob
import os
import sys

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_magician_replay as RP  # noqa: E402
import exp3b_paths as P  # noqa: E402

np, torch = RP.np, RP.torch
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings  # noqa: E402

RUN = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor", "V3-T0-D1", "V2M")
SCENES = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
STEPS = (0, 10, 20, 50)
dev = torch.device("cuda")
params = RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons", "macarons_default_training_config.json"))
rs = RasterizationSettings(image_size=(256, 456), blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=3000000)
out = os.path.join(REPO, "experiments", "cvpr_results", "manifest", "start_pose_backface.csv")
rows = []
for s in SCENES:
    obj = sorted(glob.glob(os.path.join(P.scene_dir(s), "*.obj")))[0]
    mesh = RP.load_scene(obj, params.scene_scale_factor, dev, mirror=False, mirrored_axis=None)
    v, f = mesh.verts_packed(), mesh.faces_packed()
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
    fc = v[f].mean(1)
    for k in range(5):
        for t in STEPS:
            r = np.load(os.path.join(RUN, "svc", f"V2M_{s}_s{k}_r0", f"svc_{s}_s{k}_t{t}.npz"))
            cam = FoVPerspectiveCameras(R=torch.as_tensor(r["R"][None], dtype=torch.float32, device=dev), T=torch.as_tensor(r["T"][None], dtype=torch.float32, device=dev), zfar=params.zfar, device=dev)
            with torch.no_grad():
                fr = MeshRasterizer(cameras=cam, raster_settings=rs)(mesh)
            p2f = fr.pix_to_face[0, ..., 0]
            hit = p2f >= 0
            c = cam.get_camera_center()[0]
            back = ((fn[p2f[hit]] * (fc[p2f[hit]] - c[None])).sum(-1) > 0).float().mean().item() if hit.any() else 0.0
            zb = fr.zbuf[0, ..., 0][hit]
            rows.append([s, k, t, f"{back:.4f}", f"{1.0 - hit.float().mean().item():.4f}", f"{zb.median().item():.3f}" if hit.any() else ""])
    print(s, [f"s{r[1]}t{r[2]}:{r[3]}" for r in rows[-20:] if float(r[3]) >= 0.05], flush=True)
    del mesh
    torch.cuda.empty_cache()
with open(out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["scene_id", "start_id", "step_of_V3-T0-D1_episode", "back_face_share", "sky_share", "median_depth"])
    w.writerows(rows)
print("written", out)
