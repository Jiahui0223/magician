"""EXP5 Task 0.3 -- Matterport3D geometry / coordinate validation (magician env: trimesh + PyTorch3D; CPU + 1 GPU).

  conventions : which reading of undistorted_camera_parameters (*.conf, camera-to-world 4x4 + intrinsics) puts the
                back-projected undistorted depth (16-bit PNG, 0.25 mm, z-depth) onto the house mesh. Candidates:
                image row origin top / bottom x camera y down (OpenCV, look +z) / y up (OpenGL, look -z).
                Metric: distance of back-projected points to dense mesh surface samples (KD-tree), per frame.
  render      : with the selected convention, rasterise the mesh(es) at the real camera (custom intrinsics, PyTorch3D)
                and compare with the captured depth pixel by pixel: agreement, render-nearer (mesh surface in front of
                the sensor reading), render-farther / no-hit (mesh holes, glass, missing geometry), sensor invalid.
Outputs W/preprocessing/geometry/<scan>/{conventions,render}.json (+ figures).
Data are read from /bsuhome/jiahuizhang/scratch/data/matterport3d (read-only use).
"""
import argparse
import glob
import json
import os
import time

import numpy as np

MP = "/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans"
W = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0"
DEPTH_SCALE = 4000.0          # data_organization.md: 0.25 mm per value


def read_conf(scan):
    """[(depth_name, color_name, K 3x3, c2w 4x4)] in file order."""
    p = os.path.join(MP, scan, "undistorted_camera_parameters", f"{scan}.conf")
    out, K = [], None
    for line in open(p):
        f = line.split()
        if not f:
            continue
        if f[0] == "intrinsics_matrix":
            K = np.array(list(map(float, f[1:10]))).reshape(3, 3)
        elif f[0] == "scan":
            out.append((f[1], f[2], K.copy(), np.array(list(map(float, f[3:19]))).reshape(4, 4)))
    return out


def read_depth(scan, name):
    from PIL import Image
    d = np.asarray(Image.open(os.path.join(MP, scan, "undistorted_depth_images", name)), dtype=np.float64)
    return d / DEPTH_SCALE                                  # metres, 0 = no reading


def house_mesh(scan):
    import trimesh
    m = trimesh.load(os.path.join(MP, scan, "house_segmentations", f"{scan}.ply"), process=False)
    return m


def tex_mesh(scan):
    import trimesh
    p = glob.glob(os.path.join(MP, scan, "matterport_mesh", "*", "*.obj"))[0]
    m = trimesh.load(p, process=False, force="mesh", skip_materials=True)
    return m


CONVENTIONS = {
    "cv_top": dict(flip_rows=False, y_down=True),      # OpenCV camera, image origin top-left
    "cv_bottom": dict(flip_rows=True, y_down=True),
    "gl_top": dict(flip_rows=False, y_down=False),     # OpenGL camera (y up, looks -z)
    "gl_bottom": dict(flip_rows=True, y_down=False),   # OpenGL camera, image origin bottom-left (data_organization.md)
}


def backproject(depth, K, c2w, conv, stride=8):
    H, Wd = depth.shape
    r, c = np.mgrid[0:H:stride, 0:Wd:stride]
    z = depth[r, c]
    ok = z > 0
    r, c, z = r[ok], c[ok], z[ok]
    v = (H - 1 - r) if conv["flip_rows"] else r
    x = (c - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1) if conv["y_down"] else np.stack([x, y, -z], 1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def mode_conventions(scan, n_frames=24, seed=0):
    from scipy.spatial import cKDTree
    import trimesh
    m = house_mesh(scan)
    pts, _ = trimesh.sample.sample_surface(m, 3_000_000, seed=seed)
    tree = cKDTree(pts)
    frames = read_conf(scan)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(frames), min(n_frames, len(frames)), replace=False))
    res = {k: [] for k in CONVENTIONS}
    for i in idx:
        dn, _, K, c2w = frames[i]
        d = read_depth(scan, dn)
        for k, conv in CONVENTIONS.items():
            p = backproject(d, K, c2w, conv)
            dist, _ = tree.query(p, k=1)
            res[k].append({"frame": dn, "n": int(len(p)), "median_m": float(np.median(dist)),
                           "frac_lt_2cm": float((dist < 0.02).mean()), "frac_lt_5cm": float((dist < 0.05).mean())})
    summ = {k: {"median_m": float(np.median([x["median_m"] for x in v])),
                "frac_lt_2cm": float(np.mean([x["frac_lt_2cm"] for x in v])),
                "frac_lt_5cm": float(np.mean([x["frac_lt_5cm"] for x in v]))} for k, v in res.items()}
    best = min(summ, key=lambda k: summ[k]["median_m"])
    bb = m.bounds
    out = {"scan": scan, "frames_total": len(frames), "frames_tested": [frames[i][0] for i in idx],
           "mesh": {"faces": int(len(m.faces)), "bounds_m": bb.tolist(), "extent_m": (bb[1] - bb[0]).tolist()},
           "intrinsics_unique": sorted({tuple(np.round(f[2][[0, 1, 0, 1], [0, 1, 2, 2]], 3)) for f in frames}),
           "camera_height_m": {"min": float(min(f[3][2, 3] for f in frames)), "max": float(max(f[3][2, 3] for f in frames))},
           "summary": summ, "selected": best, "per_frame": res}
    od = os.path.join(W, "preprocessing", "geometry", scan)
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "conventions.json"), "w"), indent=1, default=float)
    print(scan, json.dumps(summ), "selected", best)


def render_depth(mesh_v, mesh_f, K, c2w, H, Wd, conv, dev):
    """Rasterised z-depth (metres, inf = no hit) at the real camera, image rows in the PNG order."""
    import torch
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes
    w2c = np.linalg.inv(c2w)
    # camera frame in OpenCV axes (x right, y down, z forward)
    Rc = w2c[:3, :3].copy()
    tc = w2c[:3, 3].copy()
    if not conv["y_down"]:
        Rc[1:] *= -1
        tc[1:] *= -1
    # PyTorch3D view frame: x left, y up, z forward; X_view = X_world @ R + T
    flip = np.diag([-1.0, -1.0, 1.0])
    R = (flip @ Rc).T
    T = flip @ tc
    cy = (H - 1 - K[1, 2]) if conv["flip_rows"] else K[1, 2]
    cam = PerspectiveCameras(focal_length=((K[0, 0], K[1, 1]),), principal_point=((K[0, 2], cy),),
                             R=torch.tensor(R[None], dtype=torch.float32), T=torch.tensor(T[None], dtype=torch.float32),
                             in_ndc=False, image_size=((H, Wd),), device=dev)
    ms = Meshes(verts=[mesh_v], faces=[mesh_f])
    rs = RasterizationSettings(image_size=(H, Wd), faces_per_pixel=1, blur_radius=0.0, cull_backfaces=False,
                               max_faces_per_bin=int(len(mesh_f)), bin_size=None)
    frag = MeshRasterizer(cameras=cam, raster_settings=rs)(ms)
    z = frag.zbuf[0, ..., 0].cpu().numpy().astype(np.float64)
    z[frag.pix_to_face[0, ..., 0].cpu().numpy() < 0] = np.inf
    # rows already follow the PNG: with a bottom-left image origin the OpenCV-equivalent principal point is H-1-cy
    return z


def mode_render(scan, conv_name, n_frames=24, seed=0, which="house", down=2):
    import torch
    dev = torch.device("cuda")
    conv = CONVENTIONS[conv_name]
    m = house_mesh(scan) if which == "house" else tex_mesh(scan)
    mv = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
    mf = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev)
    frames = read_conf(scan)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(frames), min(n_frames, len(frames)), replace=False))
    rows = []
    for i in idx:
        dn, _, K, c2w = frames[i]
        d = read_depth(scan, dn)
        H, Wd = d.shape
        Kd = K.copy()
        Kd[:2] /= down
        Kd[0, 2] = (K[0, 2] + 0.5) / down - 0.5
        Kd[1, 2] = (K[1, 2] + 0.5) / down - 0.5
        ds = d[down // 2::down, down // 2::down]
        z = render_depth(mv, mf, Kd, c2w, ds.shape[0], ds.shape[1], conv, dev)
        valid = ds > 0
        hit = np.isfinite(z)
        both = valid & hit
        e = np.abs(z - ds)
        rel = e / np.maximum(ds, 1e-6)
        rows.append({"frame": dn, "sensor_valid": float(valid.mean()), "render_hit": float(hit.mean()),
                     "agree_2cm": float((e[both] < 0.02).mean()) if both.any() else None,
                     "agree_rel1pct": float((rel[both] < 0.01).mean()) if both.any() else None,
                     "median_abs_m": float(np.median(e[both])) if both.any() else None,
                     "render_nearer_10cm": float((z[both] < ds[both] - 0.10).mean()) if both.any() else None,
                     "render_farther_10cm": float((z[both] > ds[both] + 0.10).mean()) if both.any() else None,
                     "sensor_valid_render_nohit": float((valid & ~hit).mean()),
                     "sensor_invalid_render_hit": float((~valid & hit).mean()),
                     "sensor_depth_p50_p95_max": [float(np.percentile(ds[valid], 50)), float(np.percentile(ds[valid], 95)),
                                                  float(ds[valid].max())] if valid.any() else None})
        if len(rows) <= 3:
            save_fig(scan, which, dn, ds, z)
    keys = [k for k in rows[0] if k not in ("frame", "sensor_depth_p50_p95_max")]
    summ = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}
    out = {"scan": scan, "mesh": which, "convention": conv_name, "downsample": down, "summary": summ, "per_frame": rows}
    od = os.path.join(W, "preprocessing", "geometry", scan)
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"render_{which}.json"), "w"), indent=1, default=float)
    print(scan, which, json.dumps({k: round(v, 4) for k, v in summ.items()}))


def save_fig(scan, which, name, ds, z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    od = os.path.join(W, "preprocessing", "geometry", scan, "figs")
    os.makedirs(od, exist_ok=True)
    vmax = np.percentile(ds[ds > 0], 98) if (ds > 0).any() else 5
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].imshow(np.where(ds > 0, ds, np.nan), vmin=0, vmax=vmax)
    ax[0].set_title("captured depth (m)")
    ax[1].imshow(np.where(np.isfinite(z), z, np.nan), vmin=0, vmax=vmax)
    ax[1].set_title(f"rendered {which} mesh")
    diff = np.where((ds > 0) & np.isfinite(z), z - ds, np.nan)
    im = ax[2].imshow(diff, vmin=-0.2, vmax=0.2, cmap="coolwarm")
    ax[2].set_title("render - captured (m)")
    plt.colorbar(im, ax=ax[2])
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(od, f"{which}_{name[:-4]}.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["conventions", "render"])
    ap.add_argument("--scan", required=True)
    ap.add_argument("--conv", default=None)
    ap.add_argument("--mesh", default="house", choices=["house", "textured"])
    ap.add_argument("--frames", type=int, default=24)
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "conventions":
        mode_conventions(a.scan, a.frames)
    else:
        mode_render(a.scan, a.conv, a.frames, which=a.mesh)
    print(f"{time.time() - t0:.1f} s")
