"""Camera / LiDAR / mesh alignment check for UAVScenes (CPU, embree ray casting).

python -I check_alignment.py <cache.npz> <sampleinfos.json> <run_dir> <calib_py> <scene_calib_name> <out_dir> <downscale> <img_stem> [<img_stem> ...]

For each image stem:
  * ray-cast the mesh with the json pinhole K (no distortion) and T4x4 (cam->world, OpenCV axes)
    -> z-depth, hit mask, back-face fraction, vertex-colour render;
  * patch-wise phase correlation between the photo and the render (raw photo, and photo undistorted
    with the MARS-LVIG camera-LiDAR calibration distortion) -> pixel offsets centre vs border;
  * if a LiDAR scan exists: LiDAR -> camera (two conventions) -> world, ray-cast along each LiDAR
    ray from the sensor origin and compare the mesh range with the measured range.
The calibration file is parsed as a Python literal (ast), never executed.
"""
import ast
import json
import os
import sys
import glob

import numpy as np
import cv2
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache, js, run_dir, calib_py, calib_name, out, ds = sys.argv[1:8]
ds = int(ds)
stems = sys.argv[8:]
os.makedirs(out, exist_ok=True)

d = np.load(cache)
V, F, C = d["V"], d["F"], d["C"].astype(np.float32)
mesh = trimesh.Trimesh(V, F, process=False, validate=False)
rmi = RayMeshIntersector(mesh)
fn = mesh.face_normals

S = {s["OriginalImageName"]: s for s in json.load(open(js))}
tree = ast.parse(open(calib_py).read())
calib = None
for node in tree.body:
    if isinstance(node, ast.Assign) and node.targets[0].id == calib_name:
        calib = ast.literal_eval(node.value)
Kc = np.array(calib["camera_intrinsic"], float).reshape(3, 3)
dist = np.array(calib["camera_dist_coeffs"], float)
Rcl = np.array(calib["camera_ext_R"], float).reshape(3, 3)
tcl = np.array(calib["camera_ext_t"], float)


def cast(origins, dirs):
    tri, ray, loc = rmi.intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    return tri, ray, loc


def render(T, K, W, H):
    w, h = W // ds, H // ds
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    uf = (us + 0.5) * ds - 0.5
    vf = (vs + 0.5) * ds - 0.5
    x = (uf - K[0, 2]) / K[0, 0]
    y = (vf - K[1, 2]) / K[1, 1]
    dc = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)
    dw = dc @ T[:3, :3].T
    dw /= np.linalg.norm(dw, axis=1, keepdims=True)
    o = np.tile(T[:3, 3], (len(dw), 1))
    tri, ray, loc = cast(o, dw)
    z = np.full(len(dw), np.nan)
    pc = (loc - T[:3, 3]) @ T[:3, :3]  # world -> cam (R^T)
    z[ray] = pc[:, 2]
    back = np.zeros(len(dw), bool)
    back[ray] = (fn[tri] * dw[ray]).sum(1) > 0
    col = np.zeros((len(dw), 3), np.float32)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], loc)
    col[ray] = np.einsum("nk,nkc->nc", bary, C[F[tri]])
    return z.reshape(h, w), back.reshape(h, w), col.reshape(h, w, 3), (w, h)


def patch_shifts(a, b, n=(6, 5)):
    """phase-correlation shift of b w.r.t. a on an n-grid of patches (float gray)."""
    h, w = a.shape
    ph, pw = h // n[1], w // n[0]
    win = cv2.createHanningWindow((pw, ph), cv2.CV_64F)
    res = []
    for j in range(n[1]):
        for i in range(n[0]):
            pa = a[j * ph:(j + 1) * ph, i * pw:(i + 1) * pw]
            pb = b[j * ph:(j + 1) * ph, i * pw:(i + 1) * pw]
            (sx, sy), resp = cv2.phaseCorrelate(pa, pb, win)
            res.append([(i + 0.5) * pw, (j + 0.5) * ph, sx, sy, resp])
    return np.array(res)


def grad(g):
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    return np.sqrt(gx ** 2 + gy ** 2)


summary = []
for stem in stems:
    s = S[stem + ".jpg"]
    T = np.array(s["T4x4"], float)
    K = np.array(s["P3x3"], float)
    W, H = s["Width"], s["Height"]
    photo = cv2.imread(os.path.join(run_dir, "interval5_CAM", stem + ".jpg"))[:, :, ::-1]
    assert photo.shape[1] == W and photo.shape[0] == H, photo.shape
    z, back, col, (w, h) = render(T, K, W, H)
    hit = np.isfinite(z)
    rec = {"image": stem, "cam_xyz": T[:3, 3].tolist(), "render_wh": [w, h],
           "hit_frac": float(hit.mean()), "backface_frac_of_hits": float(back[hit].mean()),
           "zdepth_m_pct_1_50_99": np.nanpercentile(z, [1, 50, 99]).tolist()}
    # photo (raw) and photo undistorted with MARS-LVIG distortion, re-projected to json K
    raw_s = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    und = cv2.undistort(photo, Kc, dist, None, K)
    und_s = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
    rend_g = cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    for tag, ph in [("raw", raw_s), ("undist_marslvig", und_s)]:
        pg = cv2.cvtColor(ph, cv2.COLOR_RGB2GRAY).astype(np.float64)
        sh = patch_shifts(grad(rend_g), grad(pg))
        cx, cy = (K[0, 2] + 0.5) / ds - 0.5, (K[1, 2] + 0.5) / ds - 0.5
        rvec = sh[:, :2] - [cx, cy]
        rn = np.linalg.norm(rvec, axis=1)
        mag = np.linalg.norm(sh[:, 2:4], axis=1) * ds  # in full-res pixels
        radial = (sh[:, 2:4] * rvec).sum(1) / np.maximum(rn, 1e-6) * ds
        good = sh[:, 4] > 0.05
        inner = rn < np.percentile(rn, 40)
        rec[f"{tag}_patches_good"] = int(good.sum())
        rec[f"{tag}_shift_fullres_px_median"] = float(np.median(mag[good])) if good.any() else None
        rec[f"{tag}_shift_inner_px_median"] = float(np.median(mag[good & inner])) if (good & inner).any() else None
        rec[f"{tag}_shift_outer_px_median"] = float(np.median(mag[good & ~inner])) if (good & ~inner).any() else None
        rec[f"{tag}_radial_outer_px_median"] = float(np.median(radial[good & ~inner])) if (good & ~inner).any() else None
        rec[f"{tag}_phasecorr_response_median"] = float(np.median(sh[:, 4]))
        # affine fit: shift = A (p - c) + b  (full-res px); diag(A) = relative scale error in x / y
        if good.sum() >= 6:
            X = np.c_[rvec[good] * ds, np.ones(good.sum())]
            coef, *_ = np.linalg.lstsq(X, sh[good, 2:4] * ds, rcond=None)
            pred = X @ coef
            rec[f"{tag}_affine_scale_x_pct"] = float(coef[0, 0] * 100)
            rec[f"{tag}_affine_scale_y_pct"] = float(coef[1, 1] * 100)
            rec[f"{tag}_affine_shear_xy_yx_pct"] = [float(coef[1, 0] * 100), float(coef[0, 1] * 100)]
            rec[f"{tag}_affine_translation_px"] = coef[2].tolist()
            rec[f"{tag}_affine_residual_px_median"] = float(np.median(np.linalg.norm(sh[good, 2:4] * ds - pred, axis=1)))
        np.save(os.path.join(out, f"shifts_{tag}_{stem}.npy"), sh)
        if tag == "raw":
            sh_raw = sh
    # LiDAR range residual
    lid = glob.glob(os.path.join(run_dir, "interval5_LIDAR", f"image{stem}_lidar*.txt"))
    lid_pts_img = None
    if lid:
        P = np.loadtxt(lid[0])
        rng = np.linalg.norm(P, axis=1)
        P = P[rng > 1.0]
        rec["lidar_points"] = int(len(P))
        rec["lidar_range_m_pct_5_50_95"] = np.percentile(np.linalg.norm(P, axis=1), [5, 50, 95]).tolist()
        for conv, (pc, oc) in {"A_pc=R*pl+t": (P @ Rcl.T + tcl, tcl),
                                "B_pc=R^T(pl-t)": ((P - tcl) @ Rcl, -Rcl.T @ tcl)}.items():
            pw = pc @ T[:3, :3].T + T[:3, 3]
            ow = T[:3, :3] @ oc + T[:3, 3]
            dv = pw - ow
            meas = np.linalg.norm(dv, axis=1)
            dv /= meas[:, None]
            tri, ray, loc = cast(np.tile(ow, (len(dv), 1)), dv)
            mr = np.full(len(dv), np.nan)
            mr[ray] = np.linalg.norm(loc - ow, axis=1)
            res = mr - meas
            ok = np.isfinite(res)
            ar = np.abs(res[ok])
            rec[f"lidar_{conv}_hit_frac"] = float(ok.mean())
            rec[f"lidar_{conv}_absres_m_pct_50_90_95"] = np.percentile(ar, [50, 90, 95]).tolist() if ok.any() else None
            rec[f"lidar_{conv}_signed_res_median_m"] = float(np.median(res[ok])) if ok.any() else None
            rec[f"lidar_{conv}_frac_within_0.5m"] = float((ar < 0.5).mean()) if ok.any() else None
            rec[f"lidar_{conv}_frac_within_1m"] = float((ar < 1.0).mean()) if ok.any() else None
            if conv.startswith("A"):
                uvz = pc @ K.T
                inimg = pc[:, 2] > 0
                uv = uvz[:, :2] / uvz[:, 2:3]
                inimg &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                rec["lidar_A_frac_in_image"] = float(inimg.mean())
                lid_pts_img = (uv[inimg] / ds, np.where(ok, res, np.nan)[inimg])
    summary.append(rec)
    print(json.dumps(rec))

    # debug figure
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax[0, 0].imshow(raw_s); ax[0, 0].set_title(f"photo {stem} (raw)")
    ax[0, 1].imshow(np.clip(col, 0, 255).astype(np.uint8)); ax[0, 1].set_title("mesh vertex-colour render (json K, T4x4)")
    im = ax[0, 2].imshow(z, cmap="turbo"); plt.colorbar(im, ax=ax[0, 2], fraction=0.04, label="z-depth (m)")
    ax[0, 2].set_title(f"mesh z-depth; hit {hit.mean():.3f}, back-face {back[hit].mean():.4f}")
    pe = cv2.Canny(cv2.cvtColor(raw_s, cv2.COLOR_RGB2GRAY), 60, 150) > 0
    ov = np.clip(col, 0, 255).astype(np.uint8).copy()
    ov[pe] = [255, 0, 0]
    ax[1, 0].imshow(ov); ax[1, 0].set_title("photo edges (red) over mesh render")
    blend = (0.5 * raw_s.astype(float) + 0.5 * np.clip(col, 0, 255)).astype(np.uint8)
    ax[1, 1].imshow(blend)
    q = sh_raw
    ax[1, 1].quiver(q[:, 0], q[:, 1], q[:, 2] * 10, -q[:, 3] * 10, color="yellow", angles="xy", scale_units="xy", scale=1)
    ax[1, 1].set_title("50/50 blend; yellow = patch shift x10 (raw photo vs render)")
    if lid_pts_img is not None:
        uv, r = lid_pts_img
        ax[1, 2].imshow(raw_s)
        sc = ax[1, 2].scatter(uv[:, 0], uv[:, 1], c=np.clip(r, -2, 2), s=0.3, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar(sc, ax=ax[1, 2], fraction=0.04, label="mesh range - LiDAR range (m)")
        ax[1, 2].set_title("LiDAR (conv A) projected; colour = range residual")
    else:
        ax[1, 2].axis("off")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(out, f"align_{stem}.png"), dpi=80); plt.close()
    np.savez_compressed(os.path.join(out, f"render_{stem}.npz"), z=z.astype(np.float32), back=back)

json.dump(summary, open(os.path.join(out, "alignment_summary.json"), "w"), indent=1)
