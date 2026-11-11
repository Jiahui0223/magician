"""EXP7 / UAVScenes -- convert one UAVScenes mesh tile into a Macarons++-format scene folder (CPU, magician env, python -I).

Protocol UAV-v0 (deterministic, the same for every method; Macarons++ conventions):
  * scale s = 0.5 runtime units per metre (1 runtime unit = 2 m; Macarons++ scenes span ~1.2-3.5 m per runtime unit);
    OBJ units = runtime / 10 (MAGICIAN scene_scale_factor 10). Frame: runtime y-up, x = east, y = up, z = -north, origin = tile
    centre (east, north) and the tile ground level (1st percentile of the 2 m max-height map inside the tile).
  * mesh = all faces whose centroid lies in the tile box grown by `margin` m (context, like the Macarons++ terrain that extends
    beyond the scene box), vertex clustering with the smallest cell of {0.3, 0.4, 0.5, 0.6, 0.8} m that gives <= 1.6 M faces (mean position /
    colour per cell, degenerate and duplicate faces dropped), i.e. about the face count of a Macarons++ scene (1.4-1.5 M); per-face colour texture atlas (colour only).
  * scene box (GT surface + coverage domain): the tile rectangle; vertically ground_p0.5 - 2 m .. max(top_p99.9 + 20 m,
    longest side / 3) so the horizontal / vertical aspect stays within the Macarons++ range (bridge 5.6).
  * camera box = scene box horizontally; vertically scene floor + 5 m .. scene top; pose step ~10 runtime units (20 m) per axis
    (pose_l / w / h = round(extent / 10), >= 2); 5 elevations x 10 azimuths (all Macarons++ outdoor scenes); contrast 1.0.
  * coverage cells: grid = round(extent / 20 runtime), cell_capacity 1000, cell_resolution 0.05 (eps = 1 runtime unit = 2 m).
  * visibility_ratio, reachable / occupied poses and the start positions are written by uav_tile_gpu.py (needs the renderer).
Usage: python -I uav_build_tile.py --scene HKisland --tile S --box X0 Y0 X1 Y1 [--s 0.5] [--margin 60] [--cluster 0.3]"""
import argparse
import hashlib
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
DL = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "downloads", "uavscenes")
ROOT = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")      # Macarons++-format data root for UAV tiles


def cluster_mesh(V, F, C, cell):
    key = np.floor(V / cell).astype(np.int64)
    key -= key.min(0)
    k1 = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    uk, inv = np.unique(k1, return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    V2 = np.stack([np.bincount(inv, weights=V[:, a], minlength=n) for a in range(3)], 1) / cnt[:, None]
    C2 = np.stack([np.bincount(inv, weights=C[:, a].astype(np.float64), minlength=n) for a in range(3)], 1) / cnt[:, None]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[ok]
    s = np.sort(F2, 1)
    _, first = np.unique(s, axis=0, return_index=True)
    F2 = F2[np.sort(first)]
    used = np.unique(F2)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return V2[used], remap[F2], np.clip(C2[used], 0, 255).astype(np.uint8)


def write_obj(path, name, Vobj, F, C):
    from PIL import Image
    nF = len(F)
    T = int(np.ceil(np.sqrt(nF)))
    fc = C[F].mean(1).astype(np.uint8)                                   # per-face colour
    img = np.zeros((T * T, 3), np.uint8)
    img[:nF] = fc
    Image.fromarray(img.reshape(T, T, 3)[::-1]).save(os.path.join(os.path.dirname(path), f"{name}_tex.png"))
    j = np.arange(nF)
    uv = np.stack([(j % T + 0.5) / T, (j // T + 0.5) / T], 1)             # one texel per face (v up: image rows flipped)
    with open(os.path.join(os.path.dirname(path), f"{name}.mtl"), "w") as f:
        f.write(f"# EXP7 UAVScenes tile {name}: per-face colour atlas from Terra vertex colours\nnewmtl material_0\nKa 1.0 1.0 1.0\n"
                f"Kd 0.8 0.8 0.8\nKs 0.0 0.0 0.0\nd 1.0\nillum 1\nmap_Kd {name}_tex.png\n")
    with open(path, "w") as f:
        f.write(f"# EXP7 UAVScenes tile {name} (Macarons++ format, OBJ units = runtime / 10)\nmtllib {name}.mtl\n")
        np.savetxt(f, Vobj, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, uv, fmt="vt %.7f %.7f")
        f.write("usemtl material_0\n")
        a = np.concatenate([F + 1, (j + 1)[:, None]], 1)                  # v / vt indices (1-based); vt of face f = f
        np.savetxt(f, a[:, [0, 3, 1, 3, 2, 3]], fmt="f %d/%d %d/%d %d/%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tile", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True, help="ENU metres: east0 north0 east1 north1")
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=60.0)
    ap.add_argument("--cluster", type=float, default=0.3)
    ap.add_argument("--max-faces", type=int, default=1_600_000)
    a = ap.parse_args()
    name = f"uav_{a.scene}_{a.tile}"
    od = os.path.join(ROOT, name)
    os.makedirs(od, exist_ok=False)
    z = np.load(os.path.join(DL, "derived", f"{a.scene}_mesh.npz"))
    V, F, C = z["V"], z["F"], z["C"]
    x0, y0, x1, y1 = a.box
    cen = V[F].mean(1)
    m = a.margin
    keep = (cen[:, 0] >= x0 - m) & (cen[:, 0] <= x1 + m) & (cen[:, 1] >= y0 - m) & (cen[:, 1] <= y1 + m)
    for cl in [c for c in (0.3, 0.4, 0.5, 0.6, 0.8) if c >= a.cluster]:     # smallest cell giving <= max_faces (Macarons++ size)
        Vc, Fc, Cc = cluster_mesh(V, F[keep], C, cl)
        if len(Fc) <= a.max_faces:
            break
    a.cluster = cl
    # ground / top from the 2 m max-height map inside the tile
    hm = np.load(os.path.join(DL, "derived", f"{a.scene}_heightmap_2m.npy"))
    info0 = json.load(open(os.path.join(DL, "derived", f"{a.scene}_mesh_info.json")))
    lo = np.asarray(info0["bbox_min_m"])
    ix0, ix1 = int((x0 - lo[0]) / 2.0), int(np.ceil((x1 - lo[0]) / 2.0))
    iy0, iy1 = int((y0 - lo[1]) / 2.0), int(np.ceil((y1 - lo[1]) / 2.0))
    sub = hm[max(iy0, 0):iy1, max(ix0, 0):ix1]
    fp = float(np.isfinite(sub).mean())
    g01, g005, t999 = np.nanpercentile(sub, [1, 0.5, 99.9])
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    s = a.s
    to_rt = lambda P: np.stack([s * (P[:, 0] - cx), s * (P[:, 2] - g01), -s * (P[:, 1] - cy)], 1)
    Vrt = to_rt(Vc)
    ext_x, ext_z = s * (x1 - x0), s * (y1 - y0)
    y_lo = s * (g005 - g01) - s * 2.0
    y_hi = max(s * (t999 - g01) + s * 20.0, y_lo + max(ext_x, ext_z) / 3.0)
    sc_min = np.array([-ext_x / 2, y_lo, -ext_z / 2])
    sc_max = np.array([ext_x / 2, y_hi, ext_z / 2])
    cam_min = np.array([sc_min[0], y_lo + s * 5.0, sc_min[2]])
    cam_max = sc_max.copy()
    ce = cam_max - cam_min
    pose = [max(2, int(round(e / 10.0))) for e in ce]
    se = sc_max - sc_min
    grid = [max(1, int(round(e / 20.0))) for e in se]
    settings = {"scene": {"grid_l": grid[0], "grid_w": grid[1], "grid_h": grid[2], "cell_capacity": 1000, "cell_resolution": 0.05,
                          "x_min": (sc_min / 10).tolist(), "x_max": (sc_max / 10).tolist(), "visibility_ratio": None},
                "camera": {"pose_l": pose[0], "pose_w": pose[1], "pose_h": pose[2], "pose_n_theta": 5, "pose_n_azim": 10,
                           "x_min": (cam_min / 10).tolist(), "x_max": (cam_max / 10).tolist(), "start_positions": None, "contrast_factor": 1.0}}
    write_obj(os.path.join(od, f"{name}.obj"), name, Vrt / 10.0, Fc, Cc)
    json.dump(settings, open(os.path.join(od, "settings.pending.json"), "w"), indent=1)
    meta = {"protocol": "UAV-v0", "source_scene": a.scene, "tile": a.tile, "box_enu_m": a.box, "s_runtime_per_m": s, "margin_m": m,
            "cluster_m": a.cluster, "origin_enu_m": [cx, cy, float(g01)], "frame": "runtime y-up: x = s(E - cx), y = s(U - ground), z = -s(N - cy)",
            "faces_source_in_crop": int(keep.sum()), "faces_tile": int(len(Fc)), "vertices_tile": int(len(Vc)), "tile_footprint_fraction": fp,
            "ground_p01_m": float(g01), "ground_p005_m": float(g005), "top_p999_m": float(t999),
            "scene_box_runtime": [sc_min.tolist(), sc_max.tolist()], "camera_box_runtime": [cam_min.tolist(), cam_max.tolist()],
            "pose_grid": pose, "coverage_grid": grid, "aspect_horizontal_over_height": float(max(ext_x, ext_z) / (y_hi - y_lo)),
            "k_M": float((y_hi - y_lo) / 2.816), "dims_scene_est": [int(round(e / ((y_hi - y_lo) / 2.816) / 0.011)) for e in (ext_x, y_hi - y_lo, ext_z)],
            "mesh_sha256": hashlib.sha256(open(os.path.join(od, f"{name}.obj"), "rb").read()).hexdigest(),
            "source_mesh": os.path.relpath(os.path.join(DL, "raw", "terra_3dmap_pointcloud_mesh", a.scene, "Mesh.ply"), R), "license": "CC BY-NC-SA 4.0 (UAVScenes); not redistributed"}
    json.dump(meta, open(os.path.join(od, "tile_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
