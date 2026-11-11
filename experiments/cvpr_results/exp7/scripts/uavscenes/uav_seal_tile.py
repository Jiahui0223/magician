"""EXP7 / UAVScenes -- UAV-v1 tile = UAV-v0 tile + hole plugs (CPU, magician env, python -I).

Problem (UAV-v0): Terra meshes are 2.5D draped surfaces with holes (water, missing data). Through a hole the camera can pass the official
segment test and fly below the reconstructed surface (AMvalley_01: Ours 23.8 % of steps, HKisland_00: 12.8 %).
Rule (fixed before any v1 run): on a grid of `cell` runtime units over the whole tile mesh extent (x-z plane, y up), a cell is a HOLE if
the vertical ray through the cell centre hits no mesh face (a first version used 'no vertex in the cell', which flags large
triangles as holes; replaced before any v1 run). Every hole cell gets a closed plug: a top quad at the maximum surface height of its 8-neighbourhood (holes
filled iteratively by nearest non-hole neighbours) and 4 vertical side quads down to (minimum neighbour height - 2 runtime units), each
quad enlarged by 5 % to close seams. Everything else of the v0 tile (mesh, settings blocks, coverage grid, pose grid) is unchanged; plug
faces get a neutral grey texel. Then camera pre-selection / starts / visibility are recomputed by uav_tile_gpu.py on the v1 folder.
Check printed here (CPU): reachable positions (BFS from open sky with the official segment rule) and how many of them lie under the
surface (no open sky above) -- the target is 0.
Usage: python -I uav_seal_tile.py TILE [--cell 1.0]   (TILE = uav_<scene>_<id>; writes uav_<scene>_<id>_v1)"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import trimesh

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
ROOT = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts", "uavscenes"))
from uav_tile_gpu import seg_blocked_batch  # noqa: E402
from uav_build_tile import write_obj  # noqa: E402


def tm_ray(V, F):
    return trimesh.Trimesh(V, F, process=False).ray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tile")
    ap.add_argument("--cell", type=float, default=1.0, help="runtime units (1.0 = 2 m)")
    a = ap.parse_args()
    src = os.path.join(ROOT, a.tile)
    dst = os.path.join(ROOT, a.tile + "_v1")
    os.makedirs(dst, exist_ok=False)
    tm = trimesh.load(os.path.join(src, f"{a.tile}.obj"), process=False)
    tm = tm.dump(concatenate=True) if isinstance(tm, trimesh.Scene) else tm
    V = np.asarray(tm.vertices, np.float64) * 10.0                       # runtime units, y up
    F = np.asarray(tm.faces, np.int64)
    try:
        uv = np.asarray(tm.visual.uv)
        img = np.asarray(tm.visual.material.image.convert("RGB"))
        T = img.shape[0]
        px = np.clip((uv[:, 0] * T).astype(int), 0, T - 1); py = np.clip(((1 - uv[:, 1]) * T).astype(int), 0, T - 1)
        Cv = img[py, px]
    except Exception:
        Cv = np.full((len(V), 3), 150, np.uint8)
    c = a.cell
    x0, z0 = V[:, 0].min(), V[:, 2].min()
    nx, nz = int(np.ceil((V[:, 0].max() - x0) / c)) + 1, int(np.ceil((V[:, 2].max() - z0) / c)) + 1
    # surface heights per cell from vertical rays through the cell centre (all hits: top and bottom of the surface stack)
    gx, gz = np.meshgrid(x0 + (np.arange(nx) + 0.5) * c, z0 + (np.arange(nz) + 0.5) * c, indexing="ij")
    org = np.stack([gx.ravel(), np.full(gx.size, V[:, 1].max() + 100.0), gz.ravel()], 1)
    loc, iray, _ = tm_ray(V, F).intersects_location(ray_origins=org, ray_directions=np.tile([0.0, -1.0, 0.0], (len(org), 1)))
    hmax = np.full(nx * nz, -np.inf); hmin = np.full(nx * nz, np.inf)
    np.maximum.at(hmax, iray, loc[:, 1]); np.minimum.at(hmin, iray, loc[:, 1])
    hmax, hmin = hmax.reshape(nx, nz), hmin.reshape(nx, nz)
    hole = ~np.isfinite(hmax)
    # also every camera column of the settings pose grid: a crack exactly on a column (e.g. a Terra block seam) lets the camera pass
    st0 = json.load(open(os.path.join(src, "settings.pending.json")))["camera"]
    clo, chi = 10.0 * np.asarray(st0["x_min"]), 10.0 * np.asarray(st0["x_max"])
    cn = np.asarray([st0["pose_l"], st0["pose_w"], st0["pose_h"]])
    cx = clo[0] + (0.5 + np.arange(cn[0])) * (chi[0] - clo[0]) / cn[0]
    cz = clo[2] + (0.5 + np.arange(cn[2])) * (chi[2] - clo[2]) / cn[2]
    CX, CZ = np.meshgrid(cx, cz, indexing="ij")
    corg = np.stack([CX.ravel(), np.full(CX.size, V[:, 1].max() + 100.0), CZ.ravel()], 1)
    _, ir2, _ = tm_ray(V, F).intersects_location(ray_origins=corg, ray_directions=np.tile([0.0, -1.0, 0.0], (len(corg), 1)))
    miss = np.setdiff1d(np.arange(len(corg)), ir2)
    for q in miss:
        hole[int((corg[q, 0] - x0) / c), int((corg[q, 2] - z0) / c)] = True
    n_col_miss = int(len(miss))
    # only holes inside the convex footprint of the tile mesh (the outer margin of the crop is a square; cells outside the data stay open)
    from scipy import ndimage
    filled_max, filled_min = hmax.copy(), hmin.copy()
    idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
    filled_max = hmax[idx[0], idx[1]]; filled_min = hmin[idx[0], idx[1]]
    top = ndimage.maximum_filter(filled_max, size=3); low = ndimage.minimum_filter(filled_min, size=3) - 2.0
    hi_ = np.argwhere(hole)
    nv0 = len(V)
    newV, newF = [], []
    e = 0.05 * c
    for (i, k) in hi_:
        xa, xb = x0 + i * c - e, x0 + (i + 1) * c + e
        za, zb = z0 + k * c - e, z0 + (k + 1) * c + e
        yt, yb = top[i, k], low[i, k]
        b = nv0 + len(newV)
        newV += [[xa, yt, za], [xb, yt, za], [xb, yt, zb], [xa, yt, zb], [xa, yb, za], [xb, yb, za], [xb, yb, zb], [xa, yb, zb]]
        q = [(0, 1, 2), (0, 2, 3), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        newF += [[b + u, b + v, b + w] for u, v, w in q]
    V2 = np.vstack([V, np.asarray(newV, np.float64).reshape(-1, 3)])
    F2 = np.vstack([F, np.asarray(newF, np.int64).reshape(-1, 3)])
    C2 = np.vstack([Cv, np.full((len(V2) - len(V), 3), 128, np.uint8)])
    write_obj(os.path.join(dst, f"{a.tile}_v1.obj"), f"{a.tile}_v1", V2 / 10.0, F2, C2)
    shutil.copy(os.path.join(src, "settings.pending.json"), os.path.join(dst, "settings.pending.json"))
    meta = json.load(open(os.path.join(src, "tile_meta.json")))
    meta.update({"protocol": "UAV-v1", "v1_from": a.tile, "v1_rule": f"hole plugs on a {c} runtime-unit grid (cells whose centre vertical ray, or a camera-column vertical ray inside them, hits no face; top = max 8-neighbour height, "
                 "walls down to min neighbour height - 2; 5 % overlap)", "v1_hole_cells": int(len(hi_)), "v1_grid_cells": int(nx * nz),
                 "faces_tile": int(len(F2)), "v1_plug_faces": int(len(F2) - len(F))})
    json.dump(meta, open(os.path.join(dst, "tile_meta.json"), "w"), indent=1)
    # CPU check: reachability with the official rule and under-surface positions
    st = json.load(open(os.path.join(dst, "settings.pending.json")))
    m2 = trimesh.Trimesh(V2, F2, process=False)
    cam = st["camera"]
    lo, hi = 10.0 * np.asarray(cam["x_min"]), 10.0 * np.asarray(cam["x_max"])
    n = np.asarray([cam["pose_l"], cam["pose_w"], cam["pose_h"]])
    pidx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    X = lo + (0.5 + pidx) * (hi - lo) / n
    key = {tuple(r): i for i, r in enumerate(pidx)}
    sky = ~seg_blocked_batch(m2.ray, X, X + np.array([0.0, 1000.0, 0.0]))
    top_i = np.flatnonzero(pidx[:, 1] == n[1] - 1)
    edges = [(i, key[tuple(np.asarray(r) + d)]) for i, r in enumerate(pidx) for d in np.eye(3, dtype=int) if tuple(np.asarray(r) + d) in key]
    edges = np.asarray(edges)
    blocked = seg_blocked_batch(m2.ray, X[edges[:, 0]], X[edges[:, 1]])
    adj = [[] for _ in range(len(pidx))]
    for (i, j), bl in zip(edges, blocked):
        if not bl:
            adj[i].append(j); adj[j].append(i)
    reach = np.zeros(len(pidx), bool)
    stack = list(top_i[sky[top_i]]); reach[stack] = True
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not reach[j]:
                reach[j] = True; stack.append(j)
    res = {"tile": a.tile + "_v1", "hole_cells": int(len(hi_)), "camera_columns_without_surface": n_col_miss, "plug_faces": int(len(F2) - len(F)), "positions": int(len(pidx)),
           "reachable": int(reach.sum()), "reachable_under_surface": int((reach & ~sky).sum())}
    json.dump(res, open(os.path.join(dst, "seal_check.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
