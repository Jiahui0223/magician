"""Hit fraction / back-face fraction / depth per elevation for candidate views from the prototype grid.

python -I view_hit_stats.py <cache.npz> <layers_g1.npz> <out_dir> [n_pos]
Re-creates the legal positions of prototype_grid.py (same parameters) and casts a coarse 57x32
(vfov 60, MAGICIAN-style) image for elevations {-60,-30,0,30,60} x 8 azimuths.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

cache, layers, out = sys.argv[1:4]
npos = int(sys.argv[4]) if len(sys.argv) > 4 else 200
d = np.load(cache)
mesh = trimesh.Trimesh(d["V"], d["F"], process=False, validate=False)
rmi = RayMeshIntersector(mesh)
fn = mesh.face_normals
L = np.load(layers)
fz, filled, gx, gy = L["fz"], L["filled"], L["x"], L["y"]
g = gx[1] - gx[0]
DX, ZLEV, CLEAR = 40.0, np.arange(-150.0, 0.01, 25.0), 10.0
hmax10 = ndimage.maximum_filter(np.where(np.isfinite(fz), fz, -np.inf), size=int(2 * 10 / g) + 1)
inside_fp = ndimage.binary_erosion(filled, iterations=int(20 / g))
P = []
for z in ZLEV:
    for y in np.arange(gy[0] + DX / 2, gy[-1], DX):
        for x in np.arange(gx[0] + DX / 2, gx[-1], DX):
            ci, ri = min(int((x - gx[0]) / g), len(gx) - 1), min(int((y - gy[0]) / g), len(gy) - 1)
            if inside_fp[ri, ci] and z >= hmax10[ri, ci] + CLEAR:
                P.append([x, y, z])
P = np.array(P)
rng = np.random.default_rng(0)
P = P[rng.choice(len(P), min(npos, len(P)), replace=False)]
Wd, Hd = 57, 32
f = (Hd / 2) / np.tan(np.radians(30))
u, v = np.meshgrid(np.arange(Wd) + 0.5 - Wd / 2, np.arange(Hd) + 0.5 - Hd / 2)
dc = np.stack([u / f, v / f, np.ones_like(u)], -1).reshape(-1, 3)
res = {}
for el in (-60, -30, 0, 30, 60):
    hits, backs, deps = [], [], []
    for az in range(0, 360, 45):
        e, a = np.radians(el), np.radians(az)
        fwd = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        dw = dc @ np.stack([right, down, fwd], 1).T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        O = np.repeat(P, len(dw), 0)
        D = np.tile(dw, (len(P), 1))
        tri, ray, loc = rmi.intersects_id(O, D, multiple_hits=False, return_locations=True)
        hit = np.zeros(len(O), bool); hit[ray] = True
        bk = (fn[tri] * D[ray]).sum(1) > 0
        hits.append(hit.mean()); backs.append(bk.mean() if len(bk) else 0)
        deps.append(((loc - O[ray]) * D[ray]).sum(1))
    dd = np.concatenate(deps)
    res[f"elev{el:+d}"] = {"hit_frac": float(np.mean(hits)), "backface_frac_of_hits": float(np.mean(backs)),
                           "range_m_pct_5_50_95": np.percentile(dd, [5, 50, 95]).tolist() if len(dd) else None,
                           "frac_hits_within_175m": float((dd < 175).mean()) if len(dd) else None}
res["n_positions"] = int(len(P))
json.dump(res, open(os.path.join(out, "view_hit_stats.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
