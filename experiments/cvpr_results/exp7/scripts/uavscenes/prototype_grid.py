"""Prototype of the closed-loop ingredients on a UAVScenes mesh (CPU only, metric ENU, z-up).

python -I prototype_grid.py <cache.npz> <layers_g1.npz> <poses_interval5.csv> <out_dir>

Builds (no model, no GPU):
  1. a free-space camera position grid (xy spacing DX m, absolute z levels) + legality test
     (inside footprint, terrain clearance, 26-ray inside-geometry probe);
  2. 6-neighbour move legality by segment-mesh ray casting;
  3. an area-weighted GT surface sample inside an eroded ROI, and the fraction of it that is
     visible (line of sight, range-limited) from at least one legal grid position;
  4. timing of a 456x256 / 60 deg z-depth render by embree (CPU) from a grid position.
All numbers are written to prototype_grid.json.  Parameters are illustrative, not a frozen protocol.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cache, layers, poses_csv, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
DX = 40.0                      # xy spacing (m)
ZLEV = np.arange(-150.0, 0.01, 25.0)  # absolute z levels (m, ENU relative to SRS origin)
CLEAR = 10.0                   # min clearance above terrain (m)
ERODE_ROI = 30                 # ROI = footprint eroded by 30 m
rng = np.random.default_rng(0)

d = np.load(cache)
V, F = d["V"], d["F"]
mesh = trimesh.Trimesh(V, F, process=False, validate=False)
rmi = RayMeshIntersector(mesh)
fn = mesh.face_normals
L = np.load(layers)
fz, filled, gx, gy = L["fz"], L["filled"], L["x"], L["y"]
g = gx[1] - gx[0]
st = {"params": {"DX_m": DX, "z_levels_m": ZLEV.tolist(), "clearance_m": CLEAR, "roi_erosion_m": ERODE_ROI}}

# --- 1. camera positions
hz = np.where(np.isfinite(fz), fz, np.inf)
hmax10 = ndimage.maximum_filter(np.where(np.isfinite(fz), fz, -np.inf), size=int(2 * 10 / g) + 1)  # max terrain within ~10 m
inside_fp = ndimage.binary_erosion(filled, iterations=int(20 / g))  # keep cameras >= 20 m inside the footprint
xs = np.arange(gx[0] + DX / 2, gx[-1], DX)
ys = np.arange(gy[0] + DX / 2, gy[-1], DX)
cand, legal = [], []
for z in ZLEV:
    for y in ys:
        for x in xs:
            ci = min(int((x - gx[0]) / g), len(gx) - 1)
            ri = min(int((y - gy[0]) / g), len(gy) - 1)
            ok = bool(inside_fp[ri, ci]) and z >= hmax10[ri, ci] + CLEAR
            cand.append([x, y, z])
            legal.append(ok)
cand = np.array(cand)
legal = np.array(legal)
st["grid_shape_xyz"] = [len(xs), len(ys), len(ZLEV)]
st["grid_positions_total"] = int(len(cand))
st["grid_positions_inside_footprint"] = int(sum(inside_fp[min(int((y - gy[0]) / g), len(gy) - 1), min(int((x - gx[0]) / g), len(gx) - 1)] for x, y, _ in cand))
st["grid_positions_legal_clearance"] = int(legal.sum())
# 26-direction probe: any back-face first hit => inside geometry
dirs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
P = cand[legal]
O = np.repeat(P, 26, 0)
D = np.tile(dirs, (len(P), 1))
tri, ray, loc = rmi.intersects_id(O, D, multiple_hits=False, return_locations=True)
backhit = np.zeros(len(O), bool)
backhit[ray] = (fn[tri] * D[ray]).sum(1) > 0
inside = backhit.reshape(-1, 26).any(1)
st["legal_positions_with_backface_probe_hit"] = int(inside.sum())
# same probe at the recorded camera poses (sanity: real cameras must be outside)
import csv
RP = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in csv.DictReader(open(poses_csv))])
O2 = np.repeat(RP, 26, 0); D2 = np.tile(dirs, (len(RP), 1))
tri2, ray2, _ = rmi.intersects_id(O2, D2, multiple_hits=False, return_locations=True)
bh2 = np.zeros(len(O2), bool); bh2[ray2] = (fn[tri2] * D2[ray2]).sum(1) > 0
st["recorded_poses_probe_inside"] = int(bh2.reshape(-1, 26).any(1).sum())
st["recorded_poses_n"] = int(len(RP))

# --- 2. move legality (6-neighbour, one axis at a time) via segment ray casting
idx = -np.ones((len(ZLEV), len(ys), len(xs)), int)
li = np.where(legal)[0]
iz, iy, ix = np.unravel_index(np.arange(len(cand)), (len(ZLEV), len(ys), len(xs)))
idx[iz[li], iy[li], ix[li]] = np.arange(len(li))
edges = []
for dz, dy, dx_ in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
    a = idx[:len(ZLEV) - dz, :len(ys) - dy, :len(xs) - dx_]
    b = idx[dz:, dy:, dx_:]
    m_ = (a >= 0) & (b >= 0)
    edges.append(np.stack([a[m_], b[m_]], 1))
E = np.concatenate(edges)
A_, B_ = P[E[:, 0]], P[E[:, 1]]
seg = B_ - A_
sl = np.linalg.norm(seg, axis=1)
t0 = time.time()
tri3, ray3, loc3 = rmi.intersects_id(A_, seg / sl[:, None], multiple_hits=False, return_locations=True)
st["segment_test_time_s"] = time.time() - t0
blocked = np.zeros(len(E), bool)
hd = np.linalg.norm(loc3 - A_[ray3], axis=1)
blocked[ray3] = hd < sl[ray3] + 2.0
st["candidate_edges_between_legal_positions"] = int(len(E))
st["edges_blocked_by_mesh"] = int(blocked.sum())
G = coo_matrix((np.ones((~blocked).sum()), (E[~blocked, 0], E[~blocked, 1])), shape=(len(P), len(P)))
nc, lab = connected_components(G, directed=False)
st["legal_graph_components"] = int(nc)
st["legal_graph_largest_component"] = int(np.bincount(lab).max())
deg = np.bincount(E[~blocked].ravel(), minlength=len(P))
st["legal_node_degree_pct_5_50_95"] = np.percentile(deg, [5, 50, 95]).tolist()

# --- 3. GT surface sample in ROI and its visibility from legal positions
roi = ndimage.binary_erosion(filled, iterations=int(ERODE_ROI / g))
cen = mesh.triangles_center
ci = np.clip(((cen[:, 0] - gx[0]) / g).astype(int), 0, len(gx) - 1)
ri = np.clip(((cen[:, 1] - gy[0]) / g).astype(int), 0, len(gy) - 1)
froi = roi[ri, ci]
area = mesh.area_faces
st["roi_faces"] = int(froi.sum())
st["roi_area_m2"] = float(area[froi].sum())
st["roi_area_frac_of_mesh"] = float(area[froi].sum() / area.sum())
sub = trimesh.Trimesh(V, F[froi], process=False, validate=False)
gt, fid = trimesh.sample.sample_surface(sub, 100000, seed=0)
st["gt_points"] = int(len(gt))
st["gt_points_per_m2"] = float(len(gt) / area[froi].sum())
nrm = sub.face_normals[fid]
sel =rng.choice(len(gt), 20000, replace=False)
probe, pn = gt[sel], nrm[sel]
from scipy.spatial import cKDTree
kd = cKDTree(P)
for R in (150.0, 350.0):
    vis = np.zeros(len(probe), bool)
    nb = kd.query_ball_point(probe, R)
    pairs = [(i, j) for i, js in enumerate(nb) for j in js]
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    # front-facing only
    v = P[pj] - probe[pi]
    dist = np.linalg.norm(v, axis=1)
    front = (v * pn[pi]).sum(1) > 0
    pi, pj, v, dist = pi[front], pj[front], v[front], dist[front]
    o = probe[pi] + pn[pi] * 0.05
    dd = v / dist[:, None]
    t0 = time.time()
    trv, rv, lv = rmi.intersects_id(o, dd, multiple_hits=False, return_locations=True)
    occl = np.zeros(len(o), bool)
    occl[rv] = np.linalg.norm(lv - o[rv], axis=1) < dist[rv] - 0.5
    ok = ~occl
    vis[np.unique(pi[ok])] = True
    st[f"gt_visible_from_some_legal_position_R{int(R)}m"] = float(vis.mean())
    st[f"visibility_rays_R{int(R)}m"] = int(len(o))
    st[f"visibility_time_s_R{int(R)}m"] = time.time() - t0

# --- 4. CPU embree render timing at simulator resolution (456x256, vfov 60)
Wd, Hd, vfov = 456, 256, 60.0
f = (Hd / 2) / np.tan(np.radians(vfov / 2))
u, v_ = np.meshgrid(np.arange(Wd) + 0.5 - Wd / 2, np.arange(Hd) + 0.5 - Hd / 2)
dc = np.stack([u / f, v_ / f, np.ones_like(u)], -1).reshape(-1, 3)
c0 = P[np.argmin(np.abs(P[:, 2] - (-50)) + 1e-3 * np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1))]
el = np.radians(-60)  # look 60 deg down, azimuth north
fwd = np.array([0, np.cos(el), np.sin(el)])
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
down = np.cross(fwd, right)
Rcw = np.stack([right, down, fwd], 1)
dw = dc @ Rcw.T
dw /= np.linalg.norm(dw, axis=1, keepdims=True)
t0 = time.time()
trr, rr, lr = rmi.intersects_id(np.tile(c0, (len(dw), 1)), dw, multiple_hits=False, return_locations=True)
st["embree_render_456x256_s"] = time.time() - t0
zd = np.full(len(dw), np.nan)
zd[rr] = (lr - c0) @ fwd
st["render_cam"] = c0.tolist()
st["render_hit_frac"] = float(np.isfinite(zd).mean())
st["render_zdepth_m_pct_5_50_95"] = np.nanpercentile(zd, [5, 50, 95]).tolist()
np.save(os.path.join(out, "proto_render_zdepth.npy"), zd.reshape(Hd, Wd))

json.dump(st, open(os.path.join(out, "prototype_grid.json"), "w"), indent=1)
print(json.dumps(st, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].imshow(np.where(filled, fz, np.nan), origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="terrain")
ax[0].contour(gx, gy, roi.astype(float), levels=[0.5], colors="r", linewidths=0.8)
for z, c in zip(ZLEV, plt.cm.viridis(np.linspace(0, 1, len(ZLEV)))):
    s = P[:, 2] == z
    ax[0].scatter(P[s, 0], P[s, 1], s=4 + 0 * z, color=c, label=f"z={z:.0f}: {s.sum()}")
ax[0].plot(RP[:, 0], RP[:, 1], "k-", lw=0.4)
ax[0].legend(fontsize=6, loc="lower left"); ax[0].set_title("legal grid positions (by z level), ROI (red), recorded path (black)")
ax[0].set_aspect("equal")
im = ax[1].imshow(zd.reshape(Hd, Wd), cmap="turbo"); plt.colorbar(im, ax=ax[1], label="z-depth (m)")
ax[1].set_title(f"embree z-depth 456x256 vfov60 from grid cell {np.round(c0, 0)}, pitch -60")
plt.tight_layout(); plt.savefig(os.path.join(out, "prototype_grid.png"), dpi=100)
