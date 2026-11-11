"""Geometry quality audit of a UAVScenes Terra mesh (CPU only).

python -I inspect_mesh.py <Mesh.ply> <cache.npz> <out_dir>
Writes mesh_stats.json and debug PNGs (top-down height / boundary / layering).
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ply, cache, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
t0 = time.time()
if os.path.exists(cache):
    d = np.load(cache)
    V, F, C = d["V"], d["F"], d["C"]
else:
    m0 = trimesh.load(ply, process=False)
    V = np.asarray(m0.vertices, dtype=np.float64)
    F = np.asarray(m0.faces, dtype=np.int64)
    C = np.asarray(m0.visual.vertex_colors)[:, :3].astype(np.uint8)
    np.savez(cache, V=V, F=F, C=C)
print(f"load {time.time() - t0:.1f}s V={len(V)} F={len(F)}")
st = {"n_vertices_raw": int(len(V)), "n_faces_raw": int(len(F))}
st["finite_vertices"] = bool(np.isfinite(V).all())
st["bounds_min"] = V.min(0).tolist()
st["bounds_max"] = V.max(0).tolist()
st["extent"] = (V.max(0) - V.min(0)).tolist()

# raw (unmerged) topology
m = trimesh.Trimesh(V, F, process=False, validate=False)
area_f = m.area_faces
st["total_area_m2"] = float(area_f.sum())
st["degenerate_faces(area<1e-8)"] = int((area_f < 1e-8).sum())
st["dup_face_index(same vertex twice)"] = int(((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])).sum())
el = m.edges_unique_length
st["edge_len_m_pct_1_50_99"] = np.percentile(el, [1, 50, 99]).tolist()
st["unreferenced_vertices"] = int(len(V) - len(np.unique(F)))

# merge coincident vertices (Terra blocks are merged by CloudCompare -> seams)
mm = trimesh.Trimesh(V, F, process=False, validate=False)
mm.merge_vertices(merge_tex=True, merge_norm=True)
st["n_vertices_after_merge"] = int(len(mm.vertices))
for name, mesh in [("raw", m), ("merged", mm)]:
    e = np.sort(mesh.edges, axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    st[f"{name}_n_edges"] = int(len(cnt))
    st[f"{name}_boundary_edges(1 face)"] = int((cnt == 1).sum())
    st[f"{name}_nonmanifold_edges(>2 faces)"] = int((cnt > 2).sum())
    st[f"{name}_boundary_edge_ratio"] = float((cnt == 1).mean())
    st[f"{name}_is_watertight"] = bool(mesh.is_watertight)
    st[f"{name}_is_winding_consistent"] = bool(mesh.is_winding_consistent)
    # face connected components via shared edges
    adj = mesh.face_adjacency
    n = len(mesh.faces)
    A = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    sizes = np.sort(np.bincount(lab))[::-1]
    st[f"{name}_n_face_components"] = int(nc)
    st[f"{name}_largest_component_face_frac"] = float(sizes[0] / n)
    st[f"{name}_components_ge_1000_faces"] = int((sizes >= 1000).sum())
    if name == "merged":
        # boundary loops: connected components of the boundary-edge graph
        be = np.unique(e, axis=0)[cnt == 1]
        nv = len(mesh.vertices)
        B = sp.coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(nv, nv))
        nb, blab = connected_components(B, directed=False)
        used = np.unique(be)
        loops = np.bincount(blab[be[:, 0]], minlength=nb)
        L = np.zeros(nb)
        np.add.at(L, blab[be[:, 0]], np.linalg.norm(mesh.vertices[be[:, 0]] - mesh.vertices[be[:, 1]], axis=1))
        L = L[loops > 0]
        order = np.argsort(L)[::-1]
        st["boundary_loops"] = int(len(L))
        st["boundary_loop_perimeters_top5_m"] = L[order[:5]].tolist()
        st["boundary_loops_perimeter_gt_10m"] = int((L > 10).sum())
        st["boundary_loops_perimeter_gt_50m"] = int((L > 50).sum())
        bverts = mesh.vertices[used]
        bdry_pts = bverts

# orientation / 2.5D character
fn = m.face_normals
st["face_normal_z_pct_1_5_50"] = np.percentile(fn[:, 2], [1, 5, 50]).tolist()
st["area_frac_upward(nz>0.5)"] = float(area_f[fn[:, 2] > 0.5].sum() / area_f.sum())
st["area_frac_steep(|nz|<0.5)"] = float(area_f[np.abs(fn[:, 2]) < 0.5].sum() / area_f.sum())
st["area_frac_downward(nz<-0.5)"] = float(area_f[fn[:, 2] < -0.5].sum() / area_f.sum())

# multi-layer test: vertical rays on a 2 m grid, count all intersections
from trimesh.ray.ray_pyembree import RayMeshIntersector
rmi = RayMeshIntersector(m)
x = np.arange(V[:, 0].min() + 1, V[:, 0].max(), 2.0)
y = np.arange(V[:, 1].min() + 1, V[:, 1].max(), 2.0)
X, Y = np.meshgrid(x, y)
O = np.stack([X.ravel(), Y.ravel(), np.full(X.size, V[:, 2].max() + 50)], 1)
D = np.tile([0, 0, -1.0], (len(O), 1))
t1 = time.time()
loc, ri, ti = rmi.intersects_location(O, D, multiple_hits=True)
nh = np.bincount(ri, minlength=len(O))
print(f"vertical rays {len(O)} in {time.time() - t1:.1f}s")
first = np.full(len(O), np.nan)
order = np.lexsort((-loc[:, 2], ri))
ri_s, z_s = ri[order], loc[order, 2]
firstidx = np.r_[0, np.where(np.diff(ri_s) > 0)[0] + 1]
first[ri_s[firstidx]] = z_s[firstidx]
st["topdown_grid_m"] = 2.0
st["topdown_columns"] = int(len(O))
st["topdown_frac_no_hit(holes/outside)"] = float((nh == 0).mean())
st["topdown_frac_1_hit"] = float((nh == 1).mean())
st["topdown_frac_ge2_hits(overhang/multi-layer)"] = float((nh >= 2).mean())
# hits inside the convex footprint only (exclude outside the scanned area)
from scipy.spatial import ConvexHull, Delaunay
hull = Delaunay(V[ConvexHull(V[:, :2]).vertices, :2])
inside = hull.find_simplex(O[:, :2]) >= 0
st["topdown_inside_hull_columns"] = int(inside.sum())
st["topdown_inside_hull_frac_no_hit"] = float((nh[inside] == 0).mean())
st["topdown_inside_hull_frac_ge2_hits"] = float((nh[inside] >= 2).mean())
# first-hit back faces from above = downward-facing surface seen from the sky
first_tri = np.full(len(O), -1)
first_tri[ri_s[firstidx]] = ti[order][firstidx]
hitm = first_tri >= 0
st["topdown_first_hit_backface_frac"] = float((fn[first_tri[hitm], 2] < 0).mean())

H = first.reshape(X.shape)
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
im = ax[0].imshow(H, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="terrain")
plt.colorbar(im, ax=ax[0], label="first-hit z (m)")
ax[0].set_title("top-down first hit z (NaN = hole/outside)")
ax[1].imshow(nh.reshape(X.shape), origin="lower", extent=[x[0], x[-1], y[0], y[-1]], vmin=0, vmax=3, cmap="viridis")
ax[1].set_title("# intersections per vertical ray (0..3+)")
ax[2].scatter(bdry_pts[:, 0], bdry_pts[:, 1], s=0.05, c="k")
ax[2].set_aspect("equal"); ax[2].set_title("boundary-edge vertices (merged mesh)")
for a in ax:
    a.set_xlabel("x (E, m)"); a.set_ylabel("y (N, m)")
plt.tight_layout(); plt.savefig(os.path.join(out, "mesh_topdown.png"), dpi=100)
np.savez_compressed(os.path.join(out, "topdown_heightmap.npz"), H=H, nh=nh.reshape(X.shape), x=x, y=y)
json.dump(st, open(os.path.join(out, "mesh_stats.json"), "w"), indent=1)
print(json.dumps(st, indent=1))
