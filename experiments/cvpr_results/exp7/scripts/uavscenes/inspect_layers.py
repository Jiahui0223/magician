"""Refined 2.5D / hole test on the cached UAVScenes mesh.

python -I inspect_layers.py <cache.npz> <out_dir> [grid_m]
Vertical rays from above on a regular grid; hits on shared edges are de-duplicated
(|dz| < 5 cm).  Interior holes = no-hit cells enclosed by the hit footprint.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

cache, out = sys.argv[1:3]
g = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
d = np.load(cache)
V, F = d["V"], d["F"]
m = trimesh.Trimesh(V, F, process=False, validate=False)
st = {}
# components and Euler characteristic
adj = m.face_adjacency
n = len(F)
A = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
nc, lab = connected_components(A, directed=False)
big = np.argmax(np.bincount(lab))
Fb = F[lab == big]
mb = trimesh.Trimesh(V, Fb, process=False, validate=False)
mb.remove_unreferenced_vertices()
st["largest_component_V_E_F"] = [int(len(mb.vertices)), int(len(mb.edges_unique)), int(len(mb.faces))]
st["largest_component_euler_char"] = int(mb.euler_number)
st["largest_component_is_watertight"] = bool(mb.is_watertight)
small = np.bincount(lab)
small_ids = [i for i in range(nc) if i != big]
sm_faces = np.isin(lab, small_ids)
st["small_components_faces_total"] = int(sm_faces.sum())
st["small_components_area_m2"] = float(m.area_faces[sm_faces].sum())
cen = m.triangles_center[sm_faces]
st["small_components_centroid_z_pct_5_50_95"] = np.percentile(cen[:, 2], [5, 50, 95]).tolist() if len(cen) else []

rmi = RayMeshIntersector(m)
x = np.arange(V[:, 0].min() + g / 2, V[:, 0].max(), g)
y = np.arange(V[:, 1].min() + g / 2, V[:, 1].max(), g)
X, Y = np.meshgrid(x, y)
O = np.stack([X.ravel(), Y.ravel(), np.full(X.size, V[:, 2].max() + 50)], 1)
Dd = np.tile([0, 0, -1.0], (len(O), 1))
loc, ri, ti = rmi.intersects_location(O, Dd, multiple_hits=True)
order = np.lexsort((-loc[:, 2], ri))
ri, z, ti = ri[order], loc[order, 2], ti[order]
newray = np.r_[True, ri[1:] != ri[:-1]]
newlayer = newray | (np.abs(np.diff(z, prepend=np.inf)) > 0.05)
nl = np.bincount(ri[newlayer], minlength=len(O))
hit = (nl > 0).reshape(X.shape)
filled = ndimage.binary_fill_holes(hit)
holes = filled & ~hit
lab_h, nh = ndimage.label(holes)
hs = np.bincount(lab_h.ravel())[1:] * g * g if nh else np.array([])
st["grid_m"] = g
st["footprint_cells"] = int(filled.sum())
st["footprint_area_m2"] = float(filled.sum() * g * g)
st["interior_hole_cells"] = int(holes.sum())
st["interior_hole_frac_of_footprint"] = float(holes.sum() / filled.sum())
st["interior_hole_regions"] = int(nh)
st["interior_hole_region_area_m2_max"] = float(hs.max()) if nh else 0.0
NL = nl.reshape(X.shape)[filled]
st["footprint_frac_1_layer"] = float((NL == 1).mean())
st["footprint_frac_2_layers"] = float((NL == 2).mean())
st["footprint_frac_ge3_layers"] = float((NL >= 3).mean())
# vertical separation between first and second layer where >=2 layers
first_z = z[newray]
idx2 = np.where(newlayer & ~newray)[0]
_, firstpos = np.unique(ri[idx2], return_index=True)
sec = idx2[firstpos]
fz = np.full(len(O), np.nan); fz[ri[newray]] = z[newray]
gap = fz[ri[sec]] - z[sec]
st["layer_gap_m_pct_5_50_95"] = np.percentile(gap, [5, 50, 95]).tolist() if len(gap) else []
json.dump(st, open(os.path.join(out, f"mesh_layers_g{g:g}.json"), "w"), indent=1)
print(json.dumps(st, indent=1))
np.savez_compressed(os.path.join(out, f"layers_g{g:g}.npz"), NL=nl.reshape(X.shape), filled=filled, x=x, y=y,
                    fz=fz.reshape(X.shape))
