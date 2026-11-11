"""EXP7 / UAVScenes -- mesh cache + footprint statistics (CPU, magician env, run with python -I).
Reads the downloaded Terra Mesh.ply (ENU metres, z up, per-vertex colour), drops tiny disconnected fragments (< 1 m^2 total area
per component), and writes <downloads>/derived/<scene>_mesh.npz (V float64 ENU m, F int64, C uint8 RGB) plus a 2 m height map
(max surface z per cell) and <visuals>/uavscenes/<scene>_heightmap.png. Usage: python -I uav_mesh_cache.py SCENE [SCENE ...]"""
import json
import os
import sys

import numpy as np
import trimesh

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
DL = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "downloads", "uavscenes")
VIS = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "visuals", "uavscenes")


def main():
    os.makedirs(os.path.join(DL, "derived"), exist_ok=True)
    os.makedirs(VIS, exist_ok=True)
    for scene in sys.argv[1:]:
        m = trimesh.load(os.path.join(DL, "raw", "terra_3dmap_pointcloud_mesh", scene, "Mesh.ply"), process=False)
        V = np.asarray(m.vertices, np.float64)
        F = np.asarray(m.faces, np.int64)
        C = np.asarray(m.visual.vertex_colors[:, :3], np.uint8) if hasattr(m.visual, "vertex_colors") else np.full((len(V), 3), 160, np.uint8)
        comp = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(F))
        area = m.area_faces
        ca = np.bincount(comp, weights=area)
        keep = ca[comp] >= 1.0
        F = F[keep]
        np.savez(os.path.join(DL, "derived", f"{scene}_mesh.npz"), V=V, F=F, C=C)
        lo, hi = V[F].reshape(-1, 3).min(0), V[F].reshape(-1, 3).max(0)
        cs = 2.0
        nx, ny = int(np.ceil((hi[0] - lo[0]) / cs)), int(np.ceil((hi[1] - lo[1]) / cs))
        hm = np.full((ny, nx), np.nan)
        P = V[np.unique(F)]
        ix = np.clip(((P[:, 0] - lo[0]) / cs).astype(int), 0, nx - 1)
        iy = np.clip(((P[:, 1] - lo[1]) / cs).astype(int), 0, ny - 1)
        flat = iy * nx + ix
        zmax = np.full(nx * ny, -np.inf)
        np.maximum.at(zmax, flat, P[:, 2])
        hm = np.where(np.isfinite(zmax), zmax, np.nan).reshape(ny, nx)
        np.save(os.path.join(DL, "derived", f"{scene}_heightmap_2m.npy"), hm)
        info = {"scene": scene, "n_vertices": int(len(V)), "n_faces_kept": int(len(F)), "n_faces_dropped": int((~keep).sum()),
                "bbox_min_m": lo.tolist(), "bbox_max_m": hi.tolist(), "heightmap_cell_m": cs, "heightmap_shape": [ny, nx],
                "footprint_fraction_of_bbox": float(np.isfinite(hm).mean()), "z_p01_p50_p99": np.nanpercentile(hm, [1, 50, 99]).tolist()}
        json.dump(info, open(os.path.join(DL, "derived", f"{scene}_mesh_info.json"), "w"), indent=1)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7 * ny / nx))
        im = ax.imshow(hm, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], cmap="terrain")
        plt.colorbar(im, ax=ax, label="max surface z (m)")
        ax.set_title(f"{scene}: height map (2 m cells)")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        fig.savefig(os.path.join(VIS, f"{scene}_heightmap.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
