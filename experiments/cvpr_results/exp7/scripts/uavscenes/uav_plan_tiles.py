"""EXP7 / UAVScenes -- UAV-v0 tile list (CPU, numpy). Rule fixed before any closed loop: per scene, square tiles of TILE m on a grid
with stride TILE centred on the bounding box of the mesh footprint; a tile is kept iff >= 90 % of its 2 m cells contain mesh
surface (no void inside the scene box). Tiles are named <scene>_<row><col> (row from south, col from west). Writes
experiments/cvpr_results/exp7/manifests/uav_tiles_v0.csv. No model output and no coverage enter the choice.
Usage: python -I uav_plan_tiles.py [--tile 260]"""
import argparse
import csv
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
DL = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "downloads", "uavscenes", "derived")
OUT = os.path.join(R, "experiments", "cvpr_results", "exp7", "manifests", "uav_tiles_v0.csv")
SCENES = ["AMtown", "AMvalley", "HKairport", "HKisland"]

ap = argparse.ArgumentParser()
ap.add_argument("--tile", type=float, default=260.0)
a = ap.parse_args()
T = a.tile
rows = []
for s in SCENES:
    info = json.load(open(os.path.join(DL, f"{s}_mesh_info.json")))
    hm = np.load(os.path.join(DL, f"{s}_heightmap_2m.npy"))
    lo = np.asarray(info["bbox_min_m"])
    fin = np.isfinite(hm)
    ys, xs = np.nonzero(fin)
    e0, e1 = lo[0] + 2.0 * xs.min(), lo[0] + 2.0 * (xs.max() + 1)
    n0, n1 = lo[1] + 2.0 * ys.min(), lo[1] + 2.0 * (ys.max() + 1)
    nc, nr = max(1, int((e1 - e0) // T)), max(1, int((n1 - n0) // T))
    oe, on = 0.5 * (e0 + e1) - nc * T / 2, 0.5 * (n0 + n1) - nr * T / 2
    for r in range(nr):
        for c in range(nc):
            b = [oe + c * T, on + r * T, oe + (c + 1) * T, on + (r + 1) * T]
            ix0, ix1 = int(round((b[0] - lo[0]) / 2)), int(round((b[2] - lo[0]) / 2))
            iy0, iy1 = int(round((b[1] - lo[1]) / 2)), int(round((b[3] - lo[1]) / 2))
            sub = np.full((iy1 - iy0, ix1 - ix0), False)
            ya, yb, xa, xb = max(iy0, 0), min(iy1, hm.shape[0]), max(ix0, 0), min(ix1, hm.shape[1])
            if ya < yb and xa < xb:
                sub[ya - iy0:yb - iy0, xa - ix0:xb - ix0] = fin[ya:yb, xa:xb]
            fp = float(sub.mean())
            h = hm[ya:yb, xa:xb]
            relief = float(np.nanpercentile(h, 99.9) - np.nanpercentile(h, 1)) if np.isfinite(h).any() else float("nan")
            rows.append({"scene": s, "tile": f"{s}_{r}{c}", "east0": round(b[0], 1), "north0": round(b[1], 1), "east1": round(b[2], 1), "north1": round(b[3], 1),
                         "footprint": round(fp, 4), "relief_m": round(relief, 1), "keep": int(fp >= 0.9)})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
print("kept", sum(r["keep"] for r in rows), "of", len(rows))
