"""EXP7 / UAVScenes -- diagnostic (CPU, magician env, python -I): fraction of executed camera positions whose vertical segment to the sky
crosses the tile mesh (camera under terrain / building / tree canopy, e.g. after entering through a mesh hole). Grid positions of the
settings camera box (official pose formula), segment test = the batched official rule of uav_tile_gpu.py.
Output experiments/cvpr_results/exp7/results/uav_under_surface.csv"""
import csv, glob, json, os, sys
import numpy as np
import trimesh
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts", "uavscenes"))
from uav_tile_gpu import seg_blocked_batch
rows = []
for td in sorted(glob.glob(os.path.join(W7, "uav_macpp", "uav_*"))):
    tile = os.path.basename(td)
    if not os.path.exists(os.path.join(td, "settings.json")) or tile == "uav_HKisland_S":
        continue
    st = json.load(open(os.path.join(td, "settings.json")))
    tm = trimesh.load(os.path.join(td, f"{tile}.obj"))
    tm = tm.dump(concatenate=True) if isinstance(tm, trimesh.Scene) else tm
    tm.vertices *= 10.0
    cam = st["camera"]
    lo, hi = 10.0 * np.asarray(cam["x_min"]), 10.0 * np.asarray(cam["x_max"])
    n = np.asarray([cam["pose_l"], cam["pose_w"], cam["pose_h"]])
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    X = lo + (0.5 + idx) * (hi - lo) / n
    under = dict(zip(map(tuple, idx), seg_blocked_batch(tm.ray, X, X + np.array([0.0, 1000.0, 0.0]))))
    for m, grp, pre in (("OURS", "V2M", "V2M"), ("RANDOM", "random", "random")):
        for k in range(10):
            ep = json.load(open(os.path.join(W7, "episodes", "uavscenes", "raw", m, grp, "episodes", f"ep_{pre}_{tile}_s{k}_r0.json")))
            keys = ep.get("executed_keys") or [s["pose_key"] for s in ep["steps"]]
            pos = [tuple(json.loads(x)[:3]) for x in keys]
            u = [under[p] for p in pos]
            rows.append({"tile": tile, "method": m, "start": k, "steps": len(u), "under_surface_steps": int(sum(u)), "fraction": sum(u) / len(u), "start_under": bool(u[0])})
    print(tile, "positions under surface", int(sum(under.values())), "of", len(under), flush=True)
with open(os.path.join(R, "experiments", "cvpr_results", "exp7", "results", "uav_under_surface.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for t in sorted({r["tile"] for r in rows}):
    for m in ("OURS", "RANDOM"):
        s = [r for r in rows if r["tile"] == t and r["method"] == m]
        print(t, m, "mean fraction %.3f, episodes with any %d/10, starts under %d" % (np.mean([r["fraction"] for r in s]), sum(r["under_surface_steps"] > 0 for r in s), sum(r["start_under"] for r in s)))
