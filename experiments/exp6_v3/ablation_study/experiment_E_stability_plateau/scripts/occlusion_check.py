"""EXP6 V3 / ablation E -- line-of-sight check of the predicted novel cells of the EXECUTED view (CPU, diagnostic).
For step t, executed candidate c: predicted cells (e_predict union cells) vs the agent's real observation of frame t+1 (its S10
sample points and camera centre, from the service record; = what the agent actually saw from c). For each predicted cell, the
real points within ANG = 1.5 deg of its viewing direction are found (unit-vector KD-tree, chord <= 2 sin(ANG/2)):
  occluded   a real point in that direction is nearer than the predicted cell by > 2.0 (the real view stops in front of it:
             the predicted surface lies BEHIND a real surface = see-through prediction)
  consistent a real point in that direction is within 2.0 of the cell's depth (the real view reaches it)
  beyond     real points in that direction are all farther than the cell + 2.0 (the predicted surface is in front of the real
             one: a phantom in observed free space)
  no-return  no real point in that direction (outside the real image or no return)
Weighted by the predicted cell weight; split by the cell's class (hit / seen / empty, tolerance 2.0, as e_predict).
Output: results/occlusion_check.json and a printed table."""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
M20 = (1 << 20) - 1
ANG = np.deg2rad(1.5)
NAME = {"V3": "H0", "V3-H3": "H3", "V3-D1-H3-NORISK": "D1", "V3-YBEST-H3": "YBEST-H3"}


def dec(ids):
    return np.stack([(ids >> 40) & M20, (ids >> 20) & M20, ids & M20], 1).astype(np.float64) - (1 << 19) + 0.5


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def main():
    acc = {}
    for f in sorted(x for x in glob.glob(os.path.join(W, "oracle", "*", "*_r0_predict.jsonl")) if "/smoke/" not in x):
        mid = os.path.basename(os.path.dirname(f))
        scene, k = os.path.basename(f).replace("_r0_predict.jsonl", "").rsplit("_s", 1)
        k = int(k)
        rec = json.load(open(os.path.join(EP, mid, f"{scene}_s{k}.json")))
        c = rec["coverage_official"]
        ep = json.load(open(os.path.join(R, rec["episode_file"])))
        svc = os.path.join(os.path.dirname(os.path.dirname(os.path.join(R, rec["episode_file"]))), "svc", ep["tag"])
        oz = np.load(f.replace("_predict.jsonl", ".npz"))
        o = json.load(open(f.replace("_predict.jsonl", ".json")))
        orow = {r["t"]: r for r in o["steps"]}
        gtree = cKDTree(oz["gt_points"].astype(np.float64))
        z = np.load(f.replace(".jsonl", "_cells.npz"))
        idx = {(int(a), int(b)): (int(p), int(n)) for a, b, p, n in z["index"]}
        on = onset(c)
        low = (c[100] < 0.70) if scene == "colosseum" else None
        for line in open(f):
            r = json.loads(line)
            t, ex = r["t"], r["executed"]
            p1 = os.path.join(svc, f"svc_{scene}_s{k}_t{t + 1}.npz")
            if not os.path.exists(p1):
                continue
            off, n = idx[(t, ex)]
            if n == 0:
                continue
            cells, w = dec(z["ids"][off:off + n]), z["w"][off:off + n]
            s1 = np.load(p1)
            cam = np.asarray(s1["cam_center"], np.float64).reshape(3)
            real = s1["sample_points"].astype(np.float64).reshape(-1, 3)
            if len(real) == 0:
                continue
            cr = orow[t]["cands"][ex]
            tids = oz["cell_ids"][cr["cell_off"]:cr["cell_off"] + cr["gt_cells"]]
            hit = (cKDTree(dec(tids)).query(cells, k=1, distance_upper_bound=2.0)[0] < np.inf) if len(tids) else np.zeros(n, bool)
            surf = gtree.query(cells, k=1, distance_upper_bound=2.0)[0] < np.inf
            cls = np.where(hit, "hit", np.where(surf, "seen", "empty"))
            vr = real - cam
            dr = np.linalg.norm(vr, axis=1)
            ur = vr / dr[:, None]
            vc = cells - cam
            dc = np.linalg.norm(vc, axis=1)
            uc = vc / dc[:, None]
            tree = cKDTree(ur)
            nb = tree.query_ball_point(uc, r=2 * np.sin(ANG / 2))
            lab = np.empty(n, object)
            for i, js in enumerate(nb):
                if not js:
                    lab[i] = "no-return"
                    continue
                d = dr[js]
                if np.any(np.abs(d - dc[i]) <= 2.0):
                    lab[i] = "consistent"
                elif d.min() < dc[i] - 2.0:
                    lab[i] = "occluded"
                else:
                    lab[i] = "beyond"
            if scene == "colosseum":
                grp = ("colosseum low plateau" if (low and on is not None and t >= on) else "colosseum low pre-plateau" if low else
                       "colosseum ok t<50" if t < 50 else "colosseum ok t>=50")
            else:
                grp = f"{scene} t<20" if t < 20 else f"{scene} t>=20"
            for g in (grp, "all t<20" if t < 20 else "all t>=20"):
                a = acc.setdefault(g, {})
                for cl in ("hit", "seen", "empty"):
                    for lb in ("occluded", "consistent", "beyond", "no-return"):
                        m = (cls == cl) & (lab == lb)
                        a[f"{cl}|{lb}"] = a.get(f"{cl}|{lb}", 0.0) + float(w[m].sum())
                a["total"] = a.get("total", 0.0) + float(w.sum())
                a["steps"] = a.get("steps", 0) + 1
    json.dump(acc, open(os.path.join(EX, "results", "occlusion_check.json"), "w"), indent=1)
    print("| 步集合 | 步数 | " + " | ".join(f"{cl}: 被遮挡 / 一致 / 在真实表面前 / 无回波" for cl in ("命中", "已见", "空")) + " |")
    for g, a in sorted(acc.items()):
        T = a["total"]
        print(f"| {g} | {a['steps']} | " + " | ".join(" / ".join(f"{a[f'{cl}|{lb}'] / T:.2f}" for lb in ("occluded", "consistent", "beyond", "no-return")) for cl in ("hit", "seen", "empty")) + " |")


if __name__ == "__main__":
    main()
