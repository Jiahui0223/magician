"""EXP9 Fig. 4 v7 -- every ELIGIBLE state of one scene (any start / step), for browsing a scene the user liked (2026-11-16: Neuschwanstein,
"the current view at t0 carries too little information -> show later steps").  Same pool / eligibility / v6 window / render guards / v5
drawing as fig4_v7.py (reused); rows ordered by (start, t).  Ineligible states (UNKNOWN share < 0.20, inside the mesh, GT outside the
ROI box) are not shown.  Usage: rank+render: CUDA_VISIBLE_DEVICES=0 $MAGPY -I -B fig4_v7_scene.py render --scene S;
sheet: $PY -I -B fig4_v7_scene.py sheet --scene S  ->  fig/fig4/fig4_v7/scene_<S>_<k>.{pdf,png}"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig4_v7 as V7  # noqa: E402

FS, V6 = V7.FS, V7.V6


def states(scene):
    rows = [r for r in csv.DictReader(open(V7.OUT + "ranking_v7.csv")) if r["group"] == scene and int(r["v6_score"]) > 0]
    rows.sort(key=lambda r: (int(r["start"]), int(r["t"])))
    return [{"domain": "outdoor", "group": r["group"], "start": int(r["start"]), "t": int(r["t"]), "case": r["case"], "pick": int(r["pick"]),
             "win": (int(r["win_row0"]), int(r["win_col0"])), "v6_score": int(r["v6_score"]), "flip": float(r["flip_share_on_unknown"]),
             "unknown": float(r["unknown_share"])} for r in rows]


def render(scene):
    import fig4_v4_render as R4
    R4.W4 = V7.W7
    dev = R4.torch.device("cuda", 0)
    sc = {}
    for d in states(scene):
        if not os.path.exists(V7.W7 + f"{d['case']}.npz"):
            R4.run_case(dict(d, sets=["v7_scene"]), dev, sc)


def sheet(scene):
    import fig4_v5 as V5
    rows = []
    for d in states(scene):
        _, A, inb, code = V6.state_arrays(d)
        z = np.load(V7.W7 + f"{d['case']}.npz")
        rows.append({"meta": {"domain": "outdoor", "depth_unit": A["unit"], "case": d["case"]}, "cur_rgb": z["cur_rgb"], "cand_rgb": z["cand_rgb"],
                     "unk": A["unknown"], "gt": A["gt_depth"], "gtv": A["gt_valid"], "hyp": A["hyp_depth"], "hypv": A["hyp_valid"],
                     "vmin": A["vmin"], "vmax": A["vmax"], "code": code, "win": d["win"],
                     "label": f"{V7.NAME[scene]} s{d['start']} t{d['t']}"})
        print(f"{d['case']:32s} unknown {d['unknown']:.2f} flip {d['flip']:.3f} window score {d['v6_score']}")
    for i in range(0, len(rows), 3):
        V5.composite(rows[i:i + 3], V7.OUT + f"scene_{scene}_{i // 3 + 1}", True)
    V5.plt.close("all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["render", "sheet"])
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    {"render": render, "sheet": sheet}[a.phase](a.scene)
