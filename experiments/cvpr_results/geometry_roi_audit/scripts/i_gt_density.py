"""Geometry / ROI audit I (CPU, read-only): how the official GT target weights the surface. Per scene and coarse evaluator cell
(settings grid_l x grid_w x grid_h): share of the target surface area (area-weighted mesh samples on target faces, a1) vs share of
the official GT target points (a1 cell counts; at most 1000 per cell). weight ratio = point share / area share (1 = area-uniform).
Output results/i_gt_density.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
A1 = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")


def main():
    a1 = json.load(open(os.path.join(AUD, "results", "a1_gt_target.json")))
    out = {}
    for s, o in a1.items():
        z = np.load(os.path.join(A1, f"{s}.npz"))
        lo, hi = z["box_lo"].astype(np.float64), z["box_hi"].astype(np.float64)
        g = np.asarray(o["roi_box"]["coarse_cells"])
        p = z["mesh_sample"][z["mesh_sample_target_face"]].astype(np.float64)
        ijk = np.minimum(np.floor((p - lo) / ((hi - lo) / g)).astype(int), g - 1)
        area = {}
        for r in ijk:
            k = str(r.tolist())
            area[k] = area.get(k, 0) + 1
        n_a, n_p = sum(area.values()), o["gt_target"]["n_points"]
        rat, at_cap_area = [], 0.0
        for k, c in o["gt_target"]["cell_counts"].items():
            a_ = area.get(k, 0) / n_a
            if c > 0 and a_ > 0:
                rat.append((c / n_p) / a_)
            if c >= o["roi_box"]["cell_capacity"]:
                at_cap_area += a_
        rat = np.asarray(rat)
        out[s] = {"cells_nonempty": int(o["gt_target"]["n_cells_nonempty"]), "cells_at_capacity": int(o["gt_target"]["n_cells_at_capacity"]),
                  "target_area_in_capped_cells": at_cap_area, "weight_ratio_min": float(rat.min()), "weight_ratio_median": float(np.median(rat)), "weight_ratio_max": float(rat.max()),
                  "target_area": o["faces"]["area_all3_inside"], "area_per_gt_point": o["faces"]["area_all3_inside"] / n_p}
        print(f"{s:15s} cells {out[s]['cells_at_capacity']}/{out[s]['cells_nonempty']} at cap, holding {at_cap_area:.2f} of target area | weight ratio min {rat.min():.2f} median {np.median(rat):.2f} max {rat.max():.2f} "
              f"| area per GT point {out[s]['area_per_gt_point']:.2f}")
    json.dump(out, open(os.path.join(AUD, "results", "i_gt_density.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
