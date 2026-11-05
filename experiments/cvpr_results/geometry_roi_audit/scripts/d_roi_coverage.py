"""Geometry / ROI audit D (seen2scene env, CPU, read-only): DIAGNOSTIC split of the coverage at step 99 of the Table 1 episodes of start 0
(V3-D1-H3-NORISK and V3-DV2-KSV) into GT target points inside / outside the planner's valid P1 domain.
NOT the official metric and never written to Table 1: 'uncapped per-point coverage' = share of the official GT target points (a1,
frozen GT seed of start 0) that have one of the episode's 10 % sample points (the points the official evaluator feeds to its covered
scene; only those inside the ROI box, as fill_cells does) within eps = 1.0. The official number additionally caps both point sets at
1000 points per coarse cell with random replacement and matches only inside the same coarse cell; both values are divided by the
scene's visibility_ratio. The official final coverage and the stored monotone coverage are printed next to it for reference.
Output results/d_roi_coverage.json (+ .csv)."""
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
X = os.path.join(REPO, "experiments")
sys.path.insert(0, os.path.join(X, "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402

A1 = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")
EP = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
ALL14 = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer", "barts", "colosseum", "eiffel", "pantheon", "sestino_museum",
         "alhambra", "bridge", "neuschwanstein"]


def main():
    a1 = json.load(open(os.path.join(AUD, "results", "a1_gt_target.json")))
    out = {}
    for s in ALL14:
        z = np.load(os.path.join(A1, f"{s}.npz"))
        gt = z["gt_points"].astype(np.float64)
        lo, hi = z["box_lo"].astype(np.float64), z["box_hi"].astype(np.float64)
        fr, dims, dom, _ = TD.scene_geometry(s)
        ijk = np.floor(fr.runtime_to_model(gt) / MC.VS).astype(np.int64)
        ing = ((ijk >= 0) & (ijk < np.asarray(dims)[None])).all(-1)
        q = np.clip(ijk, 0, np.asarray(dims)[None] - 1)
        ind = ing & dom[0][q[:, 0]] & dom[1][q[:, 1]] & dom[2][q[:, 2]]
        vis = a1[s]["roi_box"]["visibility_ratio"]
        out[s] = {"n_gt": int(len(gt)), "gt_in_domain_share": float(ind.mean()), "visibility_ratio": vis}
        for m in ("V3-D1-H3-NORISK", "V3-DV2-KSV"):
            r = json.load(open(os.path.join(EP, m, f"{s}_s0.json")))
            e = json.load(open(os.path.join(REPO, r["episode_file"])))
            root = os.path.dirname(os.path.dirname(os.path.join(REPO, r["episode_file"])))
            pts = []
            for t in range(100):                 # service records exist for t = 0..99 (the last observation has no planner call)
                rec = np.load(os.path.join(root, "svc", e["tag"], f"svc_{s}_s0_t{t}.npz"))
                sp = np.asarray(rec["sample_points"], np.float64).reshape(-1, 3)
                pts.append(sp[((sp >= lo) & (sp <= hi)).all(-1)])
            pts = np.concatenate(pts)
            d, _ = cKDTree(pts).query(gt, k=1, distance_upper_bound=1.0, workers=8)
            cov = np.isfinite(d)
            out[s][m] = {"official_final": float(r["coverage_official"][99]), "mono_final": float(r["coverage_mono"][99]) if r.get("coverage_mono") else None,
                         "uncapped_all": float(cov.mean() / vis), "uncapped_in_domain_raw": float(cov[ind].mean()),
                         "uncapped_out_domain_raw": float(cov[~ind].mean()) if (~ind).any() else None,
                         "uncapped_in_domain_only_over_vis": float(cov[ind].mean() / vis), "s10_points_in_box": int(len(pts))}
        o = out[s]
        print(f"{s:15s} GT {o['n_gt']} in-domain {o['gt_in_domain_share']:.4f} vis {vis:.3f} | " + " | ".join(
            f"{m[:6]}: official {o[m]['official_final']:.3f} mono {o[m]['mono_final'] if o[m]['mono_final'] is None else round(o[m]['mono_final'], 3)} uncapped {o[m]['uncapped_all']:.3f} "
            f"in-dom raw {o[m]['uncapped_in_domain_raw']:.3f} out-dom raw {o[m]['uncapped_out_domain_raw'] if o[m]['uncapped_out_domain_raw'] is None else round(o[m]['uncapped_out_domain_raw'], 3)} "
            f"in-dom-only {o[m]['uncapped_in_domain_only_over_vis']:.3f}" for m in ("V3-D1-H3-NORISK", "V3-DV2-KSV")), flush=True)
    json.dump(out, open(os.path.join(AUD, "results", "d_roi_coverage.json"), "w"), indent=1)
    with open(os.path.join(AUD, "results", "d_roi_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "model", "n_gt", "gt_in_domain_share", "official_final", "mono_final", "uncapped_all", "uncapped_in_domain_raw", "uncapped_out_domain_raw", "uncapped_in_domain_only_over_vis"])
        for s, o in out.items():
            for m in ("V3-D1-H3-NORISK", "V3-DV2-KSV"):
                x = o[m]
                w.writerow([s, m, o["n_gt"], o["gt_in_domain_share"], x["official_final"], x["mono_final"], x["uncapped_all"], x["uncapped_in_domain_raw"], x["uncapped_out_domain_raw"], x["uncapped_in_domain_only_over_vis"]])


if __name__ == "__main__":
    main()
