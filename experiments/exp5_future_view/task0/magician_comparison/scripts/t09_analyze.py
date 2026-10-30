"""EXP5 Task 0 / T0.9 -- aggregation of the 15 official MAGICIAN runs (5 mp3d scenes x 3 repeats, released-code
protocol). Per scene: mean and SD over the repeats; overall: mean over scenes (each scene weighted once). CPU.
Output W/magician_comparison/analysis/t09.json + figures/t09_curves.png
"""
import glob
import json
import os

import numpy as np

MC = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/magician_comparison"
KEYS = ("official_auc_raw_first100", "official_max_first100", "cov100", "cov150", "auc100_task0", "auc150_norm", "wall_s", "peak_gpu_mem_gb")
PAPER = {"drone_comp_pct": 96.83, "drone_comp_cm": 2.11, "wheeled_comp_pct": 85.45, "wheeled_comp_cm": 4.93,
         "macaronspp_auc": 0.721, "macaronspp_final": 0.919}


def flat(r):
    o = {k: r.get(k) for k in KEYS}
    for T in ("100", "150"):
        c = r.get("comp_frames", {}).get(T)
        if c:
            o[f"comp5_{T}"], o[f"dist_{T}"], o[f"distmed_{T}"], o[f"far_{T}"] = c["comp5"], c["dist_mean_cm"], c["dist_median_cm"], c["frac_far_1m"]
    o["comp5_lmdb"], o["dist_lmdb"] = r.get("comp5_lmdb_points"), r.get("dist_mean_cm_lmdb_points")
    o["lmdb_check_median_rt"] = r.get("lmdb_points_on_frame_cloud_median_rt")
    return o


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(MC, "official_reproduction", "metrics", "formal_*.json")))]
    by = {}
    for r in runs:
        by.setdefault(r["scene"], []).append(r)
    per_scene = {}
    for s, rs in sorted(by.items()):
        F = [flat(r) for r in rs]
        per_scene[s] = {"n_reps": len(rs), "metric_scale": rs[0]["metric_scale"], "n_gt_vertices": rs[0].get("n_gt_vertices"),
                        **{k: {"mean": float(np.mean([f[k] for f in F if f.get(k) is not None])),
                               "sd": float(np.std([f[k] for f in F if f.get(k) is not None], ddof=1)) if len(F) > 1 else None,
                               "values": [f.get(k) for f in F]} for k in F[0] if any(f.get(k) is not None for f in F)}}
    keys = sorted({k for v in per_scene.values() for k in v if isinstance(v[k], dict) and "mean" in v[k]})
    overall = {k: float(np.mean([v[k]["mean"] for v in per_scene.values() if k in v])) for k in keys}
    out = {"doc": __doc__, "paper_reference": PAPER, "per_scene": per_scene, "overall_mean_over_scenes": overall, "n_runs": len(runs),
           "mean_curve": np.mean([r["coverage"][:151] for r in runs], 0).tolist()}
    json.dump(out, open(os.path.join(MC, "analysis", "t09.json"), "w"), indent=1)
    for s, v in per_scene.items():
        print(s, {k: (round(v[k]["mean"], 3), None if v[k]["sd"] is None else round(v[k]["sd"], 3)) for k in ("cov100", "cov150", "official_auc_raw_first100", "comp5_100", "comp5_150", "dist_100", "dist_150", "wall_s") if k in v},
              "scale", v["metric_scale"]["valid"], round(v["metric_scale"]["validation_median_nn_cm"], 4))
    print("overall", {k: round(v, 3) for k, v in overall.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in runs:
        ax.plot(r["coverage"], alpha=0.5, lw=1, label=f"{r['scene']} rep{r['rep']}" if r["rep"] == 0 else None)
    ax.axvline(100, color="k", ls=":", lw=1)
    ax.set_xlabel("pose")
    ax.set_ylabel("MAGICIAN coverage (released code)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(MC, "figures"), exist_ok=True)
    fig.savefig(os.path.join(MC, "figures", "t09_curves.png"), dpi=110)


if __name__ == "__main__":
    main()
