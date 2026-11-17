"""EXP9 Fig. 6 v4 -- candidate table (CPU): every unseen candidate with Cov@20 of both methods (from the validated mask manifest),
delta = Ours - MAGICIAN, rank of delta within the scene / level group and within the domain pool, pool medians.  Lets the user pick
columns and lets the caption state the selection honestly.  Output fig/fig6/fig6_v4/candidates_cov20.{csv,md}."""
import csv
import os

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/"
V4 = REPO + "work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/"
OUT = REPO + "experiments/cvpr_results/fig/fig6/fig6_v4/"


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = list(csv.DictReader(open(V4 + "v4_cases.csv")))
    mm = {(x["case"], x["method"]): x for x in csv.DictReader(open(V4 + "f6v4_masks_manifest.csv"))}
    rows = []
    for r in cases:
        n = f"outdoor_{r['scene']}_s{r['start']}" if r["domain"] == "outdoor" else f"indoor_{r['scene']}_S{r['start']}"
        m, o = float(mm[(n, "MAGICIAN (MAG-B)")]["official_cov20"]), float(mm[(n, "Ours (V3-T0 + D1)")]["official_cov20"])
        grp = r["scene"] if r["domain"] == "outdoor" else r["scene"].rsplit("_L", 1)[0]
        rows.append({"case": n, "domain": r["domain"], "scene": r["scene"], "group": grp, "start": r["start"], "split": r["split"],
                     "cov20_magician": m, "cov20_ours": o, "delta": o - m})
    for dom in ("outdoor", "indoor"):
        pool = [x for x in rows if x["domain"] == dom]
        ds = sorted((x["delta"] for x in pool), reverse=True)
        for x in pool:
            x["rank_in_pool"] = f"{ds.index(x['delta']) + 1}/{len(pool)}"
            x["pool_median_delta"] = float(np.median(ds))
            g = sorted((y["delta"] for y in pool if y["group"] == x["group"]), reverse=True)
            x["rank_in_group"] = f"{g.index(x['delta']) + 1}/{len(g)}"
            x["group_median_delta"] = float(np.median(g))
    rows.sort(key=lambda x: (x["domain"] != "outdoor", x["group"], x["scene"], int(x["start"])))
    keys = list(rows[0].keys())
    with open(OUT + "candidates_cov20.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in x.items()})
    md = ["| case | split | MAGICIAN Cov@20 | Ours Cov@20 | Δ | rank in scene/house | rank in pool |", "|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['case']} | {x['split']} | {x['cov20_magician']:.3f} | {x['cov20_ours']:.3f} | {x['delta']:+.3f} | "
                  f"{x['rank_in_group']} | {x['rank_in_pool']} |")
    for dom in ("outdoor", "indoor"):
        pool = [x for x in rows if x["domain"] == dom]
        md.append(f"\n{dom}: {len(pool)} episodes, median Δ {np.median([x['delta'] for x in pool]):+.3f}, "
                  f"Ours > MAGICIAN in {sum(x['delta'] > 0 for x in pool)}/{len(pool)}")
    open(OUT + "candidates_cov20.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
