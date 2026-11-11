"""CVPR Figure 3, dual panel (CPU): real coverage curves of the frozen closed loops.
  (a) Macarons++ outdoor -- All-14 x 5 official starts, 100 steps, official coverage, scene-equal mean (CVPR Table 1 canonical records,
      cvpr_lib.load / mean_curve).
  (b) MP3D indoor -- 22 val levels x 2 starts, 150 steps, level-scale surface coverage C_obs, level-equal mean (EXP7 canonical records
      work_dirs/cvpr_results/exp7/episodes/mp3d150/<METHOD>/<tag>.json, VALID only).
Shaded band = paired-free 95 % bootstrap CI over scenes / levels (10,000 draws, seed 0) of the mean curve. Output: fig3_dual.{pdf,png},
fig3_dual_data.csv (mean and CI per step) next to this script."""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
OUT_M = [("RANDOM", "Random Walk"), ("MACARONS", "MACARONS"), ("MAG-B", "MAGICIAN"), ("V3-T0-D1", "Ours")]
IN_M = [("RANDOM", "Random Walk"), ("MAG-B", "MAGICIAN"), ("OURS", "Ours")]
STY = {"Ours": RD.STYLE["V3-T0-D1"], "MAGICIAN": RD.STYLE["MAG-B"], "MACARONS": RD.STYLE["MACARONS"], "Random Walk": RD.STYLE["RANDOM"]}


def boot(per, n=10000):
    A = np.asarray(per)
    idx = np.random.default_rng(0).integers(0, len(A), (n, len(A)))
    b = A[idx].mean(1)
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def outdoor():
    res = {}
    for mid, lab in OUT_M:
        E = L.load(mid)
        m, per = L.mean_curve(E, L.SCENE_ORDER)
        if m is None:
            continue
        lo, hi = boot(list(per.values()))
        res[lab] = (m, lo, hi, len(per))
    return res


def indoor():
    res = {}
    for mid, lab in IN_M:
        by = {}
        for p in glob.glob(os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "mp3d150", mid, "*.json")):
            r = json.load(open(p))
            if r["valid"]:
                by.setdefault(r["scene_id"], []).append(np.asarray(r["coverage"], float))
        per = [np.mean(v, 0) for k, v in sorted(by.items()) if len(v) == 2]
        if not per:
            continue
        lo, hi = boot(per)
        res[lab] = (np.mean(per, 0), lo, hi, len(per))
    return res


def main():
    o, i = outdoor(), indoor()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, data, T, title, ylab in ((axs[0], o, 100, "(a) Macarons++ outdoor (14 scenes x 5 starts)", "official coverage"),
                                     (axs[1], i, 150, "(b) MP3D indoor (22 levels x 2 starts)", "surface coverage $C_{obs}$")):
        for lab, (m, lo, hi, n) in data.items():
            st = STY[lab]
            x = np.arange(len(m))
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.13, lw=0)
            ax.plot(x, m, color=st["color"], ls=st["ls"], lw=st["lw"], label=lab, marker=st["marker"], markevery=T // 5, ms=3.5)
        ax.set_xlim(0, T); ax.set_ylim(0, 1.0 if T == 100 else 0.8); ax.set_xlabel("step"); ax.set_ylabel(ylab)
        ax.set_title(title.replace(" x ", r" $\times$ "), fontsize=8.5); ax.grid(alpha=0.25, lw=0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.subplots_adjust(wspace=0.28)
    RD.save(fig, os.path.join(HERE, "fig3_dual"))
    with open(os.path.join(HERE, "fig3_dual_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "method", "n_units", "step", "mean", "ci95_lo", "ci95_hi"])
        for pan, data in (("outdoor", o), ("indoor", i)):
            for lab, (m, lo, hi, n) in data.items():
                for t in range(len(m)):
                    w.writerow([pan, lab, n, t, f"{m[t]:.5f}", f"{lo[t]:.5f}", f"{hi[t]:.5f}"])
    print({k: (round(float(v[0][-1]), 3), v[3]) for k, v in o.items()}, {k: (round(float(v[0][-1]), 3), v[3]) for k, v in i.items()})


if __name__ == "__main__":
    main()
