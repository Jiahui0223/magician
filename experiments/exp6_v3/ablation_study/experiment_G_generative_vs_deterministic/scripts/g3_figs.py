"""Experiment G3 -- qualitative figure from REAL model outputs: the averaged map vs the 4 samples (protocol G3-v1 section 8; CPU only).
Inputs: g3_fixed.py figure arrays (work_dirs/exp6_v3/experiment_G/g3/fixed/<tag>_fig.npz: per state the first-layer view with the most
M-depth floating rays). Frozen rule: all (trajectory, t) in (tag, t) order, seed-0 permutation, stable sort by the M-depth floating count
(S & T) descending, greedily <= 1 state per scene, N = 4. NOTE: chosen by the failure mode of averaging (a model-output criterion) --
illustrative, not representative. Panels: GT z*, samples k0..k3, M-raw, M-depth, M-depth floating rays over the sample-agreement map,
top-down back-projected points (GT, samples, M-depth).
Writes figures/g3/averaging_vs_samples.{pdf,png}, figures/g3/averaging_state_<i>.{pdf,png}, results/g3/qualitative_selection.json."""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
FW = os.path.join(R, "work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/states")
GH, GW, N = 64, 114, 4
SCOL = ["#c0392b", "#e67e22", "#27ae60", "#8e44ad"]


def scene_of(tag):
    with open(os.path.join(FW, f"{tag}.jsonl")) as f:
        return json.loads(f.readline())["scene"]


def img(x):
    return np.asarray(x, np.float64).reshape(GH, GW)


def draw_row(axs, c, a):
    import matplotlib.colors as mcolors
    zs = np.where(np.isfinite(a["zstar"]), a["zstar"], np.nan)
    fin = zs[np.isfinite(zs)]
    vmin, vmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (0, 1)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    norm = mcolors.Normalize(vmin, vmax)

    def depth(ax, zz, title):
        ax.imshow(img(np.where(np.isfinite(zz), zz, np.nan)), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    depth(axs[0], zs, "GT first hit z*")
    for k in range(4):
        depth(axs[1 + k], a["z_g4"][k], f"sample k={k}")
    depth(axs[5], a["z_mraw"], "M-raw (mean b, y)")
    depth(axs[6], a["z_mdepth"], "M-depth (vote, mean depth)")
    unk = a["n_u"] > 0
    nh = a["hitu_g4"].astype(int).sum(0)
    cm2 = matplotlib.colors.ListedColormap(["#f7f7f7", "#fde0c5", "#facba6", "#f59e72", "#e8513b"])
    ax = axs[7]
    ax.imshow(np.where(img(unk), img(nh), np.nan), cmap=cm2, vmin=-0.5, vmax=4.5, interpolation="nearest")
    yy, xx = np.nonzero(img(a["float_mdepth"].astype(bool)))
    ax.scatter(xx, yy, s=1.5, c="#1f3a93", marker="s", lw=0)
    ax.set_title(f"# samples hitting; blue = M-depth floating ({int(a['float_mdepth'].sum())})", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[8]

    def sc_(p, color, s, lab, alpha=0.8):
        p = np.asarray(p, np.float64)
        ok = np.isfinite(p).all(1)
        if ok.any():
            ax.scatter(p[ok, 0], p[ok, 2], s=s, c=color, lw=0, alpha=alpha, label=lab)
    sc_(a["p_gt"], "k", 0.8, "GT", 0.5)
    for k in range(4):
        sc_(a["p_g4"][k], SCOL[k], 0.7, f"k={k}", 0.45)
    fl = a["float_mdepth"].astype(bool)
    sc_(np.where(fl[:, None], a["p_mdepth"], np.nan), "#1f3a93", 2.0, "M-depth floating", 0.95)
    cc = a["c_rt"]
    ax.scatter([cc[0]], [cc[2]], marker="^", s=18, c="k")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("top-down x–z: GT, samples, floating averaged hits", fontsize=7)
    ax.tick_params(labelsize=5)
    axs[0].set_ylabel(f"{c['scene']} t={c['t']}\n(floating {c['score']})", fontsize=7)


def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default=os.path.join(R, "work_dirs/exp6_v3/experiment_G/g3/fixed"))
    ap.add_argument("--out", default=None, help="test only (figures + selection)")
    a_ = ap.parse_args()
    FIG = a_.out or os.path.join(GD, "figures/g3")
    OUT = a_.out or os.path.join(GD, "results/g3")
    cands = []
    for f in sorted(glob.glob(os.path.join(a_.fixed, "*_fig.npz"))):
        tag = os.path.basename(f)[:-8]
        z = np.load(f)
        for t in sorted({k.split("__")[0] for k in z.files}, key=lambda s: int(s[1:])):
            cands.append({"tag": tag, "t": int(t[1:]), "scene": scene_of(tag), "score": int(z[f"{t}__score"]), "file": f})
    cands = sorted(cands, key=lambda c: (c["tag"], c["t"]))
    perm = np.random.default_rng(0).permutation(len(cands))
    order = sorted([cands[i] for i in perm], key=lambda c: -c["score"])
    sel, seen = [], set()
    for c in order:
        if c["scene"] not in seen:
            sel.append(c)
            seen.add(c["scene"])
        if len(sel) == N:
            break
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rule": "seed-0 permutation of (tag, t), stable sort by M-depth floating count desc, <= 1 per scene, N = 4 (illustrative: failure-mode selection)",
               "n_candidates": len(cands), "selected": [{k: v for k, v in c.items() if k != "file"} for c in sel],
               "score_quantiles": {q: float(np.quantile([c["score"] for c in cands], q)) for q in (0.5, 0.9, 0.99)} if cands else None},
              open(os.path.join(OUT, "qualitative_selection.json"), "w"), indent=1)
    fig, axes = plt.subplots(len(sel), 9, figsize=(21, 2.15 * len(sel)), gridspec_kw={"width_ratios": [1] * 8 + [1.05]})
    axes = np.atleast_2d(axes)
    for i, c in enumerate(sel):
        z = np.load(c["file"])
        p = f"t{c['t']}__"
        arr = {k[len(p):]: z[k] for k in z.files if k.startswith(p)}
        draw_row(axes[i], c, arr)
        f1, ax1 = plt.subplots(1, 9, figsize=(21, 2.4), gridspec_kw={"width_ratios": [1] * 8 + [1.05]})
        draw_row(ax1, c, arr)
        ax1[8].legend(fontsize=5, frameon=False, markerscale=4)
        f1.tight_layout()
        for ext in ("pdf", "png"):
            f1.savefig(os.path.join(FIG, f"averaging_state_{i}.{ext}"), dpi=220)
        plt.close(f1)
    axes[0, 8].legend(fontsize=5, frameon=False, markerscale=4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"averaging_vs_samples.{ext}"), dpi=220)
    print(json.dumps([{k: v for k, v in c.items() if k != "file"} for c in sel], indent=1))


if __name__ == "__main__":
    main()
