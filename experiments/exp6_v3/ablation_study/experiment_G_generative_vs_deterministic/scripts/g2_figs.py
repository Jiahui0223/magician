"""Experiment G2 item 4 -- qualitative figure from REAL model outputs (protocol G2-v1 section 4; CPU only). Inputs: g2_pool.py figure arrays
(work_dirs/exp6_v3/experiment_G/g2/pool/<tag>_fig.npz: per state the first-layer view with the largest GT-only ambiguity score
a(v) = #labelled UNKNOWN rays in C-edge u D-edge). Selection rule (frozen): all (trajectory, t) pairs in (tag, t) order, permuted with seed 0,
stably sorted by the state score A descending, then taken greedily with at most one state per scene until N = 4. No model output is used.
Panels per state: GT first-hit z*, D0 depth, G samples k0..k3 depth, number of samples hitting in the UNKNOWN region with the GT-hit outline
(+ markers: GT hit missed by D0 but hit by >= 1 sample; samples disagree), top-down back-projected points (GT, D0, samples).
Writes figures/g2/qualitative_g2.{pdf,png}, figures/g2/qualitative_state_<i>.{pdf,png}, results/g2/qualitative_selection.json."""
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
WP = os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2/pool")
FW = os.path.join(R, "work_dirs/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/states")
FIG, OUT = os.path.join(GD, "figures/g2"), os.path.join(GD, "results/g2")
N = 4
GH, GW = 64, 114
SCOL = ["#c0392b", "#e67e22", "#27ae60", "#8e44ad"]


def scene_of(tag):
    p = os.path.join(FW, f"{tag}.jsonl")
    with open(p) as f:
        return json.loads(f.readline())["scene"]


def load_all():
    cands = []
    for f in sorted(glob.glob(os.path.join(WP, "*_fig.npz"))):
        tag = os.path.basename(f)[:-8]
        z = np.load(f)
        ts = sorted({k.split("__")[0] for k in z.files}, key=lambda s: int(s[1:]))
        sc = scene_of(tag)
        for t in ts:
            cands.append({"tag": tag, "t": int(t[1:]), "scene": sc, "score": int(z[f"{t}__score"]), "file": f})
    return cands


def select(cands):
    cands = sorted(cands, key=lambda c: (c["tag"], c["t"]))
    perm = np.random.default_rng(0).permutation(len(cands))
    order = sorted([cands[i] for i in perm], key=lambda c: -c["score"])      # Python sort is stable: ties keep the seed-0 order
    out, seen = [], set()
    for c in order:
        if c["scene"] in seen:
            continue
        out.append(c)
        seen.add(c["scene"])
        if len(out) == N:
            break
    return out


def arrays(c):
    z = np.load(c["file"])
    p = f"t{c['t']}__"
    return {k[len(p):]: z[k] for k in z.files if k.startswith(p)}


def img(x):
    return np.asarray(x, np.float64).reshape(GH, GW)


def draw_row(axs, c, a, row_label=True):
    import matplotlib.colors as mcolors
    lm = a["lm"].astype(bool)
    gt_hit = (a["b_gt"] == 1) & lm
    zs = np.where(np.isfinite(a["zstar"]), a["zstar"], np.nan)
    fin = zs[np.isfinite(zs)]
    vmin, vmax = (np.percentile(fin, 2), np.percentile(fin, 98)) if len(fin) else (0, 1)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#d9d9d9")
    norm = mcolors.Normalize(vmin, vmax)
    unk = a["n_u"] > 0

    def depth(ax, zz, hitu, title):
        zz = np.where(np.isfinite(zz), zz, np.nan)
        ax.imshow(img(zz), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=7)
    depth(axs[0], zs, None, "GT first hit z*")
    depth(axs[1], a["z_d0"], a["hitu_d0"], "D0 (single prediction)")
    for k in range(4):
        depth(axs[2 + k], a["z_g4"][k], a["hitu_g4"][k], f"Generative sample k={k}")
    nh = a["hitu_g4"].astype(int).sum(0)
    ax = axs[6]
    cm2 = matplotlib.colors.ListedColormap(["#f7f7f7", "#fde0c5", "#facba6", "#f59e72", "#e8513b"])
    ax.imshow(np.where(img(unk), img(nh), np.nan), cmap=cm2, vmin=-0.5, vmax=4.5, interpolation="nearest")
    ax.contour(img(gt_hit.astype(float)), levels=[0.5], colors="k", linewidths=0.6)
    miss = gt_hit & ~a["hitu_d0"].astype(bool) & (nh > 0)
    dis = unk & (nh > 0) & (nh < 4)
    yy, xx = np.nonzero(img(miss))
    ax.scatter(xx, yy, s=1.2, c="#2471a3", marker="s", lw=0, label="GT hit, D0 misses, ≥1 sample hits")
    ax.set_title(f"# samples hitting (0–4); disagree {int(dis.sum())} rays", fontsize=7)
    ax = axs[7]
    def sc_(p, color, s, lab, alpha=0.8):
        p = np.asarray(p, np.float64)
        ok = np.isfinite(p).all(1)
        if ok.any():
            ax.scatter(p[ok, 0], p[ok, 2], s=s, c=color, lw=0, alpha=alpha, label=lab)
    sel_gt = np.where(gt_hit[:, None], a["p_gt"], np.nan)
    sc_(sel_gt, "k", 1.0, "GT", 0.6)
    for k in range(4):
        sc_(a["p_g4"][k], SCOL[k], 0.8, f"k={k}", 0.5)
    sc_(a["p_d0"], "#2471a3", 1.2, "D0", 0.9)
    cc = a["c_rt"]
    ax.scatter([cc[0]], [cc[2]], marker="^", s=18, c="k")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("back-projected hits (top-down x–z)", fontsize=7)
    ax.tick_params(labelsize=5)
    for ax in axs[:7]:
        ax.set_xticks([]); ax.set_yticks([])
    if row_label:
        axs[0].set_ylabel(f"{c['scene']} t={c['t']}\n(A={c['score']})", fontsize=7)
    return norm, cmap


import matplotlib  # noqa: E402
matplotlib.use("Agg")


def main():
    import argparse
    import matplotlib.pyplot as plt
    global WP, FIG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None, help="test only")
    ap.add_argument("--out", default=None, help="test only")
    a_ = ap.parse_args()
    if a_.pool:
        WP = a_.pool
    if a_.out:
        FIG = OUT = a_.out
    cands = load_all()
    sel = select(cands)
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rule": "seed-0 permutation of (tag, t) order, stable sort by GT-only score A desc, <= 1 state per scene, N = 4", "n_candidates": len(cands),
               "selected": [{k: v for k, v in c.items() if k != "file"} for c in sel],
               "top10_scores": sorted([c["score"] for c in cands], reverse=True)[:10]}, open(os.path.join(OUT, "qualitative_selection.json"), "w"), indent=1)
    fig, axes = plt.subplots(len(sel), 8, figsize=(19, 2.15 * len(sel)), gridspec_kw={"width_ratios": [1] * 7 + [1.05]})
    axes = np.atleast_2d(axes)
    for i, c in enumerate(sel):
        a = arrays(c)
        norm, cmap = draw_row(axes[i], c, a)
        f1, ax1 = plt.subplots(1, 8, figsize=(19, 2.4), gridspec_kw={"width_ratios": [1] * 7 + [1.05]})
        draw_row(ax1, c, a)
        ax1[7].legend(fontsize=5, frameon=False, markerscale=4, loc="best")
        f1.tight_layout()
        for ext in ("pdf", "png"):
            f1.savefig(os.path.join(FIG, f"qualitative_state_{i}.{ext}"), dpi=220)
        plt.close(f1)
    axes[0, 7].legend(fontsize=5, frameon=False, markerscale=4, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"qualitative_g2.{ext}"), dpi=220)
    print(json.dumps([{k: v for k, v in c.items() if k != "file"} for c in sel], indent=1))


if __name__ == "__main__":
    main()
