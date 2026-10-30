"""EXP5 Task 1 -- visualisations of the T1.4 closed loop on Dev-3 (seen2scene env + ffmpeg; descriptive; reads finished
outputs only, nothing is re-planned). Added 2026-10-25 at your request.

Drawing code of the EXP4 ablation (abl_1/common/abl_viz.py) is imported READ-ONLY and pointed at the Task 1 output
directory; its own files and outputs are not touched. Methods:
  B0       = EXP4 abl_1 A2 (T1-C, K = 4 persistent 3D worlds + DistP1 + Greedy), r0 episodes and EXP4 replays
  MAGICIAN = T7a MAGICIAN: curves / AUC = 3-seed mean per start (as in all tables); maps and animations = seed 0
  V2K4     = T1.4 E1 (V2, K = 4, R10, DistP1 + Greedy); curves / AUC = 3-noise-repeat mean; maps / animations = r0
  V2K1     = T1.4 E6 (same, K = 1); B2 (K = 1 world) and E2-E5 appear in the AUC figures only
Replays of the V2 r0 episodes (W/_viz/replay, W/_viz/covrep; run_replays.sh) reproduce every stored curve bitwise.
  --auc     figures/auc_area.png, auc_by_scene.png, auc_heatmap.png, paired_dots.png, cost.png
  --cover   figures/cover/cover_<scene>_s<k>.png for all 15 starts (B0 / MAGICIAN / V2K4 / V2K1, top + front view)
  --anim S:K[,S:K]  anim/anim_<scene>_s<k>.{mp4,gif}
Output W = work_dirs/exp5_future_view/task1/_viz/
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(REPO, "experiments", "exp4_multiworld_belief", "ablation_study", "abl_1", "common"))
import abl_viz as AV  # noqa: E402

plt, matplotlib = AV.plt, AV.matplotlib
T1 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task1")
OUT = os.path.join(T1, "_viz")
AV.FIG, AV.ANIM = os.path.join(OUT, "figures"), os.path.join(OUT, "anim")
ARMS = {"E1": "E1_V2G_R10_K4", "E2": "E2_V2G_Nat_K4", "E3": "E3_V2Uraw_R10_K4", "E4": "E4_V2DistU_R10_K4",
        "E5": "E5_V2GU_R10_K4", "E6": "E6_V2G_R10_K1"}
AV.COL.update({"A2": "#8f6b4a", "E1": "#eb6834", "E6": "#f2a07b"})
AV.LAB.update({"A2": "B0: T1-C 4 worlds + DistP1", "magician": "MAGICIAN", "E1": "V2 K4 + DistP1 (ours)", "E6": "V2 K1 + DistP1"})
AV.SHORT.update({"A2": "B0", "E1": "V2K4", "E6": "V2K1"})
_rep0, _cov0 = AV.rep_file, AV.cov_file


def rep_file(m, s, k):
    return os.path.join(OUT, "replay", f"viz_{m}_{s}_s{k}_r0.npz") if m in ARMS else _rep0(m, s, k)


def cov_file(m, s, k):
    return os.path.join(OUT, "covrep", f"cov_{m}_{s}_s{k}_r0.npz") if m in ARMS else _cov0(m, s, k)


AV.rep_file, AV.cov_file = rep_file, cov_file
KEYS = AV.KEYS
SC = ("barts", "pantheon", "colosseum")
INK, INK2, MUTED, GRIDC = AV.INK, AV.INK2, AV.MUTED, AV.GRID
STYLE = {"B0": ("#8f6b4a", "B0: T1-C 4 worlds + DistP1"), "B2": ("#c3b59b", "B2: T1-C 1 world + DistP1"),
         "MAGICIAN": ("#0b0b0b", "MAGICIAN (beam)"), "E1": ("#eb6834", "V2 K4 + DistP1 (ours, E1)"),
         "E6": ("#f2a07b", "V2 K1 + DistP1 (E6)"), "E2": ("#2a78d6", "E2 V2 refresh every move"),
         "E3": ("#898781", "E3 V2 + U_raw (no distance)"), "E4": ("#1baf7a", "E4 V2 + DistU"), "E5": ("#6250d6", "E5 V2 + DistP1 & DistU")}


def check_gtpts():
    for s, k in KEYS:
        p = os.path.join(OUT, "covrep", f"gtpts_{s}_s{k}.npz")
        if os.path.exists(p):
            assert np.array_equal(np.load(p)["pts"], AV.gtpts(s, k)), f"GT sample differs for {s} s{k}"


def curves():
    """Per method and start: the coverage curve used in the tables (baselines from t14.json, arms = mean of 3 repeats)."""
    r = json.load(open(os.path.join(T1, "analysis", "t14.json")))
    C = {m: {(s, k): np.asarray(r["baselines"][m][f"{s}/{k}"]["curve"]) for s, k in KEYS} for m in ("B0", "B2", "MAGICIAN")}
    for a, d in ARMS.items():
        C[a] = {(s, k): np.mean([json.load(open(os.path.join(T1, "closed_loop", d, "episodes", f"ep_{a}_{s}_s{k}_r{rr}.json")))["coverage"]
                                 for rr in range(3)], 0) for s, k in KEYS}
    return C, r


def auc(c):
    return float(np.trapz(c, dx=1.0) / 100)


def save(fig, name):
    AV.save(fig, name)


def fig_auc_area(C):
    ms = ["B0", "B2", "MAGICIAN", "E6", "E1"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(AV.W, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    for m in ms:
        c = np.mean([C[m][x] for x in KEYS], 0)
        col, lab = STYLE[m]
        a1.plot(c, color=col, lw=2.6 if m == "E1" else 1.8, ls="--" if m in ("B0", "B2") else "-",
                label=f"{lab}   AUC {np.mean([auc(C[m][x]) for x in KEYS]):.3f} · Cov@100 {np.mean([C[m][x][100] for x in KEYS]):.3f}")
        if m == "E1":
            a1.fill_between(range(101), 0, c, color=col, alpha=0.10)
    a1.set_xlim(0, 100)
    a1.set_ylim(0, 1)
    a1.set_xlabel("move")
    a1.set_ylabel("coverage (mean of 15 Dev-3 starts)")
    a1.grid(color=GRIDC)
    a1.legend(loc="lower right", frameon=False, fontsize=8.6)
    a1.set_title("Coverage curves; shaded area / 100 = AUC@100 of V2K4", loc="left", fontsize=11)
    e1 = np.mean([C["E1"][x] for x in KEYS], 0)
    for m in ("B0", "MAGICIAN"):
        d = e1 - np.mean([C[m][x] for x in KEYS], 0)
        a2.plot(d, color=STYLE[m][0], lw=2, label=f"V2K4 − {m}  (area/100 = {d[:101].sum() / 100 - (d[0] + d[100]) / 200:+.3f})")
        a2.fill_between(range(101), 0, d, color=STYLE[m][0], alpha=0.10)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlim(0, 100)
    a2.set_xlabel("move")
    a2.set_ylabel("coverage difference")
    a2.grid(color=GRIDC)
    a2.legend(loc="upper right", frameon=False, fontsize=8.6)
    a2.set_title("Where V2K4 gains: early moves (10-40)", loc="left", fontsize=11)
    save(fig, "auc_area.png")


def fig_auc_by_scene(C):
    ms = ["B0", "MAGICIAN", "E6", "E1"]
    fig, axs = plt.subplots(1, 3, figsize=(AV.W, 4.0), sharey=True)
    for ax, s in zip(axs, SC):
        for m in ms:
            c = np.mean([C[m][(s, k)] for k in range(5)], 0)
            ax.plot(c, color=STYLE[m][0], lw=2.4 if m == "E1" else 1.6, ls="--" if m == "B0" else "-",
                    label=f"{STYLE[m][1].split(' (')[0].split(':')[0]}  {np.mean([auc(C[m][(s, k)]) for k in range(5)]):.3f}")
        ax.set_title(f"{s} (5 starts) · legend = AUC@100", loc="left", fontsize=10.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel("move")
        ax.grid(color=GRIDC)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axs[0].set_ylabel("coverage")
    save(fig, "auc_by_scene.png")


def fig_heatmap(C):
    ms = ["B0", "B2", "MAGICIAN", "E1", "E2", "E3", "E4", "E5", "E6"]
    A = np.asarray([[auc(C[m][x]) for m in ms] for x in KEYS])
    fig, ax = plt.subplots(figsize=(AV.W * 0.8, 6.4))
    im = ax.imshow(A, cmap="viridis", aspect="auto", vmin=A.min(), vmax=A.max())
    for i in range(len(KEYS)):
        for j in range(len(ms)):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if A[i, j] < A.mean() else INK)
        j = int(np.argmax(A[i]))
        ax.add_patch(matplotlib.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2.2))
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(["B0", "B2", "MAGICIAN", "V2K4\n(E1)", "E2\nNative", "E3\nU_raw", "E4\nDistU", "E5\nGU", "V2K1\n(E6)"], fontsize=9)
    ax.set_yticks(range(len(KEYS)))
    ax.set_yticklabels([f"{s} s{k}" for s, k in KEYS], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, label="AUC@100")
    ax.set_title("AUC@100 per start (V2 arms: mean of 3 noise repeats; MAGICIAN: mean of 3 seeds); orange box = best in row",
                 loc="left", fontsize=10.5)
    save(fig, "auc_heatmap.png")


def fig_paired(C, r):
    rows = [("E1", "B0", r["arms"]["E1"]["vs_B0"]["auc"], "V2K4 − B0"), ("E1", "MAGICIAN", r["arms"]["E1"]["vs_MAGICIAN"]["auc"], "V2K4 − MAGICIAN"),
            ("E1", "B2", r["arms"]["E1"]["vs_B2"]["auc"], "V2K4 − B2"), ("E6", "E1", r["arm_pairs"]["E6-E1"]["auc"], "K1 − K4 (E6 − E1)"),
            ("E2", "E1", r["arm_pairs"]["E2-E1"]["auc"], "refresh every move − R10 (E2 − E1)"),
            ("E4", "E1", r["arm_pairs"]["E4-E1"]["auc"], "DistU − DistP1 (E4 − E1)"),
            ("E5", "E1", r["arm_pairs"]["E5-E1"]["auc"], "DistP1 & DistU − DistP1 (E5 − E1)"),
            ("E3", "E1", r["arm_pairs"]["E3-E1"]["auc"], "U_raw − DistP1 (E3 − E1)")]
    fig, ax = plt.subplots(figsize=(AV.W * 0.82, 5.2))
    scol = {"barts": "#2a78d6", "pantheon": "#1baf7a", "colosseum": "#d6453d"}
    for i, (a, b, st, lab) in enumerate(rows):
        for j, (s, k) in enumerate(KEYS):
            ax.scatter(auc(C[a][(s, k)]) - auc(C[b][(s, k)]), i + (j - 7) * 0.022, s=16, color=scol[s], zorder=2)
        ax.plot(st["ci95"], [i, i], color=INK, lw=2.2, zorder=3)
        ax.scatter([st["mean"]], [i], marker="D", s=50, color=INK, zorder=4)
        ax.text(0.99, i - 0.32, f"{st['mean']:+.3f} [{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] · {st['n_pos']}/15 > 0",
                transform=ax.get_yaxis_transform(), ha="right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([x[3] for x in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUC@100 (per start; same start, same simulator)")
    ax.grid(axis="x", color=GRIDC)
    ax.legend([matplotlib.lines.Line2D([], [], marker="o", ls="", color=c) for c in scol.values()], list(scol), loc="lower left",
              frameon=False, fontsize=8.5, title="dots = starts", title_fontsize=8.5)
    ax.set_title("Paired differences on the 15 Dev-3 starts (diamond + bar = mean and bootstrap 95 % CI from the T1.4 report)",
                 loc="left", fontsize=10.5)
    save(fig, "paired_dots.png")


def fig_cost(r):
    def epmean(x, k):
        return float(np.mean([e[k] for e in x["episodes"]]))
    b0, b2 = r["B0_cost"], r["B2_cost"]
    e1, e6 = r["arms"]["E1"]["cost"], r["arms"]["E6"]["cost"]
    wall = [("B0", b0["wall_s_mean"]), ("B2", b2["wall_s_mean"]), ("E1", e1["wall_s_mean"]), ("E6", e6["wall_s_mean"])]
    mem = [("B0", max(e["peak_mem_gb"] for e in b0["episodes"])), ("B2", max(e["peak_mem_gb"] for e in b2["episodes"])),
           ("E1", e1["peak_mem_gb_max"]), ("E6", e6["peak_mem_gb_max"])]
    auc_ = [("B0", r["baselines_summary"]["B0"]["auc"]), ("B2", r["baselines_summary"]["B2"]["auc"]),
            ("E1", r["arms"]["E1"]["summary"]["auc"]["mean"] if isinstance(r["arms"]["E1"]["summary"]["auc"], dict) else r["arms"]["E1"]["summary"]["auc"]),
            ("E6", r["arms"]["E6"]["summary"]["auc"]["mean"] if isinstance(r["arms"]["E6"]["summary"]["auc"], dict) else r["arms"]["E6"]["summary"]["auc"])]
    fig, axs = plt.subplots(1, 3, figsize=(AV.W, 3.6))
    for ax, data, name, logx, fmt in ((axs[0], wall, "wall time per 100-move episode (s, log scale)", True, "{:.0f} s"),
                                     (axs[1], mem, "planner peak GPU memory (GB)", False, "{:.1f} GB"),
                                     (axs[2], auc_, "AUC@100 (15 starts)", False, "{:.3f}")):
        ax.barh(range(4), [v for _, v in data], color=[STYLE[m][0] for m, _ in data], height=0.62)
        for i, (_, v) in enumerate(data):
            ax.text(v * (1.08 if logx else 1.0) + (0 if logx else 0.01 * max(x for _, x in data)), i, fmt.format(v), va="center", fontsize=9)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(30, 20000)
        else:
            ax.set_xlim(0, max(v for _, v in data) * 1.3)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["B0 (4 worlds)", "B2 (1 world)", "V2K4", "V2K1"] if ax is axs[0] else [""] * 4, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.grid(axis="x", color=GRIDC)
    fig.suptitle("Cost vs quality: V2 needs no 3D world generation (~40x less wall time, <= 6.4 GB) at equal or better AUC",
                 x=0.01, y=1.04, ha="left", fontsize=11, color=INK2)
    save(fig, "cost.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auc", action="store_true")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--anim", default="")
    ap.add_argument("--preview", type=int)
    a = ap.parse_args()
    if a.auc:
        C, r = curves()
        fig_auc_area(C)
        fig_auc_by_scene(C)
        fig_heatmap(C)
        fig_paired(C, r)
        fig_cost(r)
    if a.cover or a.anim:
        check_gtpts()
    if a.cover:
        for s, k in KEYS:
            AV.fig_cover(s, k, ms=("A2", "magician", "E1", "E6"))
    for x in filter(None, a.anim.split(",")):
        s, k = x.split(":")
        AV.anim(s, int(k), ms=("A2", "magician", "E1", "E6"), preview=a.preview)


if __name__ == "__main__":
    main()
