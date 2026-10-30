"""EXP5 Task 2 -- figures of the final comparison (seen2scene env + ffmpeg; descriptive; reads the saved formal runs and the
recomputed metrics only; every observed point shown is a real 10 % sample of a real executed frame, never a prediction).

Indoor (MP3D, 150 moves) -- drawing code of T0.12 (task0/.../t12_viz.py, imported unchanged; paths redirected to Task 2):
  in_curves.png     scene-mean C_obs curves (4 groups) + per level; Comp@5cm vs move
  in_paired.png     per-episode differences V2M - {V2O, MAG-G, MAG-B} (AUC@150, Cov@150, Comp@5cm)
  cover_<ep>.png    5 first starts: GT surface below 2 m from above coloured by the move it is first covered, trajectories
  pcd_<ep>.png      accumulated observed point clouds after 150 moves (top + side view), per group
  anim/anim_task2_<ep>.{mp4,gif}   per-move animation (as T0.12)
Outdoor (Macarons++, 100 moves):
  out_curves.png    scene-mean monotone coverage (solid) and official coverage (dashed) for Test-3 and Dev-3; per scene
  out_traj_<scene>.png  start 0: accumulated observed points (top view) + camera trajectory per group (the last frame's
                    samples are not stored by the simulator: frames 0..99 of 101)
Runtime: runtime.png (decision seconds per move, both datasets).
Output: W2/figures/{indoor,outdoor}/, W2/figures/anim/.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task0", "magician_comparison", "scripts"))
import t25_in_metrics as TIM  # noqa: E402,F401  (redirects mc_common.svc_path / stored_curve to the Task 2 runs)
import t12_viz as V12  # noqa: E402

V, M, S, plt, pe, matplotlib = V12.V, V12.M, V12.S, V12.plt, V12.pe, V12.matplotlib
GROUPS = ("MAG-G", "MAG-B", "V2O", "V2M")
COL = {"MAG-G": "#2a78d6", "MAG-B": "#0b0b0b", "V2O": "#f2a07b", "V2M": "#eb6834"}
LAB = {"MAG-G": "MAGICIAN Greedy", "MAG-B": "MAGICIAN Beam 10x10", "V2O": "V2 Original (Greedy-NV)", "V2M": "V2-Mix (Greedy-NV, ours)"}
FIGI, FIGO = os.path.join(C.W2, "figures", "indoor"), os.path.join(C.W2, "figures", "outdoor")
V12.PANEL, V12.COL, V12.LAB = GROUPS, COL, LAB
V12.BASE = {g: g for g in GROUPS}
V12.FIG = FIGI
V.VIZ, V.ANIM = os.path.join(C.W2, "viz"), os.path.join(C.W2, "figures", "anim")
CFG_OUT = json.load(open(os.path.join(C.T2, "configs", "eval_outdoor_v1.json")))


def have(g):
    return os.path.isdir(os.path.join(C.W2, "eval", "indoor", "closed_loop", g))


def save(fig, d, name):
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name))
    plt.close(fig)
    print("fig", os.path.join(os.path.basename(d), name), flush=True)


def in_rows():
    out = {}
    for s, lv, st in C.EVAL_LEVELS:
        tag = f"{s}_L{lv}_S{st}"
        for g in GROUPS:
            pe_ = os.path.join(C.W2, "eval", "indoor", "closed_loop", g, tag, "episode.json")
            pm = os.path.join(C.W2, "eval", "indoor", "metrics", "episodes", g, f"{tag}.json")
            if os.path.exists(pe_) and os.path.exists(pm):
                e, m = json.load(open(pe_)), json.load(open(pm))
                out[(g, tag)] = {"level": f"{s}_L{lv}", "cov": np.asarray(e["coverage"]), "comp5": np.asarray([r["comp5"] for r in m["curve"]])}
    return out


def scene_mean(rows, g, key, levels):
    per = [np.mean([r[key] for (gg, _), r in rows.items() if gg == g and r["level"] == lv], 0) for lv in levels
           if any(gg == g and r["level"] == lv for (gg, _), r in rows.items())]
    return np.mean(per, 0) if per else None


def fig_in_curves():
    rows = in_rows()
    levels = sorted({r["level"] for r in rows.values()})
    fig = plt.figure(figsize=(V.W, 8.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1])
    a1, a2 = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    for g in GROUPS:
        c = scene_mean(rows, g, "cov", levels)
        if c is None:
            continue
        a1.plot(c, color=COL[g], lw=2.4 if g == "V2M" else 1.8, label=f"{LAB[g]}  AUC@150 {np.trapz(c, dx=1) / 150:.3f} · Cov@150 {c[-1]:.3f}")
        k = scene_mean(rows, g, "comp5", levels)
        a2.plot(k, color=COL[g], lw=2.4 if g == "V2M" else 1.8, label=f"{g}  {100 * k[-1]:.1f} %")
    a1.set_title("MP3D (5 val levels x 2 starts, released 6-DoF grid): scene-mean C_obs", loc="left", fontsize=11)
    a2.set_title("Comp@5cm (metric GT vertices) vs move", loc="left", fontsize=11)
    for a, yl in ((a1, "coverage (C_obs)"), (a2, "Comp@5cm")):
        a.set_xlim(0, 150)
        a.set_ylim(0, None)
        a.set_xlabel("move")
        a.set_ylabel(yl)
        a.grid(color=V.GRID)
        a.legend(frameon=False, fontsize=8.5, loc="lower right")
    for i, lv in enumerate(levels):
        ax = fig.add_subplot(gs[1, i])
        for g in GROUPS:
            c = scene_mean(rows, g, "cov", [lv])
            if c is not None:
                ax.plot(c, color=COL[g], lw=1.6)
        ax.set_title(lv, loc="left", fontsize=9.5)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1.0)
        ax.grid(color=V.GRID)
        ax.set_xlabel("move")
    fig.tight_layout()
    save(fig, FIGI, "in_curves.png")


def fig_in_paired():
    r = json.load(open(os.path.join(C.W2, "analysis", "t26.json")))
    rows = {}
    import csv
    for x in csv.DictReader(open(os.path.join(C.T2, "tables", "episodes_indoor.csv"))):
        rows[(x["group"], x["episode"])] = x
    eps = sorted({e for _, e in rows})
    pairs = [("V2M", "V2O"), ("V2M", "MAG-G"), ("V2M", "MAG-B"), ("MAG-B", "MAG-G")]
    fig, axs = plt.subplots(1, 3, figsize=(V.W, 3.8), sharey=True)
    for ax, (k, name) in zip(axs, (("auc150", "ΔAUC@150"), ("cov150", "ΔCov@150"), ("comp5_150", "ΔComp@5cm"))):
        for i, (a, b) in enumerate(pairs):
            d = [float(rows[(a, e)][k]) - float(rows[(b, e)][k]) for e in eps if (a, e) in rows and (b, e) in rows]
            ax.scatter(d, np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)) if d else [], s=14, color=V.MUTED, zorder=2)
            s = r["indoor_paired"].get(f"{a}-{b}", {}).get(k)
            if s:
                ax.plot(s["ci95"], [i, i], color=V.INK, lw=2.2, zorder=3)
                ax.scatter([s["mean"]], [i], s=46, marker="D", color=COL[a], edgecolor=V.INK, zorder=4)
                ax.text(0.99, i - 0.32, f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] · {s['n_pos']}/{s['n_episodes']} > 0",
                        transform=ax.get_yaxis_transform(), fontsize=7.5, color=V.INK2, ha="right")
        ax.axvline(0, color=V.INK2, lw=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([f"{a} − {b}" for a, b in pairs], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(axis="x", color=V.GRID)
    fig.suptitle("MP3D paired differences (dots = episodes; diamond + bar = scene-cluster bootstrap 95 % CI, 5 levels)", x=0.01, y=1.03,
                 ha="left", fontsize=10.5, color=V.INK2)
    save(fig, FIGI, "in_paired.png")


def fig_in_pcd(tag):
    """Accumulated observed points (real 10 % samples) after 150 moves; top and side view, coloured by height."""
    gs_ = [g for g in GROUPS if have(g)]
    fig, axs = plt.subplots(2, len(gs_), figsize=(V.W, 6.2), gridspec_kw={"height_ratios": [2.2, 1]})
    for c, g in enumerate(gs_):
        P = np.concatenate([np.load(M.svc_path(g, tag, t))["sample_points"] for t in range(151)])
        P = S.rt_to_zup_m(P)
        idx = np.random.default_rng(0).permutation(len(P))[:120000]
        P = P[idx]
        st = M.stored_curve(g, tag)
        for r, (i, j) in enumerate(((0, 1), (0, 2))):
            ax = axs[r, c]
            o = np.argsort(P[:, 2]) if r == 0 else np.argsort(-P[:, 1])
            ax.scatter(P[o, i], P[o, j], c=P[o, 2], s=0.15, cmap="viridis", lw=0, rasterized=True)
            ax.set_aspect("equal")
            V.clean(ax)
        axs[0, c].set_title(f"{LAB[g]}\n{len(idx)} of the observed points · Cov@150 {st['coverage'][150]:.3f}", loc="left", fontsize=9)
    axs[0, 0].text(-0.04, 0.5, "top view", transform=axs[0, 0].transAxes, rotation=90, ha="right", va="center", color=V.INK2)
    axs[1, 0].text(-0.04, 0.5, "side view", transform=axs[1, 0].transAxes, rotation=90, ha="right", va="center", color=V.INK2)
    fig.suptitle(f"{tag}: real observed point cloud after 150 moves (accumulated 10 % depth samples; colour = height)", x=0.01, ha="left",
                 fontsize=11, color=V.INK2)
    fig.tight_layout()
    save(fig, FIGI, f"pcd_{tag}.png")


def out_rows():
    rows = {}
    for split in ("test", "dev"):
        for s in CFG_OUT["scenes"][split]:
            for k in CFG_OUT["starts"]:
                for g in GROUPS:
                    p = os.path.join(C.W2, "eval", "outdoor", "metrics", g, f"{g}_{s}_s{k}_r0.json")
                    if os.path.exists(p):
                        m = json.load(open(p))
                        rows[(g, s, k)] = {"split": split, "mono": np.asarray(m["cov_mono"]), "off": np.asarray(m["cov_official"])}
    return rows


def fig_out_curves():
    rows = out_rows()
    fig = plt.figure(figsize=(V.W, 8.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.3, 1])
    for si, split in enumerate(("test", "dev")):
        ax = fig.add_subplot(gs[0, 3 * si:3 * si + 3])
        for g in GROUPS:
            sc = [np.mean([r["mono"] for (gg, s, k), r in rows.items() if gg == g and s == scn], 0) for scn in CFG_OUT["scenes"][split]
                  if any(gg == g and s == scn for (gg, s, k) in rows)]
            so = [np.mean([r["off"] for (gg, s, k), r in rows.items() if gg == g and s == scn], 0) for scn in CFG_OUT["scenes"][split]
                  if any(gg == g and s == scn for (gg, s, k) in rows)]
            if not sc:
                continue
            c, co = np.mean(sc, 0), np.mean(so, 0)
            ax.plot(c, color=COL[g], lw=2.4 if g == "V2M" else 1.8, label=f"{LAB[g]}  AUC@100 {np.trapz(c, dx=1) / 100:.3f} · Cov@100 {c[-1]:.3f}")
            ax.plot(co, color=COL[g], lw=1, ls="--", alpha=0.6)
        ax.set_title(f"Macarons++ {'Test-3 (unseen)' if split == 'test' else 'Dev-3'}: scene mean (solid = monotone, dashed = official)",
                     loc="left", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel("move")
        ax.grid(color=V.GRID)
        ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    for i, scn in enumerate(CFG_OUT["scenes"]["test"] + CFG_OUT["scenes"]["dev"]):
        ax = fig.add_subplot(gs[1, i])
        for g in GROUPS:
            r = [x["mono"] for (gg, s, k), x in rows.items() if gg == g and s == scn]
            if r:
                ax.plot(np.mean(r, 0), color=COL[g], lw=1.5)
        ax.set_title(scn, loc="left", fontsize=9.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(color=V.GRID)
        ax.set_xlabel("move")
    fig.tight_layout()
    save(fig, FIGO, "out_curves.png")


def fig_out_traj(scene, start=0):
    import glob
    gs_ = [g for g in GROUPS if os.path.exists(os.path.join(C.W2, "eval", "outdoor", "closed_loop", g, "episodes", f"ep_{g}_{scene}_s{start}_r0.json"))]
    if not gs_:
        return
    fig, axs = plt.subplots(1, len(gs_), figsize=(V.W, 4.3))
    axs = np.atleast_1d(axs)
    for ax, g in zip(axs, gs_):
        tag = f"{g}_{scene}_s{start}_r0"
        d = os.path.join(C.W2, "eval", "outdoor", "closed_loop", g, "svc", tag)
        fs = sorted(glob.glob(os.path.join(d, "svc_*.npz")), key=lambda f: int(f.split("_t")[-1][:-4]))
        P = np.concatenate([np.load(f)["sample_points"] for f in fs])
        cam = np.asarray([np.load(f)["cam_center"] for f in fs])
        Pz, cz = P[:, [0, 2, 1]] * np.asarray([1, -1, 1]), cam[:, [0, 2, 1]] * np.asarray([1, -1, 1])   # y-up runtime -> z-up
        idx = np.random.default_rng(0).permutation(len(Pz))[:150000]
        o = idx[np.argsort(Pz[idx, 2])]
        ax.scatter(Pz[o, 0], Pz[o, 1], c=Pz[o, 2], s=0.12, cmap="viridis", lw=0, rasterized=True)
        ax.plot(cz[:, 0], cz[:, 1], color=COL[g], lw=1.4, path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
        ax.scatter(cz[0, 0], cz[0, 1], s=50, facecolor="white", edgecolor=V.INK, lw=1.6, zorder=5)
        m = json.load(open(os.path.join(C.W2, "eval", "outdoor", "metrics", g, f"{tag}.json")))
        ax.set_title(f"{LAB[g]}\nCov@100 {m['cov_mono'][100]:.3f} (monotone)", loc="left", fontsize=9)
        ax.set_aspect("equal")
        V.clean(ax)
    fig.suptitle(f"Macarons++ {scene} start {start}: real observed points (top view, colour = height) and camera path; white = start",
                 x=0.01, y=1.04, ha="left", fontsize=10.5, color=V.INK2)
    save(fig, FIGO, f"out_traj_{scene}_s{start}.png")


def fig_runtime():
    import csv
    ri = list(csv.DictReader(open(os.path.join(C.T2, "tables", "episodes_indoor.csv"))))
    ro = list(csv.DictReader(open(os.path.join(C.T2, "tables", "episodes_outdoor.csv"))))
    fig, axs = plt.subplots(1, 2, figsize=(V.W * 0.8, 3.4))
    for ax, rows, name in ((axs[0], ri, "MP3D (decision s / move)"), (axs[1], ro, "Macarons++ (decision s / move)")):
        vals = [[float(x["decision_s_mean"]) for x in rows if x["group"] == g and x["decision_s_mean"]] for g in GROUPS]
        ax.barh(range(len(GROUPS)), [np.mean(v) if v else 0 for v in vals], color=[COL[g] for g in GROUPS], height=0.62)
        for i, v in enumerate(vals):
            if v:
                ax.text(np.mean(v), i, f" {np.mean(v):.2f} s", va="center", fontsize=9)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels([LAB[g] for g in GROUPS] if ax is axs[0] else [""] * len(GROUPS), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=10.5)
        ax.grid(axis="x", color=V.GRID)
    fig.suptitle("Planner decision time per move in the closed loop (not exclusive timing: several jobs per node)", x=0.01, y=1.05,
                 ha="left", fontsize=10, color=V.INK2)
    save(fig, os.path.join(C.W2, "figures"), "runtime.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indoor", action="store_true")
    ap.add_argument("--outdoor", action="store_true")
    ap.add_argument("--cover", nargs="*")
    ap.add_argument("--pcd", nargs="*")
    ap.add_argument("--anim", nargs="*")
    a = ap.parse_args()
    if a.indoor:
        fig_in_curves()
        fig_in_paired()
    if a.outdoor:
        fig_out_curves()
        for scn in CFG_OUT["scenes"]["test"] + CFG_OUT["scenes"]["dev"]:
            fig_out_traj(scn, 0)
    if a.indoor and a.outdoor:
        fig_runtime()
    for tag in a.cover or []:
        V12.fig_cover(tag)
    for tag in a.pcd or []:
        fig_in_pcd(tag)
    for tag in a.anim or []:
        V12.anim(tag)


if __name__ == "__main__":
    main()
