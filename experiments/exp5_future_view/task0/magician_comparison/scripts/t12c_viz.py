"""EXP5 Task 0 / T0.12-C -- explanatory visualisation of the "V2 does not go far" diagnosis (seen2scene env + ffmpeg;
descriptive; reads finished outputs only).
  --explain   figures/grid/diag_c_explain_<episode>.png: V2K4 (top row) vs MAGICIAN Beam (middle row) at 4 moves, top view of
              the GT surface below 2 m: purple = covered, orange = not yet covered and INSIDE V2's planning region, red =
              not yet covered and outside it, pale = not observable; V2's planning region (the box it predicts and scores
              in, re-centred on the camera every 10 moves) vs MAGICIAN's whole-level scoring box; bottom: coverage curves
              and V2's best predicted gain per move (frozen region vs the whole-level probe)
  --anim      anim/anim_region_<episode>.{mp4,gif}: the same, every move
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t12_viz as V12  # noqa: E402

V, M, S, plt, pe, matplotlib = V12.V, V12.M, V12.S, V12.plt, V12.pe, V12.matplotlib
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROWS = ("G-V2K4", "G-MAG-B")
COVERED, INREG, OUTREG = "#6d4fb3", "#f0a32f", "#d6453d"


class Data:
    def __init__(self, tag):
        self.tag = tag
        _, scan, lv, _ = V12.ep_info(tag)
        self.R = V.raster(scan, lv)
        self.crd = self.R["crd"]
        cov, _ = V.cov_obj(scan, lv)
        self.P = S.rt_to_zup_m(cov.pts)[:, :2]
        self.box = np.asarray(self.crd["box_m"])
        self.d = {}
        for g in ROWS:
            c = V12.covrep(g, tag)
            pos, fwd, st = V12.cam_track(g, tag, self.crd)
            logs = [json.loads(x) for x in open(os.path.join(M.MC, "grid_protocol", "closed_loop", g, tag, "planner.jsonl"))]
            self.d[g] = {**c, "pos": pos, "fwd": fwd, "cov": np.asarray(st["coverage"])}
        lg = [json.loads(x) for x in open(os.path.join(M.MC, "grid_protocol", "closed_loop", "G-V2K4", tag, "planner.jsonl"))]
        self.regions = [np.asarray(lg[10 * (t // 10)]["refresh"]["box_m"])[:, :2] for t in range(150)] + \
                       [np.asarray(lg[140]["refresh"]["box_m"])[:, :2]]
        self.best = np.asarray([max(x["scores"]) for x in lg])
        pr = json.load(open(os.path.join(M.MC, "diag_c", "v2_level", "G-V2K4", f"{tag}.json")))["rows"]
        self.best_level = np.asarray([max(r["scores"]) for r in pr])

    def rgba(self, g, t, region):
        d = self.d[g]
        col = np.tile(matplotlib.colors.to_rgba(V.HIDDEN), (len(d["first"]), 1))
        vis = d["visible"].astype(bool)
        rem = vis & ~((d["first"] >= 0) & (d["first"] <= t))
        if region is not None:
            ins = np.all((self.P >= region[0]) & (self.P <= region[1]), axis=1)
            col[rem & ins] = matplotlib.colors.to_rgba(INREG)
            col[rem & ~ins] = matplotlib.colors.to_rgba(OUTREG)
            share = float((rem & ins).sum() / max(1, rem.sum()))
        else:
            col[rem] = matplotlib.colors.to_rgba(OUTREG)
            share = None
        col[vis & ~rem] = matplotlib.colors.to_rgba(COVERED)
        return col, share


def draw_panel(ax, D, g, t, first=False):
    region = D.regions[t] if g == "G-V2K4" else None
    col, share = D.rgba(g, t, region)
    im = ax.imshow(V.raster_img(D.R, col), origin="lower", extent=D.R["extent"], interpolation="nearest")
    d = D.d[g]
    ln, = ax.plot(d["pos"][:t + 1, 0], d["pos"][:t + 1, 1], color=V12.COL[g], lw=1.5,
                  path_effects=[pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    L = 0.07 * float(D.box[1, 0] - D.box[0, 0])
    q = ax.quiver([d["pos"][t, 0]], [d["pos"][t, 1]], [d["fwd"][t, 0] * L], [d["fwd"][t, 1] * L], angles="xy", scale_units="xy",
                  scale=1, color=V.INK, width=0.014, headwidth=3, headlength=3.5, zorder=6)
    if region is not None:
        rect = Rectangle(region[0], *(region[1] - region[0]), fill=False, ec=V12.COL[g], lw=2.6, zorder=5)
    else:
        rect = Rectangle(D.box[0, :2], *(D.box[1, :2] - D.box[0, :2]), fill=False, ec=V.INK, lw=1.6, ls="--", zorder=5)
    ax.add_patch(rect)
    ax.set_xlim(D.box[0, 0] - 0.3, D.box[1, 0] + 0.3)
    ax.set_ylim(D.box[0, 1] - 0.3, D.box[1, 1] + 0.3)
    ax.set_aspect("equal")
    V.clean(ax)
    return {"im": im, "ln": ln, "q": q, "rect": rect, "share": share}


def title(g, t, D, share):
    c = D.d[g]["cov"][t]
    if g == "G-V2K4":
        return f"V2K4 · move {t} · coverage {c:.3f}\nleft to see inside V2's region: {100 * share:.0f}%"
    return f"MAG-B · move {t} · coverage {c:.3f}\nscores the whole level (dashed)"


def legend(fig, y):
    fig.legend([Patch(color=COVERED), Patch(color=INREG), Patch(color=OUTREG), Patch(color=V.HIDDEN),
                Patch(fill=False, ec=V12.COL["G-V2K4"], lw=2.4), Patch(fill=False, ec=V.INK, lw=1.4, ls="--")],
               ["covered", "not yet covered, inside V2's region", "not yet covered, outside V2's region", "not observable",
                "V2's planning region (re-centred every 10 moves)", "MAGICIAN's scoring range (whole level)"],
               loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, y))


def curves(ax1, ax2, D, t=None):
    for g in ROWS:
        ax1.plot(D.d[g]["cov"], color=V12.COL[g], lw=2.2, label=V12.LAB[g])
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("move")
    ax1.set_ylabel("coverage (C_obs)")
    ax1.grid(color=V.GRID)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.plot(D.best, color=V12.COL["G-V2K4"], lw=1.8, label="V2, frozen planning region (what it actually used)")
    ax2.plot(D.best_level, color=V12.COL["G-V2K4"], lw=1.2, ls="--", label="same V2, region = whole level (probe)")
    ax2.set_yscale("log")
    ax2.set_xlim(0, 150)
    ax2.set_xlabel("move")
    ax2.set_ylabel("best predicted gain (DistP1)")
    ax2.grid(color=V.GRID)
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax2.set_title("V2's best candidate score at the same states (V2K4 trajectory)", loc="left", fontsize=10)
    ax1.set_title("coverage", loc="left", fontsize=10)
    return [a.axvline(t if t is not None else 0, color=V.INK2, lw=0.9, ls=":") for a in (ax1, ax2)]


def explain(tag, moves=(20, 50, 80, 120)):
    D = Data(tag)
    asp = (D.box[1, 1] - D.box[0, 1]) / (D.box[1, 0] - D.box[0, 0])
    fig = plt.figure(figsize=(V.W, 2 * (V.W / 4) * asp + 3.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[asp, asp, 0.95], hspace=0.28, wspace=0.3)
    for r, g in enumerate(ROWS):
        for c, t in enumerate(moves):
            ax = fig.add_subplot(gs[r, c])
            p = draw_panel(ax, D, g, t)
            ax.set_title(title(g, t, D, p["share"]), loc="left", fontsize=9)
    vl = curves(fig.add_subplot(gs[2, :2]), fig.add_subplot(gs[2, 2:]), D)
    for x in vl:
        x.set_visible(False)
    for a in fig.axes[-2:]:
        for t in moves:
            a.axvline(t, color=V.MUTED, lw=0.7, ls=":")
    legend(fig, -0.035)
    fig.suptitle(f"{tag}: V2 predicts and scores only inside its planning region (solid box, ~7.8 m); MAGICIAN scores the whole level",
                 x=0.01, y=0.995, ha="left", fontsize=11, color=V.INK2)
    V12.save(fig, f"diag_c_explain_{tag}.png")


def anim(tag, fps=5):
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    D = Data(tag)
    asp = (D.box[1, 1] - D.box[0, 1]) / (D.box[1, 0] - D.box[0, 0])
    fig = plt.figure(figsize=(12, 6 * asp + 3.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[6 * asp, 3.0])
    axs = [fig.add_subplot(gs[0, c]) for c in range(2)]
    a1, a2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    vl = curves(a1, a2, D, 0)
    legend(fig, 0.0)
    sup = fig.suptitle("", x=0.01, ha="left", fontsize=11, color=V.INK2)
    state = {}

    def update(t):
        for ax, g in zip(axs, ROWS):
            ax.clear()
            p = draw_panel(ax, D, g, t)
            ax.set_title(title(g, t, D, p["share"]), loc="left", fontsize=10)
        for x in vl:
            x.set_xdata([t, t])
        sup.set_text(f"{tag} · move {t:3d} / 150 · V2's region moves only at moves 0, 10, 20, ... (to where the camera is then)")
        return []
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    ani = FuncAnimation(fig, update, frames=range(151), blit=False)
    os.makedirs(V.ANIM, exist_ok=True)
    base = os.path.join(V.ANIM, f"anim_region_{tag}")
    ani.save(base + ".mp4", writer=FFMpegWriter(fps=fps, bitrate=3000), dpi=90)
    ani.save(base + ".gif", writer=PillowWriter(fps=fps), dpi=65)
    plt.close(fig)
    print("anim", tag, state, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", nargs="*")
    ap.add_argument("--anim", nargs="*")
    a = ap.parse_args()
    for tag in a.explain or []:
        explain(tag)
    for tag in a.anim or []:
        anim(tag)


if __name__ == "__main__":
    main()
