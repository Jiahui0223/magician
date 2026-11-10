"""CVPR Fig. 4 -- selection of the example states by a pre-declared rule + the multi-hypothesis depth figure (CPU). python fig4_plot.py
Inputs: work_dirs/cvpr_results/overnight_20261110/intermediates/fig4/{<scene>.jsonl, <scene>_s<k>.npz, backface_<scene>.jsonl}
(fig4_scan.py / fig4_backface.py). Every number / pixel comes from those files; the real depth is the observation of the NEXT step and
is shown for reference only (it is never an input of the model or the planner).

Selection rule (fixed in this file before the pool was scanned; all candidates and the reasons are written to
visualization_manifest.csv so the choice can be audited):
  eligible   exactness guards true (4x grid hash and per-sample values equal the planner log; the next record is the executed pick),
             UNKNOWN rays >= 20 % of the 7,296 rays of the view, real in-range surface on >= 15 % of the UNKNOWN rays (something to
             compare with), back faces < 5 % of the executed view (camera not inside the mesh)
             AMENDMENT 2026-11-10 05:40 (after the 14-scene start-pose audit manifest/start_pose_backface.csv showed that some
             official starts, e.g. bannerman start 1, see 34 % back faces at t = 0): back faces must also be < 5 % in the current
             view and in every already observed view at steps 0, 10, ..., t (the views P_t was built from). The amendment only
             removes physically invalid states; the first selection (before the amendment) is kept in selection_v0.json.
  diversity  share of UNKNOWN rays whose hit / no-hit outcome differs between the 4 samples ("flip share")
  agreement  mean over the 4 samples of the share of UNKNOWN rays whose hit / no-hit equals the real in-range depth mask
  success    per split: among eligible states with flip share >= the median flip share of all eligible states, the highest agreement
             (ties: lower median absolute depth error)
  hard       per split: among eligible states, the lowest agreement
Panels: k = 1..4 are SAMPLE indices of one model (K = 4 noise maps), not four model configurations. Each sample is a decoded
first-hit depth along the rays of one candidate view; the method does not generate a 3D world, and samples with the same index at
two different views are only Monte-Carlo paired (same noise map), not guaranteed to describe one consistent scene."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
from matplotlib.colors import ListedColormap  # noqa: E402

D = os.path.join(L.OV, "intermediates", "fig4")
OUT = os.path.join(L.FIG, "fig4")
POOL = {"train": ["bannerman", "redeemer"], "dev": ["colosseum", "pantheon"], "test": ["alhambra", "neuschwanstein"]}
RANGE, GH, GW = 70.0, 64, 114
CLS_UNKNOWN = 0


def load_pool():
    rows = []
    for sp, scenes in POOL.items():
        for s in scenes:
            bf = {(r["start"], r["t"]): r for r in map(json.loads, open(os.path.join(D, f"backface_{s}.jsonl")))} if os.path.exists(os.path.join(D, f"backface_{s}.jsonl")) else {}
            p = os.path.join(D, f"{s}.jsonl")
            if not os.path.exists(p):
                continue
            for r in map(json.loads, open(p)):
                b = bf.get((r["start"], r["t"]))
                r.update(split=sp, back_share_next=None if b is None else b["back_share_next"], sky_share_next=None if b is None else b["sky_share_next"],
                         back_share_cur=None if b is None else b.get("back_share_cur"), back_share_hist_max=None if b is None else b.get("back_share_hist_max"))
                why = []
                if not (r["grid_hash_equal_log"] and r["values_equal_log"] and r["next_is_pick"]):
                    why.append("exactness guard failed")
                if r["n_unknown"] < 0.2 * GH * GW:
                    why.append("UNKNOWN rays < 20 %")
                if (r["real_valid_share_on_unknown"] or 0.0) < 0.15:
                    why.append("real surface on < 15 % of the UNKNOWN rays")
                if b is None:
                    why.append("no back-face check")
                elif b["back_share_next"] >= 0.05:
                    why.append("back faces >= 5 % (camera inside the mesh)")
                elif b.get("back_share_hist_max") is None:
                    why.append("no back-face check of the observed views")
                elif max(b["back_share_cur"], b["back_share_hist_max"]) >= 0.05:
                    why.append("back faces >= 5 % in an already observed view (map built from inside the mesh)")
                r["eligible"], r["ineligible_reason"] = not why, "; ".join(why)
                rows.append(r)
    return rows


def select(rows):
    el = [r for r in rows if r["eligible"]]
    med = float(np.median([r["flip_share_on_unknown"] for r in el])) if el else 0.0
    sel = {}
    for sp in POOL:
        c = [r for r in el if r["split"] == sp]
        if not c:
            continue
        div = [r for r in c if r["flip_share_on_unknown"] >= med]
        if div:
            best = sorted(div, key=lambda r: (-r["hit_agreement_on_unknown"], r["median_abs_depth_err_on_unknown"] if r["median_abs_depth_err_on_unknown"] is not None else 1e9))[0]
            sel[(sp, "success")] = best
        rest = [r for r in c if r is not sel.get((sp, "success"))]
        if rest:
            sel[(sp, "hard")] = sorted(rest, key=lambda r: r["hit_agreement_on_unknown"])[0]
    return sel, med


def state(r):
    z = np.load(os.path.join(D, f"{r['scene']}_s{r['start']}.npz"))
    g = lambda n: z[f"t{r['t']}_{n}"]  # noqa: E731
    kM, dz = float(g("k_M")), g("dz")
    dep = g("t_dec") * kM * dz[None]                                   # [4, R] z-depth (runtime units), inf = no hit
    return {"dep": dep.reshape(4, GH, GW), "cls": g("cls").reshape(GH, GW), "known": (g("t_known") * kM * dz).reshape(GH, GW), "novel": g("novel").reshape(4, GH, GW),
            "real": g("real_depth").reshape(GH, GW), "real_valid": g("real_valid").reshape(GH, GW), "value_k": g("value_k")}


CMAP = plt.get_cmap("viridis_r").copy()
CMAP.set_bad("#FFFFFF")


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#666666")


def depth_tile(ax, d, valid, title, unk=None, sub=None):
    im = ax.imshow(np.ma.masked_where(~valid, np.clip(d, 0, RANGE)), cmap=CMAP, vmin=0, vmax=RANGE, interpolation="nearest", aspect="equal")
    if unk is not None and unk.any() and not unk.all():
        ax.contour(unk.astype(float), levels=[0.5], colors="#D55E00", linewidths=0.55)
    frame(ax)
    ax.set_title(title, fontsize=7.0, pad=2.5)
    if sub:
        ax.set_xlabel(sub, fontsize=6.2, labelpad=2.5, linespacing=1.15)
    return im


def block(fig, gs, r, role):
    """gs: 2 x 4 grid of one example. Row 1: what the planner knows / the reference; row 2: the K = 4 samples of the same candidate view."""
    S = state(r)
    unk = S["cls"] == CLS_UNKNOWN
    fin = np.isfinite(S["dep"])
    name = f"{L.NAME[r['scene']]} ({RD.SPLIT_TAG[r['split']]} scene), start {r['start']}, step {r['t']}: {role} case"
    ax = fig.add_subplot(gs[0, 0])
    known = (~unk) & np.isfinite(S["known"]) & (S["known"] > 0)
    im = depth_tile(ax, S["known"], known, "Observed surface in $P_t$", unk)
    ax.text(0.0, 1.24, name, transform=ax.transAxes, fontsize=8.0, fontweight="bold", ha="left", va="bottom")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.where(unk, 2, np.where(known, 1, 0)), cmap=ListedColormap(["#FFFFFF", "#BFBFBF", "#F2C6A5"]), vmin=0, vmax=2, interpolation="nearest")
    frame(ax)
    ax.set_title(f"Candidate rays: {100 * unk.mean():.0f}% unknown", fontsize=7.0, pad=2.5)
    ax = fig.add_subplot(gs[0, 2])
    depth_tile(ax, S["real"], S["real_valid"], "Real depth after the move", unk)
    ax = fig.add_subplot(gs[0, 3])
    cnt = np.where(unk, fin.sum(0), -1)
    ax.imshow(cnt, cmap=ListedColormap(["#D0D0D0", "#FFFFFF", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"]), vmin=-1, vmax=4, interpolation="nearest")
    frame(ax)
    ax.set_title(f"Samples with a hit; differ on {100 * r['flip_share_on_unknown']:.0f}%", fontsize=7.0, pad=2.5)
    for q in range(4):
        ax = fig.add_subplot(gs[1, q])
        agree = float((fin[q][unk] == S["real_valid"][unk]).mean())
        both = unk & fin[q] & S["real_valid"]
        err = f"{float(np.median(np.abs(S['dep'][q][both] - S['real'][both]))):.1f}" if both.any() else "n/a"
        depth_tile(ax, np.where(fin[q], S["dep"][q], 0.0), fin[q], f"Sample $k={q + 1}$", unk,
                   f"new-coverage value {S['value_k'][q]:.0f}\nhit agreement {agree:.2f}, median $|\\Delta z|$ {err}")
    return im


def figure(selected, path):
    n = len(selected)
    bh = 2.55                                                     # height of one example block (inches)
    H = bh * n + 0.75
    fig = plt.figure(figsize=(7.1, H))
    outer = fig.add_gridspec(n, 1, left=0.008, right=0.992, top=1 - 0.26 / H, bottom=0.86 / H, hspace=0.36)
    im = None
    for i, (role, r) in enumerate(selected):
        im = block(fig, outer[i].subgridspec(2, 4, hspace=0.30, wspace=0.05), r, role)
    cax = fig.add_axes([0.25, 0.30 / H, 0.50, 0.07 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("depth along the candidate view (benchmark units); white = no surface within the 70-unit sensor range", fontsize=6.4, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.0, length=2)
    RD.save(fig, path)


def main():
    os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)
    rows = load_pool()
    sel, med = select(rows)
    role_of = {(r["scene"], r["start"], r["t"]): f"{sp} / {ro}" for (sp, ro), r in sel.items()}
    hdr = ["scene", "split", "start", "t", "eligible", "ineligible_reason", "selected_as", "n_unknown", "unknown_share", "real_valid_share_on_unknown", "flip_share_on_unknown", "hit_agreement_on_unknown",
           "median_abs_depth_err_on_unknown", "pred_hit_share_on_unknown", "back_share_next", "back_share_cur", "back_share_hist_max", "sky_share_next", "grid_hash_equal_log", "values_equal_log", "next_is_pick", "pick", "n_legal", "d1_score", "value_k"]
    L.write_csv(os.path.join(OUT, "visualization_manifest.csv"), hdr + ["selection_rule"],
                [[r["scene"], r["split"], r["start"], r["t"], r["eligible"], r["ineligible_reason"], role_of.get((r["scene"], r["start"], r["t"]), "")] + [r.get(k) for k in hdr[7:]]
                 + [f"success: eligible, flip >= pool median {med:.4f}, max hit agreement; hard: eligible, min hit agreement (see fig4_plot.py docstring)"] for r in rows])
    print(f"pool {len(rows)} states, eligible {sum(r['eligible'] for r in rows)}, median flip share of the eligible states {med:.4f}")
    for (sp, ro), r in sel.items():
        print(f"  {sp:5s} {ro:7s} {r['scene']} s{r['start']} t{r['t']}: unknown {r['unknown_share']:.2f} flip {r['flip_share_on_unknown']:.3f} agreement {r['hit_agreement_on_unknown']:.3f} "
              f"med|dz| {r['median_abs_depth_err_on_unknown']} back {r['back_share_next']:.3f}")
        figure([(ro, r)], os.path.join(OUT, "panels", f"{sp}_{ro}_{r['scene']}_s{r['start']}_t{r['t']}"))
    if not sel:
        return
    succ = [(ro, r) for (sp, ro), r in sel.items() if ro == "success"]
    hard = [(ro, r) for (sp, ro), r in sel.items() if ro == "hard"]
    main_sel = []
    if ("test", "success") in sel:
        main_sel.append(("success", sel[("test", "success")]))
    elif succ:
        main_sel.append(succ[0])
    if hard:
        main_sel.append(sorted(hard, key=lambda x: x[1]["hit_agreement_on_unknown"])[0])
    figure(main_sel, os.path.join(OUT, "hypothesis_depth"))
    figure(succ + hard, os.path.join(OUT, "hypothesis_depth_all"))
    json.dump({"pool": POOL, "n_states": len(rows), "n_eligible": sum(r["eligible"] for r in rows), "median_flip_eligible": med,
               "selected": {f"{sp}/{ro}": {k: r[k] for k in ("scene", "start", "t", "unknown_share", "flip_share_on_unknown", "hit_agreement_on_unknown", "median_abs_depth_err_on_unknown", "back_share_next")} for (sp, ro), r in sel.items()},
               "main_figure": [f"{ro}: {r['scene']} s{r['start']} t{r['t']}" for ro, r in main_sel]}, open(os.path.join(OUT, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
