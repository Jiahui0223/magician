"""EXP7 -- MP3D-150 tables, figures and reports from the validated results (CPU). Run after collect_mp3d150.py.
Reads experiments/cvpr_results/exp7/results/{mp3d150_summary,mp3d150_paired,mp3d150_levels,mp3d150_episodes}.csv and the canonical
records; writes exp7/fig/* and exp7/report/{MP3D_150_RESULTS.md, VALIDATION.md}. A method / split enters a table only with
status COMPLETE (every level of the split VALID for every start); otherwise the cell is '–' and the report says PARTIAL.
Main table = ALL-22 (22 levels x 2 starts); HO-12 / VALSEL-10 / T2-5 in the report. The two-panel table (Outdoor = CVPR Table 1 v2,
Indoor = this table) uses the 6 Table 1 columns; Indoor Final = C[150]."""
import csv
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E7 = os.path.join(R, "experiments", "cvpr_results", "exp7")
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import render as RD  # noqa: E402

plt = RD.plt
RES, FIG, REP = os.path.join(E7, "results"), os.path.join(E7, "fig"), os.path.join(E7, "report")
MAIN = ["RANDOM", "MAG-B", "OURS"]
NAME = {"RANDOM": "Random Walk", "MAG-B": "MAGICIAN", "OURS": "Ours", "OURS-H0": "Ours w/o lookahead (H0)"}
STY = {"OURS": RD.STYLE["V3-T0-D1"], "MAG-B": RD.STYLE["MAG-B"], "RANDOM": RD.STYLE["RANDOM"],
       "OURS-H0": dict(color="#E69F00", ls="-", marker="D", lw=1.3, label="Ours w/o lookahead (H0)")}
COLS7 = [("AUC", "@20", "AUC@20"), ("AUC", "@50", "AUC@50"), ("AUC", "@100", "AUC@100"), ("AUC", "@150", "AUC@150"),
         ("Coverage", "@20", "Cov@20"), ("Coverage", "@50", "Cov@50"), ("Coverage", "Final", "Final")]
COLS6 = [c for c in COLS7 if c[2] != "AUC@150"]
SPLIT_TXT = {"ALL-22": "all 22 levels of the 11 MP3D val houses", "HO-12": "12 levels of the 5 held-out houses",
             "VALSEL-10": "10 levels of the 6 validation-monitoring houses", "T2-5": "the 5 Task 2 levels"}


def rd(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def summary():
    return {(r["split"], r["method"]): r for r in rd("mp3d150_summary.csv")}


def ranks(vals, cols):
    out = {}
    for _, _, k in cols:
        v = sorted({round(x[k], 3) for x in vals.values()}, reverse=True)
        out[k] = ({m for m, x in vals.items() if round(x[k], 3) == v[0]}, {m for m, x in vals.items() if len(v) > 1 and round(x[k], 3) == v[1]})
    return out


def draw(panels, cols, notes, width, fs=8.0, name_w=1.0):
    """panels: [(panel title or None, [(method, values or None)])]; bold best / underline second per panel (rounded)."""
    n_rows = sum(len(p[1]) for p in panels) + sum(1 for p in panels if p[0])
    cw = [name_w] + [(width - name_w - 0.05) / len(cols)] * len(cols)
    rh = 0.165
    H = rh * (n_rows + 2.6 + 0.3 * len(panels)) + 0.14 * len(notes) + 0.1
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width); ax.set_ylim(0, H); ax.axis("off")
    x0 = [0.03 + sum(cw[:j]) for j in range(len(cw))]
    xc = [x0[j] + cw[j] / 2 for j in range(len(cw))]
    xr = x0[-1] + cw[-1]
    y = H - 0.06
    ax.plot([0.03, xr], [y, y], color="k", lw=0.9)
    y1 = y - rh * 0.62
    ax.text(x0[0] + 0.03, y1 - rh * 0.5, "Method", fontsize=fs, va="center", ha="left")
    groups = {}
    for j, (g, _, _) in enumerate(cols, 1):
        groups.setdefault(g, []).append(j)
    for g, js in groups.items():
        a, b = js[0], js[-1]
        ax.text((x0[a] + x0[b] + cw[b]) / 2, y1, g + " $\\uparrow$", fontsize=fs, ha="center", va="center")
        ax.plot([x0[a] + 0.04, x0[b] + cw[b] - 0.04], [y1 - rh * 0.42] * 2, color="k", lw=0.45)
    y2 = y1 - rh * 0.95
    for j, (_, sub, _) in enumerate(cols, 1):
        ax.text(xc[j], y2, sub, fontsize=fs, ha="center", va="center")
    y = y2 - rh * 0.55
    ax.plot([0.03, xr], [y, y], color="k", lw=0.5)
    y -= rh * 0.12
    under = []
    for pi, (title, rows) in enumerate(panels):
        if pi:
            y -= rh * 0.1
            ax.plot([0.03, xr], [y, y], color="k", lw=0.5)
            y -= rh * 0.12
        if title:
            ax.text(x0[0] + 0.03, y - rh * 0.5, title, fontsize=fs, va="center", ha="left", style="italic")
            y -= rh
        rk = ranks({m: v for m, v in rows if v is not None}, cols)
        for m, v in rows:
            if m == "OURS":
                y -= rh * 0.08
                ax.plot([0.03, xr], [y, y], color="k", lw=0.35)
                y -= rh * 0.12
            yc = y - rh * 0.5
            y -= rh
            col = "k" if v is not None else "#9A9A9A"
            ax.text(x0[0] + 0.03, yc, NAME[m], fontsize=fs, va="center", ha="left", color=col, fontweight="bold" if m == "OURS" else "normal")
            for j, (_, _, k) in enumerate(cols, 1):
                if v is None:
                    ax.text(xc[j], yc, "–", fontsize=fs, ha="center", va="center", color=col)
                    continue
                best, second = rk[k]
                t = ax.text(xc[j], yc, f"{v[k]:.3f}", fontsize=fs, ha="center", va="center", fontweight="bold" if m in best else "normal")
                if m in second:
                    under.append(t)
    y -= rh * 0.1
    ax.plot([0.03, xr], [y, y], color="k", lw=0.9)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t in under:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (a, b_), (c, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y0)])
        ax.plot([a, c], [b_ - 0.008, b_ - 0.008], color="k", lw=0.5)
    for n in notes:
        y -= 0.135
        ax.text(0.03, y, n, fontsize=fs - 2.3, va="center", ha="left", color="#333333")
    return fig


def tex(panels, cols, caption, label):
    gs = {}
    for j, (g, _, _) in enumerate(cols, 2):
        gs.setdefault(g, []).append(j)
    L = [r"% generated by experiments/cvpr_results/exp7/scripts/make_mp3d150_report.py -- do not edit by hand", r"% needs booktabs",
         r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{%d}{c}{%s $\uparrow$}" % (len(js), g) for g, js in gs.items()) + r" \\",
         " ".join(r"\cmidrule(lr){%d-%d}" % (js[0], js[-1]) for js in gs.values()),
         "Method & " + " & ".join(s for _, s, _ in cols) + r" \\", r"\midrule"]
    for pi, (title, rows) in enumerate(panels):
        if pi:
            L.append(r"\midrule")
        if title:
            L.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (len(cols) + 1, title))
        rk = ranks({m: v for m, v in rows if v is not None}, cols)
        for m, v in rows:
            if m == "OURS" and len(rows) > 1:
                L.append(r"\cmidrule(lr){1-%d}" % (len(cols) + 1))
            if v is None:
                L.append(NAME[m] + " & " + " & ".join(["--"] * len(cols)) + r" \\")
                continue
            cells = []
            for _, _, k in cols:
                s = f"{v[k]:.3f}"
                best, second = rk[k]
                cells.append(r"\textbf{" + s + "}" if m in best else (r"\underline{" + s + "}" if m in second else s))
            L.append((r"\textbf{Ours}" if m == "OURS" else NAME[m]) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(L) + "\n"


def vals(S, split, m, keys):
    r = S.get((split, m))
    if not r or r["status"] != "COMPLETE":
        return None
    return {k: float(r[k]) for k in keys}


def curves(split_levels):
    """level-equal mean curve per method from the canonical records (VALID only, levels COMPLETE)."""
    out = {}
    for m in ["RANDOM", "MAG-B", "OURS", "OURS-H0"]:
        recs = [json.load(open(p)) for p in glob.glob(os.path.join(W7, "episodes", "mp3d150", m, "*.json"))]
        by = {}
        for r in recs:
            if r["valid"]:
                by.setdefault(r["scene_id"], []).append(np.asarray(r["coverage"]))
        out[m] = by
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    S = summary()
    keys = [k for _, _, k in COLS7]
    notes = ["MP3D: 22 levels of the 11 val houses $\\times$ 2 seeded starts, 150 steps, 1 m camera grid.",
             "Level-scale surface coverage $C_{obs}$, level-equal mean; Final = coverage after 150 steps.",
             "Bold: best; underline: second best (after rounding)."]
    rows = [(m, vals(S, "ALL-22", m, keys)) for m in MAIN]
    fig = draw([(None, rows)], COLS7, notes, width=3.6)
    RD.save(fig, os.path.join(FIG, "mp3d150_table"))
    open(os.path.join(FIG, "mp3d150_table.tex"), "w").write(tex([(None, rows)], COLS7,
        r"Active reconstruction on MP3D (22 levels of the 11 official val houses $\times$ 2 seeded starts, 150 steps, level-scale surface coverage, level-equal means). "
        r"Final: coverage after 150 steps. Bold: best; underline: second best.", "tab:mp3d150"))
    with open(os.path.join(FIG, "mp3d150_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "status", "levels_complete", "levels_total"] + keys)
        for sp in ("ALL-22", "HO-12", "VALSEL-10", "T2-5"):
            for m in MAIN + ["OURS-H0"]:
                r = S.get((sp, m))
                if r:
                    w.writerow([sp, m, r["status"], r["levels_complete"], r["levels_total"]] + [f"{float(r[k]):.4f}" if r[k] != "nan" else "" for k in keys])
    # two-panel table: Outdoor = CVPR Table 1 v2 numbers (same 6 columns), Indoor = ALL-22
    t1 = {r["method_id"]: r for r in csv.DictReader(open(os.path.join(R, "experiments", "cvpr_results", "fig", "table1", "table1_all14_full.csv")))}
    omap = {"RANDOM": "RANDOM", "MAG-B": "MAG-B", "OURS": "V3-T0-D1"}
    k6 = [k for _, _, k in COLS6]
    out_rows = [(m, {k: float(t1[omap[m]][k]) for k in k6} if t1[omap[m]]["status"] == "COMPLETE" else None) for m in MAIN]
    in_rows = [(m, vals(S, "ALL-22", m, k6)) for m in MAIN]
    panels = [("Macarons++ (outdoor, 14 scenes $\\times$ 5 starts, 100 steps)", out_rows), ("MP3D (indoor, 22 levels $\\times$ 2 starts, 150 steps)", in_rows)]
    n2 = ["Outdoor: official coverage, Final = coverage after 100 steps (Table 1).",
          "Indoor: level-scale surface coverage, Final = coverage after 150 steps.",
          "Scene / level-equal means. Bold: best; underline: second best per panel."]
    fig = draw(panels, COLS6, n2, width=3.45, name_w=0.95)
    RD.save(fig, os.path.join(FIG, "table_main_two_panel"))
    open(os.path.join(FIG, "table_main_two_panel.tex"), "w").write(tex(panels, COLS6,
        r"Active reconstruction. Outdoor: Macarons++ (14 scenes $\times$ 5 official starts, 100 steps, official coverage; Final = coverage after 100 steps). "
        r"Indoor: MP3D (22 val levels $\times$ 2 starts, 150 steps, level-scale surface coverage; Final = coverage after 150 steps). Scene / level-equal means; "
        r"bold: best, underline: second best within a panel.", "tab:main_two_panel"))
    # curves
    cv = curves(None)
    lv_rows = rd("mp3d150_levels.csv")
    complete = {(r["method"], r["scene_id"]) for r in lv_rows}
    meta = {r["scene_id"]: r["split"] for r in lv_rows}
    t2 = {r["scene_id"] for r in lv_rows if r["task2_subset"] == "1"}
    sp_lv = {"ALL-22": sorted(meta), "HO-12": sorted(s for s in meta if meta[s] == "HO"), "VALSEL-10": sorted(s for s in meta if meta[s] == "VALSEL")}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, (sp, L_) in zip(axs, sp_lv.items()):
        for m in ["RANDOM", "MAG-B", "OURS-H0", "OURS"]:
            if S.get((sp, m), {}).get("status") != "COMPLETE":
                continue
            c = np.mean([np.mean(cv[m][s], 0) for s in L_ if (m, s) in complete], 0)
            st = STY[m]
            ax.plot(np.arange(151), c, color=st["color"], ls=st["ls"], lw=st["lw"], label=NAME[m], marker=st["marker"], markevery=25, ms=3.5)
        ax.set_title(f"{sp} ({SPLIT_TXT[sp].split(' of')[0]})", fontsize=8.5)
        ax.set_xlabel("step"); ax.set_xlim(0, 150); ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("coverage $C_{obs}$")
    h, lab = axs[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.08))
    RD.save(fig, os.path.join(FIG, "mp3d150_curves"))
    # paired per level: Ours - MAGICIAN
    lv = {(r["method"], r["scene_id"]): r for r in lv_rows}
    both = [s for s in sorted(meta) if ("OURS", s) in lv and ("MAG-B", s) in lv]
    if both:
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
        order = sorted(both, key=lambda s: (meta[s], s))
        for ax, k in zip(axs, ["AUC@20", "AUC@150", "Final"]):
            d = [float(lv[("OURS", s)][k]) - float(lv[("MAG-B", s)][k]) for s in order]
            colr = ["#D55E00" if meta[s] == "HO" else "#0072B2" for s in order]
            ax.barh(np.arange(len(order)), d, color=colr, height=0.7)
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"Ours $-$ MAGICIAN, {k}", fontsize=8.5); ax.grid(axis="x", alpha=0.25, lw=0.4)
        axs[0].set_yticks(np.arange(len(order)))
        axs[0].set_yticklabels([s + ("*" if s in t2 else "") for s in order], fontsize=6)
        axs[0].invert_yaxis()
        from matplotlib.patches import Patch
        fig.legend([Patch(color="#D55E00"), Patch(color="#0072B2")], ["HO (held-out houses)", "VALSEL (validation houses)"], loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
        RD.save(fig, os.path.join(FIG, "mp3d150_paired_levels"))
    print("tables / figures written to", FIG)


if __name__ == "__main__":
    main()
