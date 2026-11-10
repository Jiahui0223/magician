"""CVPR Table 1, paper layout v2 (CPU): two-level header (AUC up | Coverage up), best bold / second-best underlined, a
'Delta vs. MAGICIAN' row (scene-paired difference, * = scene-bootstrap 95 % CI excludes 0), unavailable baselines greyed.
Same numbers as table1_paper.csv (read from table1_all14_full.csv / table1_paired_ours_minus_baseline.csv, which are generated
from manifest/episodes.csv). Output: fig/table1/table1_v2.{pdf,png,tex,csv}. No LaTeX on the cluster: the PDF / PNG are drawn
with matplotlib (STIX = Times-compatible, vector, embedded TrueType); the .tex is booktabs and only syntax-checked here."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

plt = RD.plt
OUT = os.path.join(L.FIG, "table1")
COLS = [("AUC", "@20", "AUC@20"), ("AUC", "@50", "AUC@50"), ("AUC", "@100", "AUC@100"), ("Coverage", "@20", "Cov@20"), ("Coverage", "Final", "Final")]
NAMES = {"RANDOM": "Random Walk$^{\\dagger}$", "SCONE": "SCONE", "MACARONS": "MACARONS$^{\\dagger}$", "FISHERRF": "FisherRF", "MAG-B": "MAGICIAN", "V3-T0-D1": "Ours"}
TEXN = {"RANDOM": "Random Walk$^{\\dagger}$", "SCONE": "SCONE", "MACARONS": "MACARONS$^{\\dagger}$", "FISHERRF": "FisherRF", "MAG-B": "MAGICIAN", "V3-T0-D1": "Ours"}


def load():
    full = {r["method_id"]: r for r in csv.DictReader(open(os.path.join(OUT, "table1_all14_full.csv")))}
    pr = {(r["method_b"], r["metric"]): r for r in csv.DictReader(open(os.path.join(OUT, "table1_paired_ours_minus_baseline.csv")))}
    return full, pr


def ranks(full):
    """per metric: (best set, second set) among complete methods, compared after rounding to 3 decimals."""
    comp = [m for m in L.TABLE1 if full[m]["status"] == "COMPLETE"]
    out = {}
    for _, _, k in COLS:
        v = sorted({round(float(full[m][k]), 3) for m in comp}, reverse=True)
        out[k] = ({m for m in comp if round(float(full[m][k]), 3) == v[0]}, {m for m in comp if len(v) > 1 and round(float(full[m][k]), 3) == v[1]})
    return out


def draw(full, pr, base, width=3.35, fs=8.0):
    rk = ranks(full)
    methods = L.TABLE1
    nrow = len(methods) + 1
    cw = [1.10] + [(width - 1.10 - 0.05) / len(COLS)] * len(COLS)
    rh = 0.165
    H = rh * (nrow + 2.5) + 0.55
    fig = plt.figure(figsize=(width, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, width); ax.set_ylim(0, H); ax.axis("off")
    x0 = [0.03 + sum(cw[:j]) for j in range(len(cw))]
    xc = [x0[j] + cw[j] / 2 for j in range(len(cw))]
    xr = x0[-1] + cw[-1]
    y = H - 0.06
    ax.plot([0.03, xr], [y, y], color="k", lw=0.9)                                           # top rule
    y1 = y - rh * 0.62
    ax.text(x0[0] + 0.03, y1 - rh * 0.5, "Method", fontsize=fs, va="center", ha="left")
    for grp, (a, b) in (("AUC $\\uparrow$", (1, 3)), ("Coverage $\\uparrow$", (4, 5))):
        ax.text((x0[a] + x0[b] + cw[b]) / 2, y1, grp, fontsize=fs, ha="center", va="center")
        ax.plot([x0[a] + 0.04, x0[b] + cw[b] - 0.04], [y1 - rh * 0.42] * 2, color="k", lw=0.45)   # cmidrule
    y2 = y1 - rh * 0.95
    for j, (_, sub, _) in enumerate(COLS, 1):
        ax.text(xc[j], y2, sub, fontsize=fs, ha="center", va="center")
    y = y2 - rh * 0.55
    ax.plot([0.03, xr], [y, y], color="k", lw=0.5)                                           # mid rule
    y -= rh * 0.12                                                                           # y = top edge of the next row
    texts = []
    for i, m in enumerate(methods):
        if m == "V3-T0-D1":                                                                  # thin rule between the baselines and Ours
            y -= rh * 0.08
            ax.plot([0.03, xr], [y, y], color="k", lw=0.35)
            y -= rh * 0.12
        yc = y - rh / 2                                                                      # row centre
        y -= rh
        avail = full[m]["status"] == "COMPLETE"
        col = "k" if avail else "#9A9A9A"
        ax.text(x0[0] + 0.03, yc, NAMES[m], fontsize=fs, va="center", ha="left", color=col, fontweight="bold" if m == "V3-T0-D1" else "normal")
        for j, (_, _, k) in enumerate(COLS, 1):
            if not avail:
                ax.text(xc[j], yc, "–", fontsize=fs, ha="center", va="center", color=col)
                continue
            best, second = rk[k]
            t = ax.text(xc[j], yc, f"{float(full[m][k]):.3f}", fontsize=fs, ha="center", va="center", fontweight="bold" if m in best else "normal")
            if m in second:
                texts.append(t)
    yc = y - rh * 0.5
    y -= rh
    ax.text(x0[0] + 0.03, yc, "$\\Delta$ vs. MAGICIAN", fontsize=fs - 1.0, va="center", ha="left", color="#444444", style="italic")
    for j, (_, _, k) in enumerate(COLS, 1):
        r = pr[(base, k)]
        sig = float(r["ci95_lo"]) > 0 or float(r["ci95_hi"]) < 0
        ax.text(xc[j], yc, f"{float(r['mean_diff']):+.3f}" + ("$^{*}$" if sig else ""), fontsize=fs - 1.0, ha="center", va="center", color="#444444", style="italic")
    y -= rh * 0.1
    ax.plot([0.03, xr], [y, y], color="k", lw=0.9)                                           # bottom rule
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t in texts:                                                                            # underline the second-best values
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (a, b_), (c, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y0)])
        ax.plot([a, c], [b_ - 0.008, b_ - 0.008], color="k", lw=0.5)
    notes = ["Macarons++ All-14 (14 scenes $\\times$ 5 official starts, 100 steps); official coverage, scene-equal mean.",
             "Bold: best; underline: second best. $\\dagger$ adapted under the same frozen protocol; – no runnable",
             "implementation under this protocol. $^{*}$ scene-bootstrap 95% CI of the paired difference excludes 0."]
    for n in notes:
        y -= 0.135
        ax.text(0.03, y, n, fontsize=fs - 2.3, va="center", ha="left", color="#333333")
    return fig


def tex(full, pr, base):
    rk = ranks(full)
    L_ = [r"% Table 1 (paper layout v2), generated by experiments/cvpr_results/scripts/make_table1_v2.py -- do not edit by hand",
          r"% needs \usepackage{booktabs} (and \usepackage{xcolor} for the grey rows)", r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
          r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r" & \multicolumn{3}{c}{AUC $\uparrow$} & \multicolumn{2}{c}{Coverage $\uparrow$} \\", r"\cmidrule(lr){2-4} \cmidrule(l){5-6}",
          r"Method & @20 & @50 & @100 & @20 & Final \\", r"\midrule"]
    for m in L.TABLE1:
        if m == "V3-T0-D1":
            L_.append(r"\midrule")
        if full[m]["status"] != "COMPLETE":
            L_.append(r"\textcolor{gray}{" + TEXN[m] + "} & " + " & ".join([r"\textcolor{gray}{--}"] * len(COLS)) + r" \\")
            continue
        cells = []
        for _, _, k in COLS:
            v = f"{float(full[m][k]):.3f}"
            best, second = rk[k]
            cells.append(r"\textbf{" + v + "}" if m in best else (r"\underline{" + v + "}" if m in second else v))
        L_.append((r"\textbf{Ours}" if m == "V3-T0-D1" else TEXN[m]) + " & " + " & ".join(cells) + r" \\")
    d = []
    for _, _, k in COLS:
        r = pr[(base, k)]
        sig = float(r["ci95_lo"]) > 0 or float(r["ci95_hi"]) < 0
        d.append(r"\textit{" + f"{float(r['mean_diff']):+.3f}".replace("-", "$-$") + "}" + (r"$^{*}$" if sig else ""))
    L_ += [r"\textit{\footnotesize $\Delta$ vs.\ MAGICIAN} & " + " & ".join(r"{\footnotesize " + x + "}" for x in d) + r" \\", r"\bottomrule", r"\end{tabular}",
           r"\caption{Active reconstruction on Macarons++ (14 scenes $\times$ 5 official start poses, 100 steps, official coverage; scene-equal means). "
           r"Bold: best; underline: second best. $^{\dagger}$Adapted under the same frozen protocol. --: no runnable implementation under this protocol. "
           r"$\Delta$: scene-paired difference to MAGICIAN; $^{*}$its scene-bootstrap 95\% CI excludes 0.}", r"\label{tab:main}", r"\end{table}"]
    return "\n".join(L_) + "\n"


if __name__ == "__main__":
    full, pr = load()
    fig = draw(full, pr, "MAG-B")
    RD.save(fig, os.path.join(OUT, "table1_v2"))
    open(os.path.join(OUT, "table1_v2.tex"), "w").write(tex(full, pr, "MAG-B"))
    rows = [[L.METHODS[m]["display"], full[m]["status"]] + ([f"{float(full[m][k]):.3f}" for _, _, k in COLS] if full[m]["status"] == "COMPLETE" else [""] * len(COLS)) for m in L.TABLE1]
    rows.append(["Delta vs. MAGICIAN (scene-paired)", ""] + [f"{float(pr[('MAG-B', k)]['mean_diff']):+.3f} [{float(pr[('MAG-B', k)]['ci95_lo']):+.3f}, {float(pr[('MAG-B', k)]['ci95_hi']):+.3f}]" for _, _, k in COLS])
    L.write_csv(os.path.join(OUT, "table1_v2.csv"), ["Method", "status", "AUC@20 (up)", "AUC@50 (up)", "AUC@100 (up)", "Coverage@20 (up)", "Final coverage = Coverage@100 (up)"], rows)
    print("table1_v2 written")
