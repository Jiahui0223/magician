"""Small rendering helpers: booktabs-style table -> PDF (vector, embedded TrueType) + PNG (300 dpi), and the shared figure style."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300, "figure.dpi": 120})
# fixed entity colours (Okabe-Ito, colour-blind safe; same assignment as paper_tables/config.yaml) + line style / marker as a second encoding
STYLE = {"V3-T0-D1": dict(color="#D55E00", ls="-", marker="o", lw=2.0, label="Ours (V3-T0 + D1)"),
         "MAG-B": dict(color="#555555", ls="--", marker="s", lw=1.5, label="MAGICIAN"),
         "MACARONS": dict(color="#0072B2", ls="-.", marker="^", lw=1.3, label="MACARONS"),
         "RANDOM": dict(color="#999999", ls=":", marker="x", lw=1.3, label="Random Walk"),
         "V3-T0-K4D1": dict(color="#E69F00", ls="-", marker="D", lw=1.3, label="K = 4, depth 1"),
         "V3-T0-K1D2": dict(color="#009E73", ls="--", marker="v", lw=1.3, label="K = 1, depth 2"),
         "V3-T0-K1D1": dict(color="#56B4E9", ls=":", marker="P", lw=1.3, label="K = 1, depth 1")}
SPLIT_COLOR = {"train": "#333333", "dev": "#009E73", "test": "#CC79A7"}
SPLIT_TAG = {"train": "Train", "dev": "Dev", "test": "Test"}


def save(fig, base):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=300, metadata=({"CreationDate": None} if ext == "pdf" else None))
    plt.close(fig)


def table(base, header, rows, bold=(), notes=(), col_w=None, title=None, rules_after=(), align=None, fs=8.5):
    """rows: list of lists of str; bold: set of (row, col); rules_after: row indices followed by a thin rule."""
    n_c = len(header)
    col_w = col_w or [max(len(str(header[j])), max((len(str(r[j])) for r in rows), default=0)) * 0.075 + 0.25 for j in range(n_c)]
    align = align or (["left"] + ["center"] * (n_c - 1))
    W, rh = sum(col_w), 0.21
    import textwrap
    notes = [ln for n in notes for ln in textwrap.wrap(n, width=max(40, int(W / 0.0475)))]      # explicit line breaks (no renderer-dependent wrapping)
    H = rh * (len(rows) + 1.6) + (0.25 if title else 0) + 0.15 * len(notes) + 0.12
    fig = plt.figure(figsize=(W + 0.2, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W + 0.2)
    ax.set_ylim(0, H)
    ax.axis("off")
    y = H - 0.08
    if title:
        ax.text(0.1, y - 0.08, title, fontsize=fs + 0.5, fontweight="bold", va="center")
        y -= 0.25
    x0 = [0.1 + sum(col_w[:j]) for j in range(n_c)]

    def cell(j, yy, s, b=False):
        x = x0[j] + (0.04 if align[j] == "left" else col_w[j] / 2 if align[j] == "center" else col_w[j] - 0.04)
        ax.text(x, yy, s, fontsize=fs, ha=align[j], va="center", fontweight="bold" if b else "normal")
    ax.plot([0.1, 0.1 + W], [y, y], color="k", lw=1.0)
    y -= rh * 0.62
    for j, h in enumerate(header):
        cell(j, y, h)
    y -= rh * 0.55
    ax.plot([0.1, 0.1 + W], [y, y], color="k", lw=0.6)
    for i, r in enumerate(rows):
        y -= rh * 0.55
        for j, s in enumerate(r):
            cell(j, y, str(s), (i, j) in bold)
        y -= rh * 0.45
        if i in rules_after:
            ax.plot([0.1, 0.1 + W], [y, y], color="k", lw=0.35)
    ax.plot([0.1, 0.1 + W], [y, y], color="k", lw=1.0)
    y -= 0.03
    for n in notes:
        y -= 0.15
        ax.text(0.1, y, n, fontsize=fs - 2.2, va="center", color="#333333")
    save(fig, base)


def tex(path, header, rows, bold=(), caption="", label="", notes=(), rules_after=(), colspec=None):
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("—", "--").replace("−", "$-$").replace("×", r"$\times$").replace("Δ", r"$\Delta$").replace("†", r"$^{\dagger}$").replace("‡", r"$^{\ddagger}$")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + (colspec or ("l" + "c" * (len(header) - 1))) + "}", r"\toprule",
         " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append(" & ".join((r"\textbf{" + esc(s) + "}") if (i, j) in bold else esc(s) for j, s in enumerate(r)) + r" \\")
        if i in rules_after:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        L.append(r"\caption{" + esc(caption) + "}")
    if label:
        L.append(r"\label{" + label + "}")
    L.append(r"\end{table}")
    L += ["% " + n for n in notes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
