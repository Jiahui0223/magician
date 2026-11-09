"""Early-phase (AUC@20) comparison of V3-D1 against MAGICIAN on the CVPR Table 1 records.
Reads the canonical episode records (official MAGICIAN coverage, replay-verified) and writes
experiments/exp6_v3/figures/early_phase_vs_magician.{pdf,png} + early_phase_vs_magician.csv."""
import glob, json, os, yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
TB = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
OUT = os.path.join(R, "experiments", "exp6_v3", "figures")
reg = yaml.safe_load(open(os.path.join(R, "experiments", "cvpr_results", "table1", "SCENE_REGISTRY.yaml")))["scenes"]
NAME = {s["local_id"]: s["official_name"] for s in reg}
SPLIT = {s["local_id"]: s["split"] for s in reg}
PAPER = {s["local_id"]: s["magician_paper"] for s in reg}
MODELS = {"MAG-B": ("MAGICIAN (beam 10x10, re-evaluated)", "#555555", "--"), "MAG-G": ("MAGICIAN (greedy)", "#999999", ":"), "V3-D1-H3-NORISK": ("Ours: V3 + D1", "#D55E00", "-")}
rng = np.random.default_rng(0)


def load(model):
    out = {}
    for f in glob.glob(os.path.join(TB, model, "*.json")):
        r = json.load(open(f)); out[(r["scene"], int(r["start"]))] = np.asarray(r["coverage_official"], float)
    return out


E = {m: load(m) for m in MODELS}
ALL14 = sorted({k[0] for k in E["V3-D1-H3-NORISK"]} & {k[0] for k in E["MAG-B"]})
assert all(sum(1 for k in E[m] if k[0] == s) == 5 for m in MODELS for s in ALL14), "incomplete scene"
GROUPS = {"Train-6": [s for s in ALL14 if SPLIT[s].startswith("train")], "Dev-5": [s for s in ALL14 if SPLIT[s].startswith("dev")], "Test-3": [s for s in ALL14 if SPLIT[s].startswith("test")], "All-14": ALL14}
scene_curve = {m: {s: np.mean([E[m][(s, k)] for k in range(5)], 0) for s in ALL14} for m in MODELS}


def auc(c, T): return float(np.trapz(c[:T + 1]) / T)
def t_reach(c, level):
    i = np.argmax(c >= level); return int(i) if c[i] >= level else np.nan


def boot_mean(vals, n=10000):
    vals = np.asarray(vals); idx = rng.integers(0, len(vals), (n, len(vals))); b = vals[idx].mean(1)
    return float(vals.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_curve(curves, n=10000):
    A = np.stack(curves); idx = rng.integers(0, len(A), (n, len(A))); B = A[idx].mean(1)
    return A.mean(0), np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)


# ------------------------------------------------------------------ numbers
rows = []
for g, ss in GROUPS.items():
    for m in MODELS:
        for met, fn in (("AUC@20", lambda c: auc(c, 20)), ("Cov@20", lambda c: float(c[20])), ("AUC@50", lambda c: auc(c, 50)), ("AUC@100", lambda c: auc(c, 100)), ("Final", lambda c: float(c[100])), ("steps_to_50", lambda c: t_reach(c, 0.5)), ("steps_to_70", lambda c: t_reach(c, 0.7))):
            v = [fn(scene_curve[m][s]) for s in ss]
            mu, lo, hi = boot_mean(v); rows.append((g, m, met, mu, lo, hi))
    # paired difference ours - MAG-B
    for met, fn in (("AUC@20", lambda c: auc(c, 20)), ("Cov@20", lambda c: float(c[20])), ("AUC@50", lambda c: auc(c, 50)), ("AUC@100", lambda c: auc(c, 100)), ("Final", lambda c: float(c[100]))):
        d = [fn(scene_curve["V3-D1-H3-NORISK"][s]) - fn(scene_curve["MAG-B"][s]) for s in ss]
        mu, lo, hi = boot_mean(d); rows.append((g, "V3-D1 minus MAG-B", met, mu, lo, hi))
with open(os.path.join(OUT, "early_phase_vs_magician.csv"), "w") as f:
    f.write("group,model,metric,mean,ci_lo,ci_hi\n")
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
per_scene = {s: {m: auc(scene_curve[m][s], 20) for m in MODELS} for s in ALL14}
per_scene_se = {s: float(np.std([auc(E["V3-D1-H3-NORISK"][(s, k)], 20) - auc(E["MAG-B"][(s, k)], 20) for k in range(5)], ddof=1) / np.sqrt(5)) for s in ALL14}

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
fig = plt.figure(figsize=(12.8, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0], wspace=0.32)

# (a) zoomed curves, All-14
ax = fig.add_subplot(gs[0, 0])
T = 30
for m, (lab, col, ls) in MODELS.items():
    mu, lo, hi = boot_curve([scene_curve[m][s] for s in ALL14])
    if m != "MAG-G": ax.fill_between(np.arange(T + 1), lo[:T + 1], hi[:T + 1], color=col, alpha=0.13, lw=0)
    ax.plot(np.arange(T + 1), mu[:T + 1], color=col, ls=ls, lw=2.0 if m.startswith("V3") else 1.5, label=lab)
    if m != "MAG-G":
        t50 = t_reach(mu, 0.5); ax.plot([t50], [0.5], marker="o", ms=4.5, color=col, mec="white", mew=0.8)
        ax.annotate(("MAGICIAN: " if m.startswith("MAG") else "Ours: ") + f"{t50} steps to 50%", (t50, 0.5), xytext=(21.0, 0.21 if m.startswith("MAG") else 0.13), fontsize=7.3, color=col, va="center", ha="left", arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkB=3))
ax.axvspan(0, 20, color="#000000", alpha=0.04, lw=0); ax.axvline(20, color="#888888", lw=0.8, ls=(0, (2, 2)))
ax.text(10, 0.012, "AUC@20 window", ha="center", va="bottom", fontsize=7.5, color="#555555")
ax.set_xlim(0, T); ax.set_ylim(0, 0.85); ax.set_xlabel("Exploration step"); ax.set_ylabel("Surface coverage (official)")
ax.set_title("(a) Early phase, All-14 (14 scenes x 5 starts)"); ax.grid(color="#DDDDDD", lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# (b) grouped bars: AUC@20 / Cov@20 / AUC@50 by group, MAG-B vs ours
ax = fig.add_subplot(gs[0, 1])
mets = ("AUC@20", "Cov@20", "AUC@50"); groups = ("Train-6", "Dev-5", "Test-3", "All-14")
w = 0.36; x0 = np.arange(len(groups))
R_ = {(r[0], r[1], r[2]): r[3:] for r in rows}
for j, met in enumerate(mets):
    off = (j - 1) * 1.0 * (len(groups) + 1.2)   # separate blocks per metric
for gi, g in enumerate(groups):
    for mi, m in enumerate(("MAG-B", "V3-D1-H3-NORISK")):
        lab, col, _ = MODELS[m]
        mu, lo, hi = R_[(g, m, "AUC@20")]
        ax.bar(gi + (mi - 0.5) * w, mu, w * 0.92, color=col, alpha=0.9 if m.startswith("V3") else 0.75, lw=0)
        ax.errorbar(gi + (mi - 0.5) * w, mu, yerr=[[mu - lo], [hi - mu]], color="#222222", lw=0.9, capsize=2.2)
        ax.text(gi + (mi - 0.5) * w, 0.015, f"{mu:.3f}", ha="center", va="bottom", fontsize=7.2, color="white", rotation=90)
    d, dlo, dhi = R_[(g, "V3-D1 minus MAG-B", "AUC@20")]
    ax.text(gi, max(R_[(g, "MAG-B", "AUC@20")][2], R_[(g, "V3-D1-H3-NORISK", "AUC@20")][2]) + 0.02, f"$\\Delta$ {d:+.3f}\n[{dlo:+.2f}, {dhi:+.2f}]", ha="center", fontsize=6.9, color="#222222", linespacing=1.1)
ax.set_xticks(x0); ax.set_xticklabels(groups); ax.set_ylim(0, 0.70); ax.set_ylabel("AUC@20 (official coverage)")
ax.set_title("(b) AUC@20 by split, scene-bootstrap 95% CI"); ax.grid(axis="y", color="#DDDDDD", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(handles=[Line2D([], [], color=MODELS["MAG-B"][1], lw=6, label="MAGICIAN (re-eval.)"), Line2D([], [], color=MODELS["V3-D1-H3-NORISK"][1], lw=6, label="Ours: V3 + D1")], loc="upper right", frameon=False, fontsize=7.6, handlelength=1.4)

# (c) per-scene paired AUC@20
ax = fig.add_subplot(gs[0, 2])
mk = {"train": "s", "dev": "o", "test": "^"}
for s in ALL14:
    xm, ym = per_scene[s]["MAG-B"], per_scene[s]["V3-D1-H3-NORISK"]
    ax.errorbar(xm, ym, yerr=per_scene_se[s], fmt=mk[SPLIT[s].split("_")[0] if SPLIT[s] in mk else "o"], ms=5.5, color=MODELS["V3-D1-H3-NORISK"][1], mec="white", mew=0.6, elinewidth=0.8, ecolor="#BBBBBB", capsize=0)
    short = NAME[s].replace(" Castle", "").replace(" Cathedral", "").replace(" Palace", "").replace("Statue of ", "").replace("Christ the ", "").replace("Manhattan ", "").replace(" Museum", "").replace(" Tower", "")
    OFF = {"sestino_museum": (-4, -9, "right"), "eiffel": (0, 7, "center"), "colosseum": (-5, -9, "right"), "barts": (-5, 3, "right"), "redeemer": (-5, 4, "right"), "pantheon": (5, -8, "left"), "alhambra": (5, 1, "left"), "neuschwanstein": (5, -7, "left"), "liberty": (5, -6, "left"), "fushimi": (5, 1, "left"), "bannerman": (5, -6, "left"), "pisa": (5, -7, "left"), "bridge": (5, 2, "left"), "dunnottar": (5, -7, "left")}
    dx, dy, ha = OFF.get(s, (4, 2, "left"))
    ax.annotate(short, (xm, ym), xytext=(dx, dy), textcoords="offset points", fontsize=6.6, color="#333333", ha=ha)
lim = (0.1, 0.68); ax.plot(lim, lim, color="#888888", lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("MAGICIAN AUC@20"); ax.set_ylabel("Ours (V3 + D1) AUC@20"); ax.set_title("(c) Per-scene AUC@20 (mean of 5 starts)")
ax.text(0.12, 0.655, "above the diagonal = ours better", fontsize=7.2, color="#555555", va="top")
ax.legend(handles=[Line2D([], [], marker="s", ls="", color="#777777", label="Train-6"), Line2D([], [], marker="o", ls="", color="#777777", label="Dev-5"), Line2D([], [], marker="^", ls="", color="#777777", label="Test-3")], loc="lower right", frameon=False, fontsize=7.4)
ax.grid(color="#DDDDDD", lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

fig.savefig(os.path.join(OUT, "early_phase_vs_magician.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "early_phase_vs_magician.png"), dpi=180, bbox_inches="tight")

# ------------------------------------------------------------------ console summary
print("group | metric | MAG-B | V3-D1 | diff [95% CI]")
for g in groups:
    for met in ("AUC@20", "Cov@20", "AUC@50", "AUC@100", "Final"):
        a = R_[(g, "MAG-B", met)][0]; b = R_[(g, "V3-D1-H3-NORISK", met)][0]; d = R_[(g, "V3-D1 minus MAG-B", met)]
        print(f"{g} | {met} | {a:.3f} | {b:.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    for met in ("steps_to_50", "steps_to_70"):
        print(f"{g} | {met} | {R_[(g, 'MAG-B', met)][0]:.1f} | {R_[(g, 'V3-D1-H3-NORISK', met)][0]:.1f}")
print("scenes where ours < MAG-B on AUC@20:", [s for s in ALL14 if per_scene[s]["V3-D1-H3-NORISK"] < per_scene[s]["MAG-B"]])
print("paper MAGICIAN AUC@100 All-14 (no AUC@20 reported):", round(np.mean([PAPER[s]["auc"]["magician"] for s in ALL14]), 3))
