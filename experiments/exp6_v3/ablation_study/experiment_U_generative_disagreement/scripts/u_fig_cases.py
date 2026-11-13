"""Experiment U -- Figure U2 (geometric disagreement visualisation) from REAL model outputs of the Phase D replay
(work_dirs/exp6_v3/experiment_U/fixed/arrays/<tag>_t<t>.npz: the planner's own K = 4 decoded first hits of every legal candidate).
Case selection rule (fixed in this file before looking at any figure):
  A  largest within-state range of the U-best score among Dev-5 states with t <= 20 (shows high vs low disagreement candidates)
  B  'high disagreement, low gain': among results/fixed/high_u_low_gain_cases.csv rows of the U-best scorer, the one with the largest oracle max
  C  U-best pick == C1 pick == oracle-best candidate, earliest t (ties: first in file order); states with oracle max <= 0 or an all-zero U-best
     (fallback) excluded (amended 2026-11-13 after the first draft picked a degenerate all-zero state, eiffel s2 t = 0)
Per case two rows: the U-best pick and the C1 pick (if identical: the lowest-U candidate). Columns: K = 4 depth hypotheses (generated
UNKNOWN hits only; runtime ray distance; grey = known surface, white = no generated hit), per-ray 4p(1-p), top-down view (runtime x / z) of
the back-projected generated points coloured by the cell frequency p_v (U3 grid), observed S10 points in grey, camera centre.
GT is not drawn (oracle gains appear only in the panel titles, offline)."""
import csv
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
HERE = os.path.dirname(os.path.abspath(__file__))
UD = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import u_metrics as UM  # noqa: E402

W = os.path.join(R, "work_dirs", "exp6_v3", "experiment_U", "fixed")
GH, GW = 64, 114


def load_states():
    st = list(csv.DictReader(open(os.path.join(UD, "results", "fixed", "fixed_states.csv"))))
    return st


def pick_cases(st, ub):
    dev = {"colosseum", "pantheon", "eiffel", "barts", "sestino_museum"}
    rows = {}
    for f in sorted(os.listdir(os.path.join(W, "states"))):
        for line in open(os.path.join(W, "states", f)):
            r = json.loads(line)
            rows[(r["tag"], r["t"])] = r
    key = ub.lower()
    A = None
    best = -1
    for x in st:
        r = rows[(x["tag"], int(x["t"]))]
        if x["scene"] in dev and int(x["t"]) <= 20:
            v = [m[key] for m in r["u"]]
            if max(v) - min(v) > best:
                best, A = max(v) - min(v), (x["tag"], int(x["t"]))
    B = None
    p = os.path.join(UD, "results", "fixed", "high_u_low_gain_cases.csv")
    cs = [c for c in csv.DictReader(open(p)) if c.get("scorer") == ub]
    if cs:
        c = max(cs, key=lambda c: float(c["oracle_max"]))
        B = (c["tag"], int(c["t"]))
    Cc = None
    for x in sorted(st, key=lambda x: int(x["t"])):
        if x[f"pick_{ub}"] == x["pick_C1"] and float(x["top1_C1"]) == 1.0 and float(x["oracle_max"]) > 0 and x[f"fb_{ub}"] == "False":
            Cc = (x["tag"], int(x["t"]))
            break
    return {"A": A, "B": B, "C": Cc}, rows


def draw(case, tag, t, r, ub, ax_row_pairs, plt):
    z = np.load(os.path.join(W, "arrays", f"{tag}_t{t}.npz"))
    key = ub.lower()
    u = np.asarray([m[key] for m in r["u"]])
    j_u = int(np.flatnonzero(u == u.max())[0]) if u.max() > 0 else 0
    c1 = np.asarray(r["c1"])
    j_c = int(np.flatnonzero(c1 == c1.max())[0])
    same = j_c == j_u
    if same:
        j_c = int(np.argmin(u))
    for axs, j, lab in ((ax_row_pairs[0], j_u, f"{ub} pick" + (" = C1 pick" if same else "")), (ax_row_pairs[1], j_c, f"lowest {ub}" if same else "C1 pick")):
        dep, hu, rc, tk = z["depth_rt"][j], z["hit_u"][j], z["rc"][j], z["tknown_rt"][j]
        finite = dep[hu & rc[None]]
        vmax = np.percentile(finite, 98) if finite.size else 1.0
        for k in range(4):
            img = np.full(GH * GW, np.nan)
            img[hu[k] & rc] = dep[k][hu[k] & rc]
            base = np.where(np.isfinite(tk) & ~(hu[k] & rc), 0.5, 1.0).reshape(GH, GW)
            axs[k].imshow(base, cmap="gray", vmin=0, vmax=1)
            im = axs[k].imshow(img.reshape(GH, GW), cmap="viridis", vmin=0, vmax=vmax)
            if k == 3:
                plt.colorbar(im, ax=axs[k], fraction=0.035, pad=0.02).set_label("ray distance (runtime)", fontsize=6)
            axs[k].set_xticks([]); axs[k].set_yticks([])
            axs[k].set_title(f"hyp. k={k}" + (f"  ({lab})" if k == 0 else ""), fontsize=7)
        p = (hu & rc[None]).mean(0)
        b = np.where(rc, 4 * p * (1 - p), np.nan)
        axs[4].imshow(b.reshape(GH, GW), cmap="magma", vmin=0, vmax=1)
        axs[4].set_xticks([]); axs[4].set_yticks([])
        m = r["u"][j]
        axs[4].set_title(f"4p(1-p): U1 {m['u1']:.3f} U2 {m['u2']:.2f} U3 {m['u3']:.2f}", fontsize=7)
        # back-projected generated points, coloured by p_v
        o, d = z["o_rt"][j], z["d_rt"][j].astype(np.float64)
        pts, keys = [], []
        for k in range(4):
            sel = hu[k] & rc
            pk = o[None] + d[sel] * dep[k][sel][:, None]
            pts.append(pk); keys.append(np.unique(UM.cell_keys(pk)) if len(pk) else np.zeros(0, np.int64))
        allk = np.concatenate(keys) if any(len(x) for x in keys) else np.zeros(0, np.int64)
        uk, cnt = np.unique(allk, return_counts=True)
        pvm = dict(zip(uk.tolist(), (cnt / 4.0).tolist()))
        P = np.concatenate(pts) if any(len(x) for x in pts) else np.zeros((0, 3))
        ax = axs[5]
        obs = z["obs_rt"]
        box = np.concatenate([P, o[None]]) if len(P) else o[None]
        lo, hi = box.min(0) - 6.0, box.max(0) + 6.0
        if len(obs):
            ins = (obs[:, 0] >= lo[0]) & (obs[:, 0] <= hi[0]) & (obs[:, 2] >= lo[2]) & (obs[:, 2] <= hi[2])
            ax.scatter(obs[ins, 0], obs[ins, 2], s=0.3, c="#c8c8c8", edgecolors="none")
        if len(P):
            pv = np.asarray([pvm[k_] for k_ in UM.cell_keys(P)])
            sc = ax.scatter(P[:, 0], P[:, 2], s=2.0, c=pv, cmap="coolwarm_r", vmin=0.25, vmax=1.0, edgecolors="none")
            plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02).set_label("p_v (share of hypotheses)", fontsize=6)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
        ax.scatter([o[0]], [o[2]], marker="*", s=60, c="k")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        g = r["oracle_pp"][j]
        ax.set_title(f"top-down x/z; GT gain {g:.2f} pp (best {max(r['oracle_pp']):.2f})", fontsize=7)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = json.load(open(os.path.join(UD, "results", "fixed", "fixed_summary.json")))
    ub = summ["u_best"]["U_best"]
    st = load_states()
    cases, rows = pick_cases(st, ub)
    out = {}
    for name, ck in cases.items():
        if ck is None:
            out[name] = None
            continue
        tag, t = ck
        r = rows[(tag, t)]
        fig, axs = plt.subplots(2, 6, figsize=(15, 4.6))
        draw(name, tag, t, r, ub, (axs[0], axs[1]), plt)
        fig.suptitle(f"Figure U2 case {name}: {tag} t={t} ({r['scene']}), U-best = {ub}; real V3-T0 K = 4 hypotheses (predictions only; GT only in titles)", fontsize=9)
        fig.tight_layout()
        p = os.path.join(UD, "figures", f"U2_case_{name}_{tag}_t{t}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out[name] = {"tag": tag, "t": t, "file": os.path.relpath(p, UD)}
    json.dump({"u_best": ub, "cases": out}, open(os.path.join(UD, "results", "fixed", "figure_U2_cases.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
