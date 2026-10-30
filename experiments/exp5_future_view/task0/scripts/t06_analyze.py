"""EXP5 Task 0.6 analysis (CPU, evaluation side; protocol_v1.3_closedloop.md §4): per method (frozen, C, V2K4, V2K1)
and episode (22 levels, first start): Cov@10/20/50/100, AUC@100 (trapz / 100), the reference random policy of the same
start (W/preprocessing/traj/<level>_S<start>/traj.json); paired differences between methods and vs the reference,
cluster bootstrap over houses (10,000, seed 0); cost (wall s, mean decision s, refresh s, GPU-h); failed / retried
episodes. Output W/analysis/t06.json + figure.
"""
import glob
import json
import os

import numpy as np

W = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0"
METHODS = ["frozen", "C", "V2K4", "V2K1"]
AT = (10, 20, 50, 100)


def metrics(c):
    c = np.asarray(c, float)
    return {**{f"cov{t}": float(c[t]) for t in AT}, "auc": float(np.trapz(c, dx=1.0) / 100), "curve": c.tolist()}


def cboot(vals, groups, n=10000, seed=0):
    by = {}
    for v, g in zip(vals, groups):
        if v is not None:
            by.setdefault(g, []).append(v)
    if not by:
        return None
    cm = np.asarray([np.mean(v) for v in by.values()])
    rng = np.random.default_rng(seed)
    bs = cm[rng.integers(0, len(cm), (n, len(cm)))].mean(1)
    allv = [x for v in by.values() for x in v]
    return {"mean": float(cm.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n_pos": int(sum(x > 0 for x in allv)), "n": len(allv), "clusters": len(cm)}


def main():
    eps = [x.split() for x in open(os.path.join(W, "manifests", "cl_episodes.txt")) if x.strip()]
    rows, failed = {}, {}
    for scan, lv, st in eps:
        tag = f"{scan}_L{lv}_S{st}"
        tr = json.load(open(os.path.join(W, "preprocessing", "traj", tag, "traj.json")))
        rows[tag] = {"house": scan, "ref": metrics([r["cov"] for r in tr["rows"]])}
        for m in METHODS:
            p = os.path.join(W, "closed_loop", m, tag, "episode.json")
            if os.path.exists(p):
                e = json.load(open(p))
                st_ = e["steps"]
                ref_s = [s["refresh_s"] for s in st_ if s.get("refresh_s") is not None]
                rows[tag][m] = {**metrics(e["coverage"]), "wall_s": e["wall_s"],
                                "decision_s_mean": float(np.mean([s["decision_s"] for s in st_])),
                                "refresh_s_mean": float(np.mean(ref_s)) if ref_s else None,
                                "fallbacks": int(sum(bool(s["fallback"]) for s in st_))}
            att = os.path.join(W, "closed_loop", m, f"{tag}.attempts")
            if os.path.exists(att) or os.path.exists(os.path.join(W, "closed_loop", m, f"{tag}.failed.json")):
                failed.setdefault(m, []).append({"tag": tag, "attempts": open(att).read().strip() if os.path.exists(att) else None,
                                                 "completed": os.path.exists(p)})
    summ = {}
    houses = [rows[t]["house"] for t in rows]
    for m in ["ref"] + METHODS:
        vals = {k: [rows[t][m][k] if m in rows[t] else None for t in rows] for k in ("auc",) + tuple(f"cov{t}" for t in AT)}
        summ[m] = {k: cboot(v, houses) for k, v in vals.items()}
        summ[m]["n_episodes"] = sum(m in rows[t] for t in rows)
        if m != "ref":
            ws = [rows[t][m]["wall_s"] for t in rows if m in rows[t]]
            summ[m]["cost"] = {"wall_s_mean": float(np.mean(ws)) if ws else None, "gpu_h_total": float(sum(ws) / 3600.0),
                               "decision_s_mean": float(np.mean([rows[t][m]["decision_s_mean"] for t in rows if m in rows[t]])) if ws else None,
                               "refresh_s_mean": float(np.nanmean([rows[t][m]["refresh_s_mean"] or np.nan for t in rows if m in rows[t]])) if ws else None}
    pairs = [("V2K4", "C"), ("V2K4", "frozen"), ("C", "frozen"), ("V2K4", "V2K1"), ("V2K1", "C"), ("V2K1", "frozen")] + \
            [(m, "ref") for m in METHODS]
    paired = {}
    for a, b in pairs:
        paired[f"{a}-{b}"] = {k: cboot([rows[t][a][k] - rows[t][b][k] if a in rows[t] and b in rows[t] else None for t in rows], houses)
                              for k in ("auc", "cov20", "cov50", "cov100")}
    out = {"doc": __doc__, "summary": summ, "paired": paired, "failed_or_retried": failed, "per_episode": rows}
    os.makedirs(os.path.join(W, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(W, "analysis", "t06.json"), "w"), indent=1, default=float)
    for m in ["ref"] + METHODS:
        s = summ[m]
        print(m, s["n_episodes"], {k: round(s[k]["mean"], 3) for k in ("auc", "cov20", "cov100") if s.get(k)}, s.get("cost", ""))
    for k, v in paired.items():
        print(k, {kk: "%+.3f [%+.3f,%+.3f] %d/%d" % (vv["mean"], *vv["ci95"], vv["n_pos"], vv["n"]) for kk, vv in v.items() if vv})
    figure(out)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, ls in (("ref", "k:"), ("frozen", "C0-"), ("C", "C1-"), ("V2K4", "C2-"), ("V2K1", "C3--")):
        cs = [r[m]["curve"] for r in out["per_episode"].values() if m in r]
        if cs:
            ax.plot(np.mean(cs, 0), ls, label=f"{m} (n={len(cs)})")
    ax.set_xlabel("move")
    ax.set_ylabel("level-scale C_obs (mean over episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(W, "analysis", "t06_curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
