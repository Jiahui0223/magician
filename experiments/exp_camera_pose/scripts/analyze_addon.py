"""Camera Pose -- user add-on (2026-11-02): B2 / E2-D100 / V4 / V5 under C0 and C1 and the MAGICIAN beam search (MAG-B)
on the same pilot episodes (seen2scene env, CPU). Pre-registered pilot: B2 / E2-D100 (lambda* 0.1). V4 / V5 C1 use
lambda 0.1 although their pre-registered lambda* is 0 -> EXPLORATORY. MAG-B = MAGICIAN official beam 10 x 10 on the original
grid (indoor: new runs of Task 2 t23_in_sim --group MAG-B on the VALSEL pilot episodes; outdoor: the historical Task 2
G3 MAG-B episodes of the same Dev-3 starts). Indoor metrics episode.json auc150 / cov_final / cov20 / cov50 (C_obs);
outdoor metrics auc_mono / cov_mono_final / cov_mono[20 / 50]. V4 / V5 C0 (new entry) are also compared with their
historical G3 episodes (outdoor) as a regression. Output reports/addon.{json,md}, reports/figures/addon_coverage.png."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
sys.path.insert(0, HERE)
import analyze_closed as AC  # noqa: E402

W = AC.W
IN_EPS = [("8194nk5LbLH", 0, 10), ("EU6Fwq7SyZv", 0, 67), ("X7HyMhZNoso", 0, 30), ("pLe4wQe7qrG", 0, 22)]
OUT_EPS = [(s, k) for s in ("barts", "pantheon", "colosseum") for k in (0, 1)]
ARMS = ("B2", "E2D", "V4", "V5")
HIST_OUT = {"V4": os.path.join(R, "work_dirs", "exp6_v4", "eval", "metrics", "outdoor", "V4", "V2M"), "V5": os.path.join(R, "work_dirs", "exp6_v5", "eval", "metrics", "outdoor", "V5", "V2M")}
MAG_OUT = os.path.join(R, "work_dirs", "exp5_future_view", "task2", "eval", "outdoor", "metrics", "MAG-B")


def indoor(arm, cond, ep):
    if arm == "MAGB":
        p = os.path.join(W, "MAGB_ORIG", "closed_loop", "indoor", "MAG-B", f"{ep[0]}_L{ep[1]}_S{ep[2]}", "episode.json")
        if not os.path.exists(p):
            return None
        e = json.load(open(p))
        c = np.asarray(e["coverage"], float)
        r = {"auc": float(e["auc150"]), "final": float(e["cov_final"]), "cov20": float(e["cov20"]), "cov50": float(e["cov50"]), "curve": c.tolist()}
        r.update(AC.path_stats([s["pose"] for s in e["steps"]]))
        return r
    r = AC.episode("indoor", arm, cond, ep, False)
    if r:
        e = json.load(open(AC.indoor_paths(arm, cond, *ep, False)[0]))
        r["cov50"] = float(e["cov50"])
    return r


def outdoor(arm, cond, ep):
    if arm == "MAGB":
        p = os.path.join(MAG_OUT, f"MAG-B_{ep[0]}_s{ep[1]}_r0.json")
        if not os.path.exists(p):
            return None
        m = json.load(open(p))
        c = np.asarray(m["cov_mono"], float)
        return {"auc": float(m["auc_mono"]), "final": float(m["cov_mono_final"]), "cov20": float(c[20]), "cov50": float(c[50]), "curve": c.tolist()}
    hist = arm in ("B2", "E2D") and cond == "C0"
    r = AC.episode("outdoor", arm, cond, ep, hist)
    if r:
        r["cov50"] = float(r["curve"][50])
    return r


def summ(rs):
    ok = [x for x in rs if x]
    if not ok:
        return None
    s = {k: float(np.mean([x[k] for x in ok])) for k in ("auc", "final", "cov20", "cov50")}
    s["n"] = len(ok)
    for k in ("translations", "distinct_positions"):
        if all(k in x for x in ok):
            s[k] = float(np.mean([x[k] for x in ok]))
    return s


def main():
    res = {"note": __doc__.split("\n\n")[0], "lambda": json.load(open(os.path.join(E, "configs", "c1_lambda.json"))), "domains": {}}
    curves = {}
    for dom, eps, fn in (("indoor", IN_EPS, indoor), ("outdoor", OUT_EPS, outdoor)):
        D = {"methods": {}, "paired_C1_minus_C0": {}, "vs_MAGB": {}}
        runs = {}
        for arm in ARMS:
            for cond in ("C0", "C1"):
                runs[(arm, cond)] = [fn(arm, cond, ep) for ep in eps]
        runs[("MAGB", "ORIG")] = [fn("MAGB", "ORIG", ep) for ep in eps]
        for (arm, cond), rs in runs.items():
            D["methods"][f"{arm}_{cond}"] = summ(rs)
            curves[(dom, arm, cond)] = [x["curve"] for x in rs if x]
        for arm in ARMS:
            a, b = runs[(arm, "C0")], runs[(arm, "C1")]
            pr = [(x, y) for x, y in zip(a, b) if x and y]
            if pr:
                d = [y["auc"] - x["auc"] for x, y in pr]
                D["paired_C1_minus_C0"][arm] = {"n": len(pr), "dAUC_mean": float(np.mean(d)), "dFinal_mean": float(np.mean([y["final"] - x["final"] for x, y in pr])),
                                                "wins": int(sum(v > 0 for v in d)), "losses": int(sum(v < 0 for v in d)), "min_dAUC": float(min(d)),
                                                "pre_registered": arm in ("B2", "E2D")}
            m = runs[("MAGB", "ORIG")]
            for cond in ("C0", "C1"):
                pr = [(x, y) for x, y in zip(m, runs[(arm, cond)]) if x and y]
                if pr:
                    D["vs_MAGB"][f"{arm}_{cond}"] = {"n": len(pr), "dAUC_mean": float(np.mean([y["auc"] - x["auc"] for x, y in pr])),
                                                     "dFinal_mean": float(np.mean([y["final"] - x["final"] for x, y in pr])), "wins": int(sum(y["auc"] > x["auc"] for x, y in pr))}
        if dom == "outdoor":
            reg = {}
            for arm in ("V4", "V5"):
                eq = []
                for ep, x in zip(eps, runs[(arm, "C0")]):
                    p = os.path.join(HIST_OUT[arm], f"V2M_{ep[0]}_s{ep[1]}_r0.json")
                    if x and os.path.exists(p):
                        eq.append(json.load(open(p))["cov_mono"] == x["curve"])
                reg[arm] = {"compared": len(eq), "identical_curves": int(sum(eq))}
            D["regression_new_C0_vs_historical"] = reg
        res["domains"][dom] = D
    json.dump(res, open(os.path.join(E, "reports", "addon.json"), "w"), indent=1, default=float)
    plot(curves, os.path.join(E, "reports", "figures", "addon_coverage.png"))
    L = ["# 附加比较：B2 / E2-D100 / V4 / V5 × C0 / C1，与 MAGICIAN Beam（MAG-B）（同一 pilot episode；评估场景非盲；pilot 不是完整 G3）", "",
         "λ：B2 / E2-D100 = 0.1（预注册 λ*）；V4 / V5 = 0.1（**探索性**，用户要求；它们的预注册 λ* = 0）。MAG-B：MAGICIAN 官方 beam 10×10，原网格（室内为新跑，室外为历史 Task 2 G3 的同一 episode）。", ""]
    for dom, nm in (("indoor", "MP3D VALSEL 4 episode × 150 步"), ("outdoor", "Macarons++ Dev-3 6 episode × 100 步")):
        D = res["domains"][dom]
        L += [f"## {nm}", "", "| 方法 | n | AUC | Final | Cov@20 | Cov@50 | 平移次数 | 不同位置 |", "|---|---|---|---|---|---|---|---|"]
        for k, s in D["methods"].items():
            if s:
                L.append(f"| {k} | {s['n']} | {s['auc']:.3f} | {s['final']:.3f} | {s['cov20']:.3f} | {s['cov50']:.3f} | {s.get('translations', float('nan')):.1f} | {s.get('distinct_positions', float('nan')):.1f} |")
            else:
                L.append(f"| {k} | 0 | — | — | — | — | — | — |")
        L += ["", "| 模型 | C1 − C0 ΔAUC | ΔFinal | 胜 / 负 | 最差 ΔAUC | 口径 |", "|---|---|---|---|---|---|"]
        for arm, x in D["paired_C1_minus_C0"].items():
            L.append(f"| {arm} | {x['dAUC_mean']:+.4f} | {x['dFinal_mean']:+.4f} | {x['wins']} / {x['losses']} | {x['min_dAUC']:+.4f} | {'预注册' if x['pre_registered'] else '探索性'} |")
        L += ["", "| 方法 − MAG-B | ΔAUC | ΔFinal | AUC 胜 / n |", "|---|---|---|---|"]
        for k, x in D["vs_MAGB"].items():
            L.append(f"| {k} | {x['dAUC_mean']:+.4f} | {x['dFinal_mean']:+.4f} | {x['wins']} / {x['n']} |")
        if "regression_new_C0_vs_historical" in D:
            L += ["", "V4 / V5 新入口 C0 与历史 G3 同一 episode 的覆盖曲线逐位相同：" + "，".join(f"{a} {v['identical_curves']}/{v['compared']}" for a, v in D["regression_new_C0_vs_historical"].items())]
        L.append("")
    L.append("曲线：`figures/addon_coverage.png`。")
    open(os.path.join(E, "reports", "addon.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({d: {"paired": res["domains"][d]["paired_C1_minus_C0"], "vsMAGB": res["domains"][d]["vs_MAGB"]} for d in res["domains"]}, indent=1, default=float))


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2), squeeze=False)
    for i, dom in enumerate(("indoor", "outdoor")):
        for j, arm in enumerate(ARMS):
            ax = axes[i][j]
            for (cond, col, ls, lab) in (("C0", "#495057", "-", "C0"), ("C1", "#1971c2", "-", "C1 (λ 0.1)")):
                cs = curves.get((dom, arm, cond), [])
                if cs:
                    n = min(len(x) for x in cs)
                    ax.plot(np.mean([x[:n] for x in cs], 0), color=col, lw=2.2, ls=ls, label=f"{arm} {lab} (n {len(cs)})")
            cs = curves.get((dom, "MAGB", "ORIG"), [])
            if cs:
                n = min(len(x) for x in cs)
                ax.plot(np.mean([x[:n] for x in cs], 0), color="#c92a2a", lw=1.8, ls="--", label=f"MAGICIAN beam (n {len(cs)})")
            ax.set_title(f"{'MP3D VALSEL' if dom == 'indoor' else 'Macarons++ Dev-3'} · {arm}" + ("  [C1 exploratory]" if arm in ("V4", "V5") else ""), fontsize=9.5)
            ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pilot episodes: mean coverage of C0 vs C1 per model, with MAGICIAN beam search (same episodes; pilot, not full G3)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
