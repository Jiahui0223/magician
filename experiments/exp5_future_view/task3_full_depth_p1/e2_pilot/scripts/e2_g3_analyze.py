"""EXP5 Task 3 / E2 pilot -- G3 closed-loop analysis (seen2scene env, CPU; protocol §5 G3, exploratory run after G2 fail).

Groups B0 / B2 (frozen V2-Mix, Task 3 runs) and E2S / E2D (fine-tuned step-5000 EMA; E2S planner = B0 style, E2D = B2
style). Reuses the Task 3 episode readers and paired scene-bootstrap summary (scripts/t34_analyze.py) with the four
groups; pairs E2D-E2S (primary), E2D-B2, E2S-B0, E2D-B0; frozen G3 decision; Cov@k, behaviour, cost, MP3D mid-episode
stall (a >= 30-step window inside t in [30, 120] with coverage gain < 0.01), Macarons++ Cov@20 E2D vs B2.
Outputs analysis/g3.json, analysis/g3.md, figures/g3_coverage_curves.png
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(E2), "scripts"))
import t34_analyze as T  # noqa: E402

T.GRP = ("B0", "B2", "E2S", "E2D")
COL = {"B0": "#86b6ef", "B2": "#1c5cab", "E2S": "#f2a07b", "E2D": "#eb6834"}
LAB = {"B0": "B0 frozen V2-Mix, sparse P1", "B2": "B2 frozen V2-Mix, D100 input", "E2S": "E2-S10 fine-tuned, sparse P1",
       "E2D": "E2-D100 fine-tuned, D100 input"}
PAIRS = (("E2D", "E2S"), ("E2D", "B2"), ("E2S", "B0"), ("E2D", "B0"))
OUT, FIG = os.path.join(E2, "analysis"), os.path.join(E2, "figures")
plt = T.plt


def stall(curve, lo=30, hi=120, eps=0.01):
    """Longest window [s, e] inside [lo, hi] with coverage gain c[e] - c[s] < eps (moves)."""
    c = np.asarray(curve)
    best = 0
    for s in range(lo, hi):
        e = s
        while e + 1 <= hi and c[e + 1] - c[s] < eps:
            e += 1
        best = max(best, e - s)
    return best


def decision(P):
    m = {d: P["E2D-E2S"][d]["criteria"]["meaningful"] for d in ("indoor", "outdoor")}
    deg = {d: P["E2D-E2S"][d]["criteria"]["degraded"] for d in ("indoor", "outdoor")}
    dauc = {d: P["E2D-E2S"][d]["paired"]["auc"]["mean"] for d in ("indoor", "outdoor")}
    vs_b2 = {d: P["E2D-B2"][d]["paired"]["auc"]["mean"] for d in ("indoor", "outdoor")}
    one_meaningful = (m["indoor"] and not deg["outdoor"]) or (m["outdoor"] and not deg["indoor"])
    if one_meaningful and all(v >= 0 for v in vs_b2.values()):
        return "Go"
    if (all(v > 0 for v in dauc.values()) and not any(deg.values())) or one_meaningful:
        return "Conditional Go"
    return "No-Go"


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    T.ST.setup()
    rows_in, rows_out = T.indoor_episodes(), T.outdoor_episodes()
    P = {}
    for hi, lo in PAIRS:
        P[f"{hi}-{lo}"] = {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)}
    dec = decision(P)
    for r in rows_in:
        for g in T.GRP:
            r[g]["stall"] = stall(r[g]["curve"])
    st = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rows_in)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rows_in]))}
          for g in T.GRP}
    res = {"decision_frozen_G3_criteria": dec, "note": "exploratory: G2 did not pass (protocol §5); run on user instruction 2026-10-29",
           "B0_equals_task2": {"indoor": sum(r["B0_equals_task2"] for r in rows_in), "outdoor": sum(r["B0_equals_task2"] for r in rows_out)},
           "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P["E2D-E2S"][d]["means"] for d in ("indoor", "outdoor")}, "stall_mp3d": st,
           "per_scene": {d: P["E2D-E2S"][d]["per_scene"] for d in ("indoor", "outdoor")}}
    json.dump(res, open(os.path.join(OUT, "g3.json"), "w"), indent=1, default=float)
    # ---- tables
    f = T.fmt
    L = ["# E2 pilot — G3 闭环（探索性：G2 未通过，按用户指示运行；判定标准未改）", "", f"按冻结的 G3 标准：**{dec}**", "",
         "| 数据集 | 组 | AUC | Final | Cov@10 | Cov@20 | Cov@50 | Cov@100 | 每步时间 s | 峰值显存 GB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D（150 步）"), ("outdoor", "Macarons++（100 步）")):
        M = res["means"][d]
        for g in T.GRP:
            L.append(f"| {nm} | {LAB[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | {f(M['cov10'][g])} | {f(M['cov20'][g])} | {f(M['cov50'][g])} | "
                     f"{f(M['cov100'][g]) if 'cov100' in M else '—'} | {M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    L += ["", "| 配对差 | 数据集 | ΔAUC | 胜/负 | ΔFinal | 胜/负 | ΔCov@20 | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | "
                     f"{p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "逐场景 AUC（场景均值）：", ""]
    for d in ("indoor", "outdoor"):
        for s, v in res["per_scene"][d].items():
            L.append(f"- {s}: " + "，".join(f"{g} {v[g]['auc']:.3f}" for g in T.GRP))
    L += ["", "MP3D 中期停滞（t ∈ [30, 120] 内 ≥ 30 步覆盖增长 < 0.01 的 episode 数 / 10）：" + "，".join(f"{g} {st[g]['episodes_with_stall']}" for g in T.GRP)]
    for d in ("indoor", "outdoor"):
        L.append(f"- 行为（{d}）：" + "；".join(f"{g} 重访 {res['means'][d]['revisit_pose'][g]['mean']:.3f}" + (
            f"、平移 {res['means'][d]['translations'][g]['mean']:.1f}" if d == "indoor" else "") for g in T.GRP))
    open(os.path.join(OUT, "g3.md"), "w").write("\n".join(L) + "\n")
    # ---- curves
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, rows, nm, xmax in ((axs[0], rows_in, "MP3D (C_obs), 10 episodes", 150), (axs[1], rows_out, "Macarons++ (monotone coverage), 30 episodes", 100)):
        for g in T.GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.0, ls="--" if g in ("B0", "B2") else "-", label=f"{LAB[g]}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("move")
        ax.set_ylabel("coverage")
        ax.set_title(nm, loc="left")
        ax.legend(fontsize=8, loc="lower right")
        T.ST.clean(ax, "both")
    fig.suptitle("Task 3 / E2 pilot G3 (exploratory, G2 not passed): frozen B0 / B2 (dashed) vs fine-tuned E2-S10 / E2-D100 (solid); measured",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "g3_coverage_curves.png"))
    plt.close(fig)
    print(dec)
    print(json.dumps({k: {d: (round(v[d]["paired"]["auc"]["mean"], 4), v[d]["paired"]["auc"]["wins"], round(v[d]["paired"]["final"]["mean"], 4))
                          for d in ("indoor", "outdoor")} for k, v in res["pairs"].items()}))


if __name__ == "__main__":
    main()
