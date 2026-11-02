"""EXP6 V5 -- G3 closed-loop analysis (seen2scene env, CPU): exp6_v4/scripts/analyze_g3_v4.py machinery (exp6_v3/shared/
analyze_g3 readers, paired scene / floor bootstrap, Task 3 §4 criteria) with the new groups V5 (V5-Warm12k-S10Score) and
V4C (V4-Continue12k-S10Score) from work_dirs/exp6_v5/eval, the V4@40k (A0) and V3.1 groups of exp6_v4, Experiment A's
A1 (V4 + D100 DistP1 scoring, diagnostic only) and the historical references MAG-G, MAG-B, V2O, B0, B2, E2D, T0, T1.
Primary pair V5 - V4C (same parent, same data / steps), then V5 - V4, V4C - V4, V5 / V4C vs B2, E2D, MAG-G, MAG-B, V31.
Output experiments/exp6_v5/reports/g3_v5.{json,md}, figures/g3_v5_coverage_curves.png. Usage: python eval_g3.py [--groups ...]"""
import argparse
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v4", "scripts"))
import analyze_g3_v4 as AV  # noqa: E402

AG, T, G3 = AV.AG, AV.T, AV.G3
W5 = os.path.join(REPO, "work_dirs", "exp6_v5", "eval")
WA = os.path.join(REPO, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "eval")
E5 = os.path.join(REPO, "experiments", "exp6_v5")
OUT, FIG = os.path.join(E5, "reports"), os.path.join(E5, "figures")
TASK2 = AV.TASK2
GRP = ("MAG-G", "MAG-B", "V2O", "B0", "B2", "E2D", "T0", "T1", "V31", "V4", "A1", "V4C", "V5")
PAIRS = (("V5", "V4C"), ("V5", "V4"), ("V4C", "V4"), ("V5", "B2"), ("V4C", "B2"), ("V5", "E2D"), ("V4C", "E2D"), ("V5", "MAG-G"), ("V5", "MAG-B"),
         ("V4C", "MAG-G"), ("V4C", "MAG-B"), ("V5", "V31"), ("V5", "V2O"), ("V5", "A1"), ("A1", "V4"))
LAB = dict(AV.LAB, V4="V4@40k Outdoor-only (A0, S10 score)", A1="V4 + D100 DistP1 score (Experiment A1, diagnostic)",
           V4C="V4-Continue12k-S10Score (control)", V5="V5-Warm12k-S10Score")
SRC = dict(AV.SRC, A1="V4@40k, D100 / D100", V4C="V4@40k + 12k Outdoor-only, D100 / sparse", V5="V4@40k U-Net+Flow + new decoder, Bridge2k + Joint10k Outdoor-only, D100 / sparse")
COL = dict(AV.COL, A1="#f59f00", V4C="#495057", V5="#7048e8")
AG.ROOTS.update({"A1": (os.path.join(WA, "closed_loop"), os.path.join(WA, "metrics", "outdoor"), "V2M")})
AG.ROOTS.update({g: (os.path.join(W5, "closed_loop"), os.path.join(W5, "metrics", "outdoor"), "V2M") for g in ("V5", "V4C")})


def main():
    global GRP, PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.groups:
        GRP = tuple(g for g in GRP if g in a.groups)
        PAIRS = tuple((x, y) for x, y in PAIRS if x in GRP and y in GRP)
    T.GRP = GRP
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    T.ST.setup()
    rows_in, rows_out = AG.indoor_episodes(), AG.outdoor_episodes()
    P = {f"{hi}-{lo}": {"indoor": T.summarize(rows_in, "indoor", hi, lo), "outdoor": T.summarize(rows_out, "outdoor", hi, lo)} for hi, lo in PAIRS}
    for r in rows_in:
        for g in GRP:
            r[g]["stall"] = G3.stall(r[g]["curve"])
    st = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rows_in)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rows_in]))} for g in GRP}
    dec = {}
    for hi, lo in PAIRS:
        k = f"{hi}-{lo}"
        c = {d: P[k][d]["criteria"] for d in ("indoor", "outdoor")}
        dec[k] = {"indoor": c["indoor"], "outdoor": c["outdoor"],
                  "verdict": "meaningful on both" if c["indoor"]["meaningful"] and c["outdoor"]["meaningful"] else
                  ("degraded" if c["indoor"]["degraded"] or c["outdoor"]["degraded"] else
                   ("meaningful on one" if c["indoor"]["meaningful"] or c["outdoor"]["meaningful"] else "neither"))}
    primary = f"{PAIRS[0][0]}-{PAIRS[0][1]}"
    per_ep = {"indoor": [{"scene": r["scene"], "start": r["start"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "stall", "translations", "time_per_move", "peak_gb")} for g in GRP}} for r in rows_in],
              "outdoor": [{"scene": r["scene"], "start": r["start"], "split": r["split"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "hover", "time_per_move", "peak_gb")} for g in GRP}} for r in rows_out]}
    res = {"groups": list(GRP), "sources": {g: SRC[g] for g in GRP}, "B0_equals_task2": {"indoor": sum(r["B0_equals_task2"] for r in rows_in), "outdoor": sum(r["B0_equals_task2"] for r in rows_out)},
           "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P[primary][d]["means"] for d in ("indoor", "outdoor")}, "stall_mp3d": st, "decisions": dec,
           "per_scene": {d: P[primary][d]["per_scene"] for d in ("indoor", "outdoor")}, "per_episode": per_ep}
    json.dump(res, open(os.path.join(OUT, f"g3_v5{a.suffix}.json"), "w"), indent=1, default=float)
    f = T.fmt
    L = ["# V5 G3 闭环（自动生成，`exp6_v5/scripts/eval_g3.py`；均值 = 场景 / 楼层均值的平均，[CI] = 场景级 bootstrap 95 %；评估场景非盲）", "",
         "| 数据集 | 组 | 训练来源 / P1 口径（输入 / 评分） | AUC | Final | Cov@20 | Cov@50 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D zero-shot（10 ep × 150 步）"), ("outdoor", "Macarons++（30 ep × 100 步）")):
        M = res["means"][d]
        for g in GRP:
            L.append(f"| {nm} | {LAB[g]} | {SRC[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | {f(M['cov20'][g])} | {f(M['cov50'][g])} | "
                     f"{M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    L += ["", "\\* 每步时间为各自运行时的共享负载记录，不可直接比较。", "", "| 配对差 | 数据集 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | "
                     f"{p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "判定（Task 3 §4 标准）：" + "；".join(f"{k}: {v['verdict']}" for k, v in dec.items()), "", "逐场景 / 楼层 AUC / Final / Cov@20：", ""]
    for d in ("indoor", "outdoor"):
        for s, v in res["per_scene"][d].items():
            L.append(f"- {s}: " + "，".join(f"{g} {v[g]['auc']:.3f} / {v[g].get('final', float('nan')):.3f} / {v[g].get('cov20', float('nan')):.3f}" for g in GRP))
    L += ["", "MP3D 中期停滞 episode 数（/ 10）：" + "，".join(f"{g} {st[g]['episodes_with_stall']}" for g in GRP)]
    open(os.path.join(OUT, f"g3_v5{a.suffix}.md"), "w").write("\n".join(L) + "\n")
    plt = AG.plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, rows, nm, xmax in ((axs[0], rows_in, "MP3D zero-shot (C_obs), 10 episodes", 150), (axs[1], rows_out, "Macarons++ (monotone coverage), 30 episodes", 100)):
        for g in GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.4 if g in ("V5", "V4C", "V4") else 1.2, ls=":" if g in TASK2 else ("--" if g in ("B0", "B2", "E2D", "A1") else "-"),
                    label=f"{g}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax); ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.set_title(nm, loc="left")
        ax.legend(fontsize=7, loc="lower right")
        T.ST.clean(ax, "both")
    fig.suptitle("EXP6 V5 G3: V5 / V4-Continue12k / V4@40k (thick) vs history; measured, not blind", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"g3_v5{a.suffix}_coverage_curves.png")); plt.close(fig)
    print(json.dumps({k: v["verdict"] for k, v in dec.items()}))


if __name__ == "__main__":
    main()
