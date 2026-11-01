"""Experiment A -- G3 analysis (CPU): exp6_v4/scripts/analyze_g3_v4.py machinery with the new group A1 (work_dirs/exp6_v4/
ablation_study/experiment_A/eval) against A0 = V4 (exp6_v4/eval) and the historical references. Primary pair A1-V4;
also A1-B2, A1-E2D, A1-V2O, A1-V31, A1-MAG-G, A1-MAG-B. Output results/a_g3.{json,md}, figures/a_g3_coverage_curves.png."""
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v4", "scripts"))
import analyze_g3_v4 as AV  # noqa: E402

AG, T, G3 = AV.AG, AV.T, AV.G3
A = os.path.join(REPO, "experiments", "exp6_v4", "ablation_study", "experiment_A_distp1_dense_vs_sparse")
WA = os.path.join(REPO, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "eval")
GRP = ("MAG-G", "MAG-B", "V2O", "B2", "E2D", "T1", "V31", "V4", "A1")
PAIRS = (("A1", "V4"), ("A1", "B2"), ("A1", "E2D"), ("A1", "V2O"), ("A1", "V31"), ("A1", "T1"), ("A1", "MAG-G"), ("A1", "MAG-B"))
LAB = dict(AV.LAB, V4="A0 = V4 SparseScore（S10 band_dil）", A1="A1 = V4 DenseScore（D100 band_dil）")
COL = dict(AV.COL, A1="#f59f00")
AG.ROOTS["A1"] = (os.path.join(WA, "closed_loop"), os.path.join(WA, "metrics", "outdoor"), "V2M")
T.GRP = GRP


def main():
    out_dir, fig_dir = os.path.join(A, "results"), os.path.join(A, "figures")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
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
                  ("degraded" if c["indoor"]["degraded"] or c["outdoor"]["degraded"] else ("meaningful on one" if c["indoor"]["meaningful"] or c["outdoor"]["meaningful"] else "neither"))}
    per_ep = {"indoor": [{"scene": r["scene"], "start": r["start"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "stall", "translations", "time_per_move")} for g in ("V4", "A1")},
                          "same_trajectory": r["V4"]["executed"] == r["A1"]["executed"], "first_divergence": next((i for i, (a, b) in enumerate(zip(r["V4"]["executed"], r["A1"]["executed"])) if a != b), None)} for r in rows_in],
              "outdoor": [{"scene": r["scene"], "start": r["start"], "split": r["split"], **{g: {k: r[g][k] for k in ("auc", "final", "cov20", "hover", "time_per_move")} for g in ("V4", "A1")},
                           "same_trajectory": r["V4"]["keys"] == r["A1"]["keys"], "first_divergence": next((i for i, (a, b) in enumerate(zip(r["V4"]["keys"], r["A1"]["keys"])) if a != b), None)} for r in rows_out]}
    res = {"groups": list(GRP), "pairs": {k: {d: {"paired": v[d]["paired"], "criteria": v[d]["criteria"]} for d in ("indoor", "outdoor")} for k, v in P.items()},
           "means": {d: P["A1-V4"][d]["means"] for d in ("indoor", "outdoor")}, "stall_mp3d": st, "decisions": dec,
           "per_scene": {d: P["A1-V4"][d]["per_scene"] for d in ("indoor", "outdoor")}, "per_episode": per_ep}
    json.dump(res, open(os.path.join(out_dir, "a_g3.json"), "w"), indent=1, default=float)
    f = T.fmt
    L = ["# Experiment A — G3 闭环（自动生成 `a_analyze_g3.py`；均值 = 场景 / 楼层均值的平均，[CI] = 场景级 bootstrap 95 %）", "",
         "| 数据集 | 组 | AUC | Final | Cov@20 | Cov@50 | 每步时间 s* |", "|---|---|---|---|---|---|---|"]
    for d, nm in (("indoor", "MP3D（10 ep × 150）"), ("outdoor", "Macarons++（30 ep × 100）")):
        M = res["means"][d]
        for g in GRP:
            L.append(f"| {nm} | {LAB[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | {f(M['cov20'][g])} | {f(M['cov50'][g])} | {M['time_per_move'][g]['mean']:.2f} |")
    L += ["", "| 配对差 | 数据集 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["pairs"].items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | {p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "判定：" + "；".join(f"{k}: {v['verdict']}" for k, v in dec.items()), "", "逐场景 / 楼层 AUC · Final（V4 = A0 → A1）：", ""]
    for d in ("indoor", "outdoor"):
        for s, v in res["per_scene"][d].items():
            L.append(f"- {s}: {v['V4']['auc']:.3f} / {v['V4'].get('final', float('nan')):.3f} → {v['A1']['auc']:.3f} / {v['A1'].get('final', float('nan')):.3f}")
    L += ["", "逐 episode（AUC / Final / Cov@20；轨迹是否相同、首次分歧步）：", ""]
    for d in ("indoor", "outdoor"):
        for e in res["per_episode"][d]:
            L.append(f"- {d} {e['scene']} s{e['start']}: A0 {e['V4']['auc']:.3f} / {e['V4']['final']:.3f} / {e['V4']['cov20']:.3f} → A1 {e['A1']['auc']:.3f} / {e['A1']['final']:.3f} / {e['A1']['cov20']:.3f}；同轨迹 {e['same_trajectory']}，首次分歧 t = {e['first_divergence']}")
    L += ["", "MP3D 中期停滞 episode 数（/ 10）：" + "，".join(f"{g} {st[g]['episodes_with_stall']}" for g in GRP)]
    open(os.path.join(out_dir, "a_g3.md"), "w").write("\n".join(L) + "\n")
    plt = AG.plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, rows, nm, xmax in ((axs[0], rows_in, "MP3D zero-shot, 10 episodes", 150), (axs[1], rows_out, "Macarons++, 30 episodes", 100)):
        for g in GRP:
            c = np.mean([r[g]["curve"] for r in rows], 0)
            ax.plot(c, color=COL[g], lw=2.4 if g in ("V4", "A1") else 1.2, ls=":" if g in AV.TASK2 else ("--" if g in ("B2", "E2D") else "-"),
                    label=f"{g}  AUC {np.mean([r[g]['auc'] for r in rows]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax); ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.set_title(nm, loc="left"); ax.legend(fontsize=7.5, loc="lower right"); T.ST.clean(ax, "both")
    fig.suptitle("Experiment A: V4 with sparse (A0 = V4) vs dense (A1) DistP1 band_dil; references dotted / dashed; measured", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "a_g3_coverage_curves.png")); plt.close(fig)
    print(json.dumps({k: v["verdict"] for k, v in dec.items()}))


if __name__ == "__main__":
    main()
