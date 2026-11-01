"""EXP6 V3 Task 2 / Experiment 2 -- analysis of the offline counterfactuals R1 (scoring region), R2 (frontier progress,
offline part) and R3 (candidate space) on the MP3D fixed states (CPU). Output results/e2_offline_summary.json, figures and
report/{E2_ROI_COUNTERFACTUAL, E2_CANDIDATE_COUNTERFACTUAL}.md and the offline section of E2_FRONTIER_EXPLORATION.md
(the closed-loop sections are written by t2_g3_analyze.py).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
T2 = os.path.dirname(EXP)
E6 = os.path.dirname(T2)
REPO = os.path.dirname(os.path.dirname(E6))
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
import ab_data as D  # noqa: E402

OUT, FIG, REP = (os.path.join(EXP, k) for k in ("results", "figures", "report"))
WR3 = os.path.join(REPO, "work_dirs", "exp6_v3", "task2", "exp2", "r3")


def hb(rows, fn):
    return D.boot(D.hier_mean(rows, fn).values())


def main():
    for p in (OUT, FIG, REP):
        os.makedirs(p, exist_ok=True)
    recs, _ = D.load_pool("cl", "indoor")
    stall = {r["sid"]: bool(r.get("stall", False)) for r in recs}
    phase = {r["sid"]: r["phase"] for r in recs}
    summary = {}
    # ------------------------------------------------------------------ R1
    r1 = json.load(open(os.path.join(OUT, "r1_region.json")))
    rows = r1["states"]
    for r in rows:
        r["stall"] = stall.get(r["sid"], False)
        r["phase"] = phase.get(r["sid"], "")
    keys = ["n5|cap_hard", "n7|cap_hard", "n9|cap_hard", "n0|cap_hard", "n5|cap_lin", "n7|cap_lin", "n9|cap_lin", "n0|cap_lin", "n5|area_lin", "n9|area_lin", "n0|area_lin"]
    L = ["# E2 — R1 评分区域 counterfactual（自动生成 `scripts/e2_analyze.py`；MP3D Pool CL 368 状态、5 评估楼层；GT 深度 t\\* 只在评估侧）", "",
         "评分 = t\\*（全网格）在区域出口 / 射程处截断，权重 min(1,(z/z0)²)（cap）或 (z/z0)²（area），新颖性相对历史 10% 采样点（hard = d_obs > eps；lin = clip(d_obs / 2eps)），体素去重；n_win 5 = 现行区域，7 / 9 = 扩大，0 = 整层（楼层框）。oracle 增益 = 整层 C_obs（复现断言 ≤ 1e-12）。均值 = 楼层均值的平均，CI = 楼层级 bootstrap。", ""]
    tab = {}
    for name, sub in (("全部", rows), ("停滞态（Experiment A 定义）", [r for r in rows if r["stall"]]), ("非停滞", [r for r in rows if not r["stall"]])):
        L += [f"## {name}（{len(sub)} 状态）", "", "| 评分（区域 \\| 权重新颖性） | 被选动作真实增益 pp [CI]（越大越好） | Top-1 | Spearman | regret pp（附录） |", "|---|---|---|---|---|"]
        tab[name] = {}
        gs = hb(sub, lambda r: r["gstar_pp"])
        for k in keys:
            reg = hb(sub, lambda r, k=k: r["metrics"][k]["regret"]); top = hb(sub, lambda r, k=k: float(r["metrics"][k]["top1"])); rho = hb(sub, lambda r, k=k: r["metrics"][k].get("rho"))
            gp = hb(sub, lambda r, k=k: r["metrics"][k]["gmax"] - r["metrics"][k]["regret"])
            tab[name][k] = {"regret": reg, "top1": top, "rho": rho, "gain_pick": gp}
            L.append(f"| {k} | {D.fmt(gp)} | {D.fmt(top)} | {D.fmt(rho) if rho else '—'} | {D.fmt(reg)} |")
        L.append(f"| Oracle 最优（上限） | {D.fmt(gs)} | 1 | — | 0 |")
        L += ["", "| 量 | n_win 5 | 7 | 9 | 整层 |", "|---|---|---|---|---|"]
        rem = {n: hb(sub, lambda r, n=n: r["remaining_in_region"][n]) for n in ("5", "7", "9")}
        bg = {n: hb(sub, lambda r, n=n: r["best_gain_in_region_share"][n]) for n in ("5", "7", "9")}
        L.append(f"| 剩余未覆盖可见表面在区域内的比例 | {D.fmt(rem['5'])} | {D.fmt(rem['7'])} | {D.fmt(rem['9'])} | 1 |")
        L.append(f"| oracle 最优候选的增益在区域内的份额 | {D.fmt(bg['5'])} | {D.fmt(bg['7'])} | {D.fmt(bg['9'])} | 1 |")
        tab[name]["remaining_in_region"] = rem; tab[name]["best_gain_share"] = bg
        L.append("")
    summary["R1"] = tab
    L += ["## 逐楼层：被选动作真实增益 pp（cap_hard；括号内 oracle 最优）", "", "| 楼层 | n | n5 | n7 | n9 | 整层 | oracle 最优 |", "|---|---|---|---|---|---|---|"]
    for s in sorted({r["scene"] for r in rows}):
        sub = [r for r in rows if r["scene"] == s]
        L.append(f"| {s} | {len(sub)} | " + " | ".join(f"{np.mean([r['metrics'][k]['gmax'] - r['metrics'][k]['regret'] for r in sub]):.3f}" for k in ("n5|cap_hard", "n7|cap_hard", "n9|cap_hard", "n0|cap_hard")) + f" | {np.mean([r['gstar_pp'] for r in sub]):.3f} |")
    open(os.path.join(REP, "E2_ROI_COUNTERFACTUAL.md"), "w").write("\n".join(L) + "\n")
    # ------------------------------------------------------------------ R3
    r3 = [json.load(open(f)) for f in glob.glob(os.path.join(WR3, "*.json"))]
    for r in r3:
        r["stall"] = stall.get(r["sid"], False); r["episode"] = f"{r['tag']}|{r['source']}"; r["phase"] = phase.get(r["sid"], "")
        r["max1_pp"], r["max2_pp"] = 100 * r["max1"], 100 * r["max2"]
        r["gain_ratio"] = r["max2"] / max(r["max1"], 1e-9)
        r["ext_better"] = float(r["max2"] > r["max1"] + 5e-4)
        r["ext_double"] = float(r["max2"] >= 2 * max(r["max1"], 1e-9))
        r["best2_unknown"] = float(bool(r["best2_cell"] and r["best2_cell"]["unknown"]))
        r["best2_visited"] = float(bool(r["best2_cell"] and r["best2_cell"]["visited"]))
    L = ["# E2 — R3 候选空间 counterfactual（自动生成；MP3D Pool CL 368 状态；GT 只在评估侧，不改在线候选生成）", "",
         "原合法集 = 环境一步合法动作（官方邻居 + GT 线段）；扩展集 = 两步可达（每个合法邻居的合法邻居，去掉当前与一步集），每个位姿的真实增益相对当前覆盖状态单独评估（帧 t+1 采样种子；不含中间观测 → 乐观）。'新格' = 位姿所在 2D 格在记录构建的 frontier 图中为 UNKNOWN。", "",
         "| 子集 | n | 一步 G\\* pp（中位 / 均值）| 两步 G\\* pp（中位 / 均值）| 两步 > 一步 + 0.05 pp | 两步 ≥ 2 × 一步 | 扩展位姿数 | 其中进入 UNKNOWN 格 | 两步最优在 UNKNOWN 格 | 两步最优在已访问格 |", "|---|---|---|---|---|---|---|---|---|---|"]
    t3 = {}
    for name, sub in (("全部", r3), ("停滞态", [r for r in r3 if r["stall"]]), ("非停滞", [r for r in r3 if not r["stall"]])):
        m1 = np.array([r["max1_pp"] for r in sub]); m2 = np.array([r["max2_pp"] for r in sub])
        t3[name] = {"n": len(sub), "max1_med": float(np.median(m1)), "max1_mean": float(m1.mean()), "max2_med": float(np.median(m2)), "max2_mean": float(m2.mean()),
                    "ext_better": hb(sub, lambda r: r["ext_better"]), "ext_double": hb(sub, lambda r: r["ext_double"]), "n_ext": float(np.mean([r["n_ext2"] for r in sub])),
                    "n_ext_new": float(np.mean([r["n_ext_new_cells"] for r in sub])), "best2_unknown": float(np.mean([r["best2_unknown"] for r in sub])), "best2_visited": float(np.mean([r["best2_visited"] for r in sub]))}
        x = t3[name]
        L.append(f"| {name} | {x['n']} | {x['max1_med']:.3f} / {x['max1_mean']:.3f} | {x['max2_med']:.3f} / {x['max2_mean']:.3f} | {D.fmt(x['ext_better'])} | {D.fmt(x['ext_double'])} | {x['n_ext']:.1f} | {x['n_ext_new']:.1f} | {x['best2_unknown']:.2f} | {x['best2_visited']:.2f} |")
    summary["R3"] = t3
    L += ["", "## 逐楼层（一步 / 两步 G\\* 中位 pp；停滞态数）", "", "| 楼层 | n | 一步 | 两步 | 停滞态 |", "|---|---|---|---|---|"]
    for s in sorted({r["scene"] for r in r3}):
        sub = [r for r in r3 if r["scene"] == s]
        L.append(f"| {s} | {len(sub)} | {np.median([r['max1_pp'] for r in sub]):.3f} | {np.median([r['max2_pp'] for r in sub]):.3f} | {sum(r['stall'] for r in sub)} |")
    open(os.path.join(REP, "E2_CANDIDATE_COUNTERFACTUAL.md"), "w").write("\n".join(L) + "\n")
    # ------------------------------------------------------------------ R2 offline
    r2 = json.load(open(os.path.join(OUT, "r2_offline.json")))
    S = r2["summary"]
    L = ["# E2 — R2 Frontier-aware exploration（自动生成；离线部分：MP3D Pool CL 368 状态；闭环部分见 `E2_G3_RESULTS.md`）", "",
         f"frontier 图只用真实记录（`common/t2_frontier.py`）；FP ∈ {{−1, 0, +1}}；bonus λ = {S['lambda']}，trigger θ = {S['theta']}（预登记，VALSEL 上 B2 的分数分布）。oracle 只用于评估。", "",
         "| 子集 | n | 触发率 | 选点 ≠ S0（bonus / trigger）| 被执行动作的真实增益 pp S0 / bonus / trigger（越大越好）| 一步 regret pp S0 / bonus / trigger（附录）| 选到 FP > 0 动作的比例 S0 / bonus / trigger | 到最近可达 frontier 的路径距离（中位格）| 剩余表面在 frontier 3 m 内的份额 | G\\* 中位 pp |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, k in (("全部", "all"), ("停滞态", "stall"), ("非停滞", "non_stall"), ("触发态", "triggered")):
        if k not in S["tables"]:
            continue
        x = S["tables"][k]
        L.append(f"| {name} | {x['n']} | {(S['n_triggered'] / S['n_states']) if k == 'all' else '—'} | {x['pick_differs_from_S0']['bonus']:.2f} / {x['pick_differs_from_S0']['trigger']:.2f} | "
                 f"{x['gain_pick']['S0']['mean']:.3f} / {x['gain_pick']['bonus']['mean']:.3f} / {x['gain_pick']['trigger']['mean']:.3f} | {x['regret']['S0']['mean']:.3f} / {x['regret']['bonus']['mean']:.3f} / {x['regret']['trigger']['mean']:.3f} | "
                 f"{x['fp_pick_pos']['S0']:.2f} / {x['fp_pick_pos']['bonus']:.2f} / {x['fp_pick_pos']['trigger']:.2f} | {x['dist_to_frontier_median']:.0f} | {x['remaining_share_near_frontier']:.3f} | {x['gstar_pp_median']:.3f} |")
    L += ["", "解读：frontier 导向的动作按构造牺牲当下的一步增益（bonus / trigger 的 regret 高于 S0）；它是否换来长期覆盖只能由闭环回答。'剩余表面在 frontier 3 m 内的份额' 低说明最近的 frontier 格附近往往没有多少未覆盖表面（frontier 图的 UNKNOWN 格多在走廊 / 边缘）。"]
    open(os.path.join(REP, "E2_FRONTIER_EXPLORATION.md"), "w").write("\n".join(L) + "\n")
    summary["R2_offline"] = S
    json.dump(summary, open(os.path.join(OUT, "e2_offline_summary.json"), "w"), indent=1, default=float)
    # ------------------------------------------------------------------ figure: R1 regret vs region, R3 max1 vs max2
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axs[0]
    for name, col in (("全部", "#1c5cab"), ("停滞态（Experiment A 定义）", "#d62728")):
        vals = [tab[name][f"n{n}|cap_hard"]["regret"] for n in (5, 7, 9, 0)]
        ax.errorbar([5, 7, 9, 11], [v["mean"] for v in vals], yerr=[[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]], marker="o", color=col, label=name.split("（")[0], capsize=3)
    ax.set_xticks([5, 7, 9, 11]); ax.set_xticklabels(["n_win 5", "7", "9", "whole level"]); ax.set_ylabel("GT-depth one-step regret (pp; lower = better)"); ax.set_title("R1: scoring region (GT depth, hard novelty)", loc="left"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[1]
    m1 = np.array([r["max1_pp"] for r in r3]); m2 = np.array([r["max2_pp"] for r in r3]); st = np.array([r["stall"] for r in r3])
    ax.scatter(m1[~st], m2[~st], s=8, alpha=0.5, color="#1c5cab", label="non-stall"); ax.scatter(m1[st], m2[st], s=14, color="#d62728", label="stall")
    lim = max(m2.max(), 1e-3); ax.plot([0, lim], [0, lim], "k--", lw=0.8); ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("max true gain, 1-step legal set (pp)"); ax.set_ylabel("max true gain, 2-step reachable set (pp)"); ax.set_title("R3: candidate space", loc="left"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[2]
    names = ["all", "stall", "non_stall", "triggered"]; xs = np.arange(len(names))
    for k, (m, col) in enumerate((("S0", "#1c5cab"), ("bonus", "#eb6834"), ("trigger", "#8e44ad"))):
        ax.bar(xs + (k - 1) * 0.27, [S["tables"][n]["regret"][m]["mean"] for n in names], 0.27, color=col, label=m)
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylabel("one-step regret (pp)"); ax.set_title("R2 offline: frontier term vs S0 (B2 scores)", loc="left"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e2_offline.png"), dpi=130); plt.close(fig)
    print(json.dumps({"R1_all": {k: round(v["regret"]["mean"], 3) for k, v in tab["全部"].items() if "regret" in v}, "R3": {k: (round(v["max1_med"], 3), round(v["max2_med"], 3)) for k, v in t3.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
