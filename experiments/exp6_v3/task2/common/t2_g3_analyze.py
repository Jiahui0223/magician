"""EXP6 V3 Task 2 -- G3 closed-loop analysis (seen2scene env, CPU): the Task 2 arms (configs/G3_ARMS.json; indoor under
<g3>/closed_loop/indoor/<arm>/V2M/<tag>, outdoor under <g3>/closed_loop/outdoor/<arm>/V2M/, metrics <g3>/metrics/outdoor/
<arm>/V2M/) against the historical groups MAG-G / MAG-B / B0 / B2 / E2D / T0 / T1 / C1 / C2 (Experiment C readers), with the
Task 3 §4 paired scene bootstrap (t34_analyze.summarize). Output exp1/results/g3_summary.json, figures, and the report
fragments exp1/report/E1_G3_RESULTS.md and exp2/report/E2_G3_RESULTS.md.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.dirname(HERE)
E6 = os.path.dirname(T2)
sys.path.insert(0, os.path.join(E6, "ablation_study", "experiment_C_gtdepth_closed_loop", "scripts"))
import c_analyze as CA  # noqa: E402

T = CA.T
G3 = CA.G3
CC = CA.CC
WG = os.path.join(CC.REPO, "work_dirs", "exp6_v3", "task2", "g3")
ARMS = json.load(open(os.path.join(T2, "configs", "G3_ARMS.json")))
ARM_DEF = dict(ARMS["formal"], **ARMS["exploratory"])
HIST9 = ("MAG-G", "MAG-B", "B0", "B2", "E2D", "T0", "T1", "C1", "C2")
LAB = dict(CA.LAB)
COL = dict(CA.COL)
for a, d in ARM_DEF.items():
    LAB[a] = f"{a} ({d['model']} + {d['scorer']}" + (f", frontier {d['frontier']}" if d.get("frontier", "none") != "none" else "") + (", exploratory)" if a in ARMS["exploratory"] else ")")
    COL[a] = {"B2_S1c": "#ff7f0e", "T1_S1c": "#9467bd", "B2_FRbonus": "#2ca02c", "B2_FRtrig": "#17becf"}.get(a, "#e377c2" if d["model"] == "T1" else "#bcbd22")
OUT1, REP1 = os.path.join(T2, "exp1_coverage_scoring", "results"), os.path.join(T2, "exp1_coverage_scoring", "report")
REP2, FIG = os.path.join(T2, "exp2_long_range", "report"), os.path.join(T2, "exp1_coverage_scoring", "figures")


def arm_indoor(arm, tag):
    return os.path.join(WG, "closed_loop", "indoor", arm, "V2M", tag)


def arm_outdoor(arm, s, k):
    base = os.path.join(WG, "closed_loop", "outdoor", arm, "V2M")
    tg = f"V2M_{s}_s{k}_r0"
    return base, tg, os.path.join(WG, "metrics", "outdoor", arm, "V2M", f"{tg}.json")


def rows_for(dataset, arms):
    groups = HIST9 + tuple(arms)
    rows = []
    for ep in T.A.episodes(dataset):
        if dataset == "indoor":
            r = {"dataset": "indoor", "scene": ep["scene"], "start": ep["start"], "split": "mp3d_val", "tag": ep["tag"], "B0_equals_task2": True}
            for g in HIST9:
                r[g] = CA._row_indoor(CA.indoor_dir(g, ep["tag"]), g)
            for a in arms:
                r[a] = CA._row_indoor(arm_indoor(a, ep["tag"]), "ARM")       # plain reader (no Experiment C diagnostics)
                d = arm_indoor(a, ep["tag"])
                t2 = [json.loads(x) for x in open(os.path.join(d, "t2_steps.jsonl"))] if os.path.exists(os.path.join(d, "t2_steps.jsonl")) else []
                r[a]["t2_score_s"] = float(np.mean([x["t2_score_s"] for x in t2])) if t2 else 0.0
                r[a]["triggered"] = int(sum(1 for x in t2 if (x.get("frontier") or {}).get("triggered"))) if t2 else 0
        else:
            r = {"dataset": "outdoor", "scene": ep["scene"], "start": ep["start"], "split": ep["split"], "tag": ep["tag"], "B0_equals_task2": True}
            for g in HIST9:
                r[g] = CA._row_outdoor(g, ep["scene"], ep["start"])
            for a in arms:
                base, tg, mp = arm_outdoor(a, ep["scene"], ep["start"])
                orig_paths = CA.outdoor_paths
                CA.outdoor_paths = lambda g, s_, k_, triple=(base, tg, mp): triple if g == "ARM" else orig_paths(g, s_, k_)
                try:
                    r[a] = CA._row_outdoor("ARM", ep["scene"], ep["start"])  # plain reader
                finally:
                    CA.outdoor_paths = orig_paths
                plp = os.path.join(base, "planner_logs", f"t2_{tg}.jsonl")
                t2 = [json.loads(x) for x in open(plp)] if os.path.exists(plp) else []
                r[a]["t2_score_s"] = float(np.mean([x["t2_score_s"] for x in t2])) if t2 else 0.0
        rows.append(r)
    return rows, groups


def main():
    T.ST.setup()
    for p in (OUT1, REP1, REP2, FIG):
        os.makedirs(p, exist_ok=True)
    arms_in = [a for a, d in ARM_DEF.items() if d["dataset"] == "indoor" and all(os.path.exists(os.path.join(arm_indoor(a, ep["tag"]), "episode.json")) for ep in T.A.episodes("indoor"))]
    arms_out = [a for a, d in ARM_DEF.items() if d["dataset"] == "outdoor" and all(os.path.exists(arm_outdoor(a, ep["scene"], ep["start"])[2]) for ep in T.A.episodes("outdoor"))]
    res = {"arms_indoor": arms_in, "arms_outdoor": arms_out, "pairs": {}, "means": {}, "per_scene": {}, "stall": {}, "cost": {}}
    rows = {}
    for dataset, arms in (("indoor", arms_in), ("outdoor", arms_out)):
        if not arms:
            continue
        rws, groups = rows_for(dataset, arms)
        rows[dataset] = (rws, groups)
        T.GRP = groups
        pairs = []
        for a in arms:
            base = ARM_DEF[a]["model"]
            pairs += [(a, base), (a, "C1"), (a, "C2"), (a, "MAG-B")]
            if base == "T1":
                pairs.append((a, "B2"))
        for a1 in arms:
            for a2 in arms:
                if a1 < a2 and ARM_DEF[a1]["model"] == ARM_DEF[a2]["model"]:
                    pairs.append((a1, a2))
        P = {f"{hi}-{lo}": T.summarize(rws, dataset, hi, lo) for hi, lo in pairs}
        res["pairs"][dataset] = {k: {"paired": v["paired"], "criteria": v["criteria"]} for k, v in P.items()}
        res["means"][dataset] = P[list(P)[0]]["means"]
        res["per_scene"][dataset] = P[list(P)[0]]["per_scene"]
        if dataset == "indoor":
            for r in rws:
                for g in groups:
                    r[g]["stall"] = G3.stall(r[g]["curve"])
            res["stall"]["indoor"] = {g: {"episodes_with_stall": int(sum(r[g]["stall"] >= 30 for r in rws)), "mean_longest_window": float(np.mean([r[g]["stall"] for r in rws]))} for g in groups}
        res["cost"][dataset] = {a: {"wall_s_mean": float(np.mean([r[a]["wall_s"] for r in rws])), "decision_s": float(np.mean([r[a]["time_per_move"] for r in rws])), "t2_score_s": float(np.mean([r[a]["t2_score_s"] for r in rws])),
                                    "peak_gb": float(max(r[a]["peak_gb"] for r in rws)), "triggered_steps": int(sum(r[a].get("triggered", 0) for r in rws))} for a in arms}
    json.dump(res, open(os.path.join(OUT1, "g3_summary.json"), "w"), indent=1, default=float)
    f = T.fmt
    # ---- report fragments
    def main_table(dataset, groups):
        M = res["means"][dataset]
        cov_keys = ("cov10", "cov20", "cov50", "cov100") if dataset == "indoor" else ("cov10", "cov20", "cov50")
        L = ["| 组 | AUC | Final | " + " | ".join(k.replace("cov", "Cov@") for k in cov_keys) + " | 重访率 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|" + "---|" * len(cov_keys) + "---|---|---|"]
        for g in groups:
            L.append(f"| {LAB[g]} | {f(M['auc'][g])} | {f(M['final'][g])} | " + " | ".join(f(M[k][g]) for k in cov_keys) + f" | {M['revisit_pose'][g]['mean']:.3f} | {M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
        return "\n".join(L)

    def pair_table(dataset):
        L = ["| 配对 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 [CI] | 有意义 | 退化 | 逐场景 ΔAUC |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in res["pairs"][dataset].items():
            p, c = v["paired"], v["criteria"]
            L.append(f"| {k} | {f(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {f(p['final'])} | {p['final']['wins']}/{p['final']['losses']} | {f(p['cov20'])} | {'✅' if c['meaningful'] else '—'} | {'❌' if c['degraded'] else '—'} | "
                     + "，".join(f"{s} {x:+.3f}" for s, x in p["auc"]["per_scene"].items()) + " |")
        return "\n".join(L)

    def ep_table(dataset, groups):
        rws = rows[dataset][0]
        L = ["| 场景 | 起点 | " + " | ".join(f"{g} AUC" for g in groups) + " | " + " | ".join(f"{g} Final" for g in groups) + " |", "|---|---|" + "---|" * (2 * len(groups))]
        for r in rws:
            L.append(f"| {r['scene']} | {r['start']} | " + " | ".join(f"{r[g]['auc']:.3f}" for g in groups) + " | " + " | ".join(f"{r[g]['final']:.3f}" for g in groups) + " |")
        return "\n".join(L)
    hdr = "均值 = 场景（楼层）均值的平均；[CI] = 场景级 bootstrap 95%；配对按 (scene, start)；判定 = Task 3 §4（有意义 ΔAUC ≥ +0.010 且 ΔFinal ≥ +0.010 且胜 ≥ 4 场景且无场景 < −0.030；退化 ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010）。\\* 每步时间为各自运行时的共享负载记录。历史组 = 正式 G3 / Experiment C 结果（不重跑）。"
    L1 = ["# E1 — G3 闭环结果（自动生成 `common/t2_g3_analyze.py`）", "", hdr, "", f"臂（`configs/G3_ARMS.json`）：正式 = {list(ARMS['formal'])}（Macarons++，Gate 两模型通过的 S1c）；探索性 = {[a for a in ARMS['exploratory'] if ARM_DEF[a].get('frontier', 'none') == 'none']}（MP3D，Gate 未两模型通过；不作验收证据）。", ""]
    for dataset, nm in (("outdoor", "Macarons++（100 步，单调覆盖）"), ("indoor", "MP3D（150 步，C_obs）")):
        if dataset not in rows:
            continue
        e1_arms = [a for a in (arms_in if dataset == "indoor" else arms_out) if ARM_DEF[a].get("frontier", "none") == "none"]
        groups = HIST9 + tuple(e1_arms)
        L1 += [f"## {nm}", "", main_table(dataset, groups), "", "### 配对差", "", "\n".join(l for l in pair_table(dataset).split("\n") if not any(fa in l for fa in ("FRbonus", "FRtrig")) or l.startswith("|---") or l.startswith("| 配对")), "",
               "### 逐 episode", "", ep_table(dataset, ("B2", "T1", "C1", "C2") + tuple(e1_arms)), ""]
        if dataset == "indoor":
            L1 += ["MP3D 停滞 episode（≥ 30 步增长 < 0.01）/ 10：" + "，".join(f"{g} {res['stall']['indoor'][g]['episodes_with_stall']}" for g in groups), ""]
        L1 += ["成本：" + "；".join(f"{a} 墙钟 {c['wall_s_mean']:.0f} s / 条、决策 {c['decision_s']:.2f} s / 步（其中新评分 {c['t2_score_s']:.2f} s）、峰值 {c['peak_gb']:.1f} GB" for a, c in res["cost"][dataset].items() if a in e1_arms), ""]
    open(os.path.join(REP1, "E1_G3_RESULTS.md"), "w").write("\n".join(L1) + "\n")
    fr_arms = [a for a in arms_in if ARM_DEF[a].get("frontier", "none") != "none"]
    L2 = ["# E2 — G3 闭环结果（自动生成 `common/t2_g3_analyze.py`；MP3D 10 条）", "", hdr, "", f"臂：{fr_arms}（B2 + S0 + frontier；λ = 119.1 / θ = 75.3 预登记）。", ""]
    if "indoor" in rows and fr_arms:
        groups = HIST9 + tuple(fr_arms)
        L2 += [main_table("indoor", groups), "", "### 配对差", "", "\n".join(l for l in pair_table("indoor").split("\n") if any(fa in l for fa in fr_arms) or l.startswith("|---") or l.startswith("| 配对")), "",
               "### 逐 episode", "", ep_table("indoor", ("B2", "C1", "C2") + tuple(fr_arms)), "",
               "MP3D 停滞 episode / 10：" + "，".join(f"{g} {res['stall']['indoor'][g]['episodes_with_stall']}（平均最长窗口 {res['stall']['indoor'][g]['mean_longest_window']:.1f}）" for g in groups), "",
               "成本：" + "；".join(f"{a} 墙钟 {c['wall_s_mean']:.0f} s / 条、决策 {c['decision_s']:.2f} s / 步、触发步数 {c['triggered_steps']}" for a, c in res["cost"]["indoor"].items() if a in fr_arms), ""]
    open(os.path.join(REP2, "E2_G3_RESULTS.md"), "w").write("\n".join(L2) + "\n")
    # ---- curves
    plt = T.plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, dataset, xmax in ((axs[0], "indoor", 150), (axs[1], "outdoor", 100)):
        if dataset not in rows:
            ax.axis("off"); continue
        rws, groups = rows[dataset]
        for g in ("B2", "T1", "C1", "C2") + tuple(a for a in groups if a in ARM_DEF):
            c = np.mean([r[g]["curve"] for r in rws], 0)
            ax.plot(c, color=COL[g], lw=2.2 if g in ARM_DEF else 1.3, ls="--" if g in ("B2", "T1") else (":" if g in ("C1", "C2") else "-"), label=f"{LAB[g]}  AUC {np.mean([r[g]['auc'] for r in rws]):.3f} · final {c[-1]:.3f}")
        ax.set_xlim(0, xmax); ax.set_xlabel("move"); ax.set_ylabel("coverage"); ax.set_title(f"{dataset}: Task 2 arms vs B2 / T1 (dashed) and C1 / C2 (dotted)", loc="left"); ax.legend(fontsize=7.5, loc="lower right"); T.ST.clean(ax, "both")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t2_g3_curves.png"), dpi=130); plt.close(fig)
    for dataset in res["pairs"]:
        print(dataset, {k: (round(v["paired"]["auc"]["mean"], 4), v["paired"]["auc"]["wins"], round(v["paired"]["final"]["mean"], 4), v["criteria"]["meaningful"], v["criteria"]["degraded"]) for k, v in res["pairs"][dataset].items()})
    for dataset in res["means"]:
        print(dataset, "AUC/Final:", {g: (round(res["means"][dataset]["auc"][g]["mean"], 3), round(res["means"][dataset]["final"][g]["mean"], 3)) for g in res["means"][dataset]["auc"]})


if __name__ == "__main__":
    main()
