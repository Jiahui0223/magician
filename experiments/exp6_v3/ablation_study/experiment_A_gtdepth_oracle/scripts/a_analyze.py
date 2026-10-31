"""EXP6 V3 Ablation Experiment A -- GT-depth / DistP1 / oracle decomposition on the fixed states (CPU; protocol §2).

Tables per pool x dataset (scene-level bootstrap CI): ranking metrics of V2O* / V2M* (historical EXP5, sparse input),
B2 / E2D / T0 / T1 (D100 input), GTD and the random reference; paired scene differences; the regret decomposition
r_pred / r_gt / Delta and the state classes; prediction / scoring fidelity; MP3D stall states (CL) and Macarons++
high-regret states (CL); automatically picked failure cases. Output experiment_A_gtdepth_oracle/{results,tables,figures}
and report/{A_G0_VALIDATION, A_FIXED_STATE, A_FAILURE_CASES}.md
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(EXP), "shared"))
import ab_data as D  # noqa: E402

OUT = {k: os.path.join(EXP, k) for k in ("results", "tables", "figures", "report")}
SC_ORDER = ["V2O", "V2M", "B2", "E2D", "T0", "T1", "GTD"]
LABEL = {"V2O": "V2O* (sparse, hist.)", "V2M": "V2M* (= B0, sparse, hist.)", "B2": "B2 (V2-Mix, D100)", "E2D": "E2-D100", "T0": "V3-T0", "T1": "V3-T1",
         "GTD": "GT depth + DistP1", "Random": "Random"}
PAIRS = [("T1", "T0"), ("T1", "B2"), ("T0", "B2"), ("E2D", "B2"), ("T1", "E2D"), ("GTD", "B2"), ("GTD", "T1"), ("B2", "V2M")]
MODELS_DECOMP = ["B2", "E2D", "T0", "T1", "V2M"]


def mval(r, sc, key):
    return r["random"][key] if sc == "Random" else r["metrics"][sc].get(key)


def agg(recs, sc, key):
    return D.boot(D.hier_mean(recs, lambda r: (None if mval(r, sc, key) is None else float(mval(r, sc, key)))).values())


def scene_vals(recs, sc, key):
    return D.hier_mean(recs, lambda r: (None if mval(r, sc, key) is None else float(mval(r, sc, key))))


def main_table(recs, scorers):
    rows = ["| 评分器 | 状态 | Spearman(score, G) | Top-1（τ） | Top-1 严格 | Pairwise | regret pp [CI] | 归一化 regret | fallback |", "|---|---|---|---|---|---|---|---|---|"]
    for sc in scorers + ["Random"]:
        n = len(recs)
        rho = agg(recs, sc, "rho") if sc != "Random" else None
        top1 = agg(recs, sc, "top1"); st = agg(recs, sc, "top1_strict") if sc != "Random" else None
        pw = agg(recs, sc, "pairwise") if sc != "Random" else None
        reg = agg(recs, sc, "regret"); nreg = agg(recs, sc, "nregret") if sc != "Random" else None
        fb = sum(r["fallback"].get(sc, False) for r in recs) if sc != "Random" else 0
        rows.append(f"| {LABEL[sc]} | {n} | {D.fmt(rho) if rho else '—'} | {D.fmt(top1)} | {D.fmt(st) if st else '—'} | {D.fmt(pw) if pw else '—'} | {D.fmt(reg)} | {D.fmt(nreg) if nreg else '—'} | {fb} |")
    return "\n".join(rows)


def paired_table(recs, pairs, key="regret", nd=3):
    rows = [f"| 配对（{key}，a − b；regret 为负 = a 更好） | 均值 [CI] | 场景 a 更好 / b 更好 | 逐场景 |", "|---|---|---|---|"]
    for a, b in pairs:
        if any(a not in r["metrics"] or b not in r["metrics"] for r in recs):
            continue
        p = D.paired(scene_vals(recs, a, key), scene_vals(recs, b, key))
        if p is None:
            continue
        better = p["wins"] if key == "regret" else p["losses"]
        worse = p["losses"] if key == "regret" else p["wins"]
        rows.append(f"| {a} − {b} | {D.fmt(p, nd)} | {better} / {worse} | " + "，".join(f"{s} {v:+.{nd}f}" for s, v in p["per_scene"].items()) + " |")
    return "\n".join(rows)


def decomposition(recs, models):
    rows = ["| 模型 | r_pred [CI] | r_gt（GTD）[CI] | Δ = r_pred − r_gt [CI] | Δ / r_pred | 选对 | 选错且 GTD 选对（可归因预测） | 选错且 GTD 也错（评分 / 候选） |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for m in models:
        if any(m not in r["metrics"] for r in recs):
            continue
        rp = agg(recs, m, "regret"); rg = agg(recs, "GTD", "regret")
        d = D.paired(scene_vals(recs, m, "regret"), scene_vals(recs, "GTD", "regret"))
        right = D.boot(D.hier_mean(recs, lambda r: float(r["metrics"][m]["top1"])).values())
        pw_gr = D.boot(D.hier_mean(recs, lambda r: float((not r["metrics"][m]["top1"]) and r["metrics"]["GTD"]["top1"])).values())
        pw_gw = D.boot(D.hier_mean(recs, lambda r: float((not r["metrics"][m]["top1"]) and not r["metrics"]["GTD"]["top1"])).values())
        share = d["mean"] / rp["mean"] if rp and rp["mean"] > 0 else None
        out[m] = {"r_pred": rp, "r_gt": rg, "delta": d, "share": share, "right": right, "pred_attr": pw_gr, "score_attr": pw_gw}
        rows.append(f"| {LABEL[m]} | {D.fmt(rp)} | {D.fmt(rg)} | {D.fmt(d)} | {share:.2f} | {D.fmt(right)} | {D.fmt(pw_gr)} | {D.fmt(pw_gw)} |" if share is not None else
                    f"| {LABEL[m]} | {D.fmt(rp)} | {D.fmt(rg)} | {D.fmt(d)} | — | {D.fmt(right)} | {D.fmt(pw_gr)} | {D.fmt(pw_gw)} |")
    return "\n".join(rows), out


def fidelity_table(recs, models):
    rows = ["| 模型 | 预测保真度 Spearman(S_pred, S_GTD) | 评分保真度 Spearman(S_GTD, G) | pick = GTD pick | pick = oracle best（τ） | GTD pick = oracle best |", "|---|---|---|---|---|---|"]
    gf = D.boot(D.hier_mean(recs, lambda r: r["metrics"]["GTD"]["rho"]).values())
    gt = D.boot(D.hier_mean(recs, lambda r: float(r["metrics"]["GTD"]["top1"])).values())
    for m in models:
        if any(m not in r["metrics"] for r in recs):
            continue
        f = D.boot(D.hier_mean(recs, lambda r: r["fidelity"].get(m)).values())
        eq = D.boot(D.hier_mean(recs, lambda r: float(r["picks"][m] == r["picks"]["GTD"])).values())
        t1 = D.boot(D.hier_mean(recs, lambda r: float(r["metrics"][m]["top1"])).values())
        rows.append(f"| {LABEL[m]} | {D.fmt(f)} | {D.fmt(gf)} | {D.fmt(eq)} | {D.fmt(t1)} | {D.fmt(gt)} |")
    return "\n".join(rows)


def by_phase(recs, scorers, key="regret"):
    rows = ["| 评分器 | early | mid | late |", "|---|---|---|---|"]
    for sc in scorers + ["Random"]:
        cells = []
        for ph in ("early", "mid", "late"):
            sub = [r for r in recs if r["phase"] == ph]
            cells.append(D.fmt(agg(sub, sc, key)) + f" (n={len(sub)})" if sub else "—")
        rows.append(f"| {LABEL[sc]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def per_scene(recs, scorers, key="regret"):
    scenes = sorted({r["scene"] for r in recs})
    rows = ["| 场景 | n | " + " | ".join(scorers) + " |", "|---|---|" + "---|" * len(scorers)]
    for s in scenes:
        sub = [r for r in recs if r["scene"] == s]
        vals = [scene_vals(sub, sc, key).get(s) for sc in scorers]
        rows.append(f"| {s} | {len(sub)} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    return "\n".join(rows)


def stall_analysis(recs, models):
    st = [r for r in recs if r.get("stall")]
    ns = [r for r in recs if r.get("stall") is False]
    out = {"n_stall": len(st), "n_nonstall": len(ns), "n_total": len(recs)}
    if not st:
        return out, "停滞状态数 0。"
    lines = [f"- 停滞状态（来源轨迹在 [t−30, t] 覆盖增长 < 0.01，t ≥ 30）：**{len(st)} / {len(recs)}**；按来源：" +
             "，".join(f"{s} {sum(1 for r in st if r['source'] == s)}" for s in ("V2O", "V2M", "MAG-G", "MAG-B")) +
             f"；按场景：" + "，".join(f"{s} {sum(1 for r in st if r['scene'] == s)}" for s in sorted({r['scene'] for r in st})) + "。"]
    gpos = np.mean([r["gstar"] > D.TAU * 100 for r in st]); gmean = np.mean([r["gstar"] for r in st]); gmed = np.median([r["gstar"] for r in st])
    lines.append(f"- 停滞状态中存在合法正收益候选（G\\* > τ）的比例 **{gpos:.2f}**；G\\* 均值 {gmean:.3f} pp，中位数 {gmed:.3f} pp（非停滞状态 G\\* 均值 {np.mean([r['gstar'] for r in ns]):.3f} pp）。")
    out.update({"gstar_pos_frac": float(gpos), "gstar_mean": float(gmean), "gstar_median": float(gmed)})
    rows = ["| 模型 | regret 停滞 [CI] | regret 非停滞 [CI] | Top-1 停滞 | Top-1 非停滞 | 选中候选 = 区域内最优 | 停滞状态 oracle 最优候选的区域外增益占比 |", "|---|---|---|---|---|---|---|"]
    def out_share(r):
        g = r["gain"]; gi = r.get("gain_in")
        if gi is None or g.max() <= 0:
            return None
        b = int(np.argmax(g[r["cand"]])); j = r["cand"][b]
        return float(1 - gi[j] / g[j]) if g[j] > 0 else None
    osh = D.boot(D.hier_mean(st, out_share).values())
    for m in models + ["GTD"]:
        inbest = D.boot(D.hier_mean(st, lambda r: None if r.get("gain_in") is None else float(r["picks"][m] in [r["cand"][i] for i in np.flatnonzero(r["gain_in"][r["cand"]] >= r["gain_in"][r["cand"]].max() - D.TAU)])).values())
        rows.append(f"| {LABEL[m]} | {D.fmt(agg(st, m, 'regret'))} | {D.fmt(agg(ns, m, 'regret'))} | {D.fmt(agg(st, m, 'top1'))} | {D.fmt(agg(ns, m, 'top1'))} | {D.fmt(inbest)} | {D.fmt(osh)} |")
    rem = D.boot(D.hier_mean(st, lambda r: r.get("remaining_in_region_share")).values())
    lines.append(f"- 停滞状态里仍未覆盖表面落在当前规划区域内的比例（历史 a1_region_diag 的 remaining_in_region_share）：{D.fmt(rem)}。")
    out["remaining_in_region_share"] = rem; out["oracle_best_out_share"] = osh
    return out, "\n".join(lines) + "\n\n" + "\n".join(rows)


def high_regret_outdoor(recs, models=("B2", "T1"), thr=0.5):
    lines, out = [], {}
    for m in models:
        hi = [r for r in recs if r["metrics"][m]["regret"] >= thr]
        if not hi:
            lines.append(f"- {m}：无 regret ≥ {thr} pp 的状态。"); continue
        gtd_right = np.mean([r["metrics"]["GTD"]["top1"] for r in hi]); rgt = np.mean([r["metrics"]["GTD"]["regret"] for r in hi])
        ph = {p: sum(1 for r in hi if r["phase"] == p) for p in ("early", "mid", "late")}
        sc = {s: sum(1 for r in hi if r["scene"] == s) for s in sorted({r["scene"] for r in hi})}
        rel = []
        for r in hi:
            rv = r.get("relevance", {}).get("rel_frac")
            if isinstance(rv, list) and len(rv) == r["n_legal"]:
                rel.append(rv[r["picks"][m]])
        relm = float(np.mean([x for x in rel if x is not None])) if rel else None
        fb = sum(r["fallback"][m] for r in hi)
        out[m] = {"n": len(hi), "share": len(hi) / len(recs), "gtd_right": float(gtd_right), "r_gt_mean": float(rgt), "phase": ph, "scene": sc,
                  "rel_frac_pick": relm, "fallback": fb, "gstar_mean": float(np.mean([r["gstar"] for r in hi])), "regret_mean": float(np.mean([r["metrics"][m]["regret"] for r in hi]))}
        lines.append(f"- **{m}** 高 regret 状态（≥ {thr} pp）：{len(hi)} / {len(recs)}（{100 * len(hi) / len(recs):.0f}%），平均 regret {out[m]['regret_mean']:.2f} pp，平均 G\\* {out[m]['gstar_mean']:.2f} pp；"
                     f"相位 {ph}；场景 {sc}；其中 GTD 选对的比例 **{gtd_right:.2f}**（可归因预测），GTD 平均 regret {rgt:.2f} pp（评分 / 候选上限）；"
                     + (f"被选候选 GT 新表面与评估点集相关的比例 {relm:.2f}；" if relm is not None else "") + f"fallback {fb}。")
    return out, "\n".join(lines)


def failure_cases(recs, model, dataset):
    def best(r):
        g = r["gain"][r["cand"]]; return r["cand"][int(np.argmax(g))]
    def line(r, kind):
        pk, gp, bo = r["picks"][model], r["picks"]["GTD"], best(r)
        return (f"| {kind} | {r['scene']} s{r['start']} {r['source']} t={r['t']} ({r['phase']}) | {r['sid']} | {r['gstar']:.3f} | "
                f"{model}: 候选 {pk} 分 {r['scores'][model][pk]:.1f} 增益 {100 * r['gain'][pk]:.3f} pp；GTD: 候选 {gp} 增益 {100 * r['gain'][gp]:.3f} pp；oracle 最优 候选 {bo} 分（{model}）{r['scores'][model][bo]:.1f} GTD {r['gtd'][bo]:.1f} | "
                f"{r['metrics'][model]['regret']:.3f} / {r['metrics']['GTD']['regret']:.3f} |")
    rows = [f"| 类型 | 状态 | sid | G\\* pp | 选点 | regret {model} / GTD pp |", "|---|---|---|---|---|---|"]
    pf = [r for r in recs if (r["metrics"][model]["nregret"] or 0) >= 0.5 and r["metrics"]["GTD"]["top1"]]
    sf = [r for r in recs if (r["metrics"]["GTD"]["nregret"] or 0) >= 0.5]
    if pf:
        rows.append(line(max(pf, key=lambda r: r["metrics"][model]["regret"]), "预测失败（模型归一化 regret ≥ 0.5 且 GTD 选对，取 regret 最大）"))
    else:
        rows.append("| 预测失败 | 无符合规则的状态 | | | | |")
    if sf:
        rows.append(line(max(sf, key=lambda r: r["metrics"]["GTD"]["regret"]), "评分失败（GTD 归一化 regret ≥ 0.5，取最大）"))
    else:
        rows.append("| 评分失败 | 无符合规则的状态 | | | | |")
    if dataset == "indoor":
        stl = [r for r in recs if r.get("stall") and r["gstar"] > 0.5]
        if stl:
            rows.append(line(max(stl, key=lambda r: r["metrics"][model]["regret"]), "停滞状态且 G\\* > 0.5 pp，取模型 regret 最大"))
        else:
            rows.append("| 停滞 | 无 G\\* > 0.5 pp 的停滞状态 | | | | |")
    return "\n".join(rows)


def validation(all_recs, missing):
    lines = ["| 池 / 数据集 | 已加载 | 缺失 | 回归：B2 / E2D / T0 / T1 逐位（有参照的状态） | 数值等同（|Δ| ≤ 1e-8，偏差 #1） | GTD = EXP5（逐位） | fallback 选点（任一评分器） |", "|---|---|---|---|---|---|---|"]
    out = {}
    for (pool, ds), recs in all_recs.items():
        reg = {m: [r["regression"][m]["bitwise"] for r in recs if m in r["regression"]] for m in D.NEW}
        num = {m: [r["regression"][m]["bitwise"] or (r["regression"][m]["maxdiff"] is not None and r["regression"][m]["maxdiff"] <= 1e-8) for r in recs if m in r["regression"]] for m in D.NEW}
        regs = " / ".join(f"{sum(v)}/{len(v)}" for v in reg.values()) + " | " + " / ".join(f"{sum(v)}/{len(v)}" for v in num.values())
        ge = [r["gtd_eq_exp5"] for r in recs if r["gtd_eq_exp5"] is not None]
        fb = sum(any(r["fallback"].values()) for r in recs)
        out[f"{pool}/{ds}"] = {"loaded": len(recs), "missing": missing[(pool, ds)], "regression": {m: [int(sum(v)), len(v)] for m, v in reg.items()}, "numerically_equal": {m: [int(sum(v)), len(v)] for m, v in num.items()},
                               "gtd_eq": [int(sum(ge)), len(ge)], "fallback_states": fb}
        lines.append(f"| {pool} / {ds} | {len(recs)} | {missing[(pool, ds)]} | {regs} | {sum(ge)}/{len(ge)} | {fb} |")
    return out, "\n".join(lines)


def figures(all_recs):
    for (pool, ds), recs in all_recs.items():
        if not recs:
            continue
        scs = [s for s in SC_ORDER if all(s in r["metrics"] for r in recs)]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        xs = np.arange(len(scs))
        means = [agg(recs, s, "regret") for s in scs]
        ax[0].bar(xs, [m["mean"] for m in means], yerr=[[m["mean"] - m["lo"] for m in means], [m["hi"] - m["mean"] for m in means]], color="#4c72b0", capsize=3)
        ax[0].set_xticks(xs); ax[0].set_xticklabels(scs, rotation=30); ax[0].set_ylabel("regret (pp), scene mean [95% CI]"); ax[0].set_title(f"{pool} {ds}: one-step regret")
        for m, c in (("B2", "#1f77b4"), ("T1", "#8e44ad")):
            if m in scs:
                ax[1].scatter([r["metrics"]["GTD"]["regret"] for r in recs], [r["metrics"][m]["regret"] for r in recs], s=8, alpha=0.5, label=m, color=c)
        lim = max(0.1, max(r["gstar"] for r in recs))
        ax[1].plot([0, lim], [0, lim], "k--", lw=0.8); ax[1].set_xlabel("r_gt (GT depth + DistP1), pp"); ax[1].set_ylabel("r_pred, pp"); ax[1].legend(); ax[1].set_title("per state: r_pred vs r_gt")
        fig.tight_layout(); fig.savefig(os.path.join(OUT["figures"], f"a_regret_{pool}_{ds}.png"), dpi=130); plt.close(fig)


def main():
    for d in OUT.values():
        os.makedirs(d, exist_ok=True)
    all_recs, missing = {}, {}
    for pool, ds in (("cl", "indoor"), ("cl", "outdoor"), ("fx", "indoor"), ("fx", "outdoor")):
        all_recs[(pool, ds)], missing[(pool, ds)] = D.load_pool(pool, ds)
        print(pool, ds, len(all_recs[(pool, ds)]), "missing", missing[(pool, ds)], flush=True)
    # per-state CSV
    with open(os.path.join(OUT["results"], "a_states.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pool", "dataset", "sid", "scene", "start", "source", "episode", "t", "phase", "split", "n_cand", "gstar_pp", "stall", "scorer", "pick", "top1", "regret_pp", "nregret", "rho", "pairwise", "fallback"])
        for recs in all_recs.values():
            for r in recs:
                for sc, m in r["metrics"].items():
                    w.writerow([r["pool"], r["dataset"], r["sid"], r["scene"], r["start"], r["source"], r["episode"], r["t"], r["phase"], r["split"], len(r["cand"]), f"{r['gstar']:.5f}", r.get("stall"), sc, r["picks"][sc], int(m["top1"]), f"{m['regret']:.5f}", "" if m["nregret"] is None else f"{m['nregret']:.4f}", "" if m["rho"] is None else f"{m['rho']:.4f}", "" if m["pairwise"] is None else f"{m['pairwise']:.4f}", int(r["fallback"][sc])])
    summary = {}
    val, val_md = validation(all_recs, missing)
    summary["validation"] = val
    md = ["# Experiment A — 固定状态结果（自动生成 `scripts/a_analyze.py`；协议 v1.0 §2）", "",
          "历史列 V2O\\* / V2M\\*：EXP5 A1 的稀疏输入模型（`historical EXP5, not new V3 result`；V2M\\* = B0 配置）。B2 / E2-D100 / V3-T0 / V3-T1：本轮 D100 输入、同状态、同候选、同噪声。GTD：真实深度经同一 DistP1（K=1）。所有均值为场景均值的平均，CI = 场景级 bootstrap 95%（MP3D 5 楼层，Macarons++ 6 场景）。regret 单位 pp（百分点覆盖率）。", ""]
    for (pool, ds), recs in all_recs.items():
        if not recs:
            continue
        scs = [s for s in SC_ORDER if all(s in r["metrics"] for r in recs)]
        models = [m for m in MODELS_DECOMP if m in scs]
        title = {("cl", "indoor"): "Pool CL · MP3D（闭环状态，5 楼层）", ("cl", "outdoor"): "Pool CL · Macarons++（闭环状态，Test-3 + Dev-3）",
                 ("fx", "indoor"): "Pool FX · MP3D VALSEL（EXP6 G2 状态）", ("fx", "outdoor"): "Pool FX · Macarons++ Dev-3 B2 层（EXP6 G2 状态）"}[(pool, ds)]
        ph_note = "相位 = EXP5 t_bin" if pool == "cl" else "相位 = EXP6 phase_of"
        md += [f"## {title}（{len(recs)} 状态；{ph_note}）", "", "### A1 / A2 / A3 排序指标", main_table(recs, scs), "",
               "### 配对差（regret，场景配对）", paired_table(recs, PAIRS), "", "### 配对差（Top-1）", paired_table(recs, PAIRS, "top1"), "",
               "### 拆分 r_pred / r_gt / Δ", decomposition(recs, models)[0], "", "### 保真度与一致率", fidelity_table(recs, models), "",
               "### 按相位的 regret", by_phase(recs, scs), "", "### 逐场景 regret", per_scene(recs, scs), ""]
        summary[f"{pool}/{ds}"] = {"n": len(recs), "regret": {s: agg(recs, s, "regret") for s in scs + ["Random"]}, "top1": {s: agg(recs, s, "top1") for s in scs + ["Random"]},
                                   "rho": {s: agg(recs, s, "rho") for s in scs}, "decomp": decomposition(recs, models)[1],
                                   "paired_regret": {f"{a}-{b}": D.paired(scene_vals(recs, a, "regret"), scene_vals(recs, b, "regret")) for a, b in PAIRS if a in scs and b in scs},
                                   "phase": {ph: {s: agg([r for r in recs if r["phase"] == ph], s, "regret") for s in scs} for ph in ("early", "mid", "late")}}
        if pool == "cl" and ds == "outdoor":
            for split, scenes in (("Test-3", D.TEST3), ("Dev-3", D.DEV3)):
                sub = [r for r in recs if r["scene"] in scenes]
                md += [f"### {split}（{len(sub)} 状态，3 场景均值，无 CI）", main_table(sub, scs), ""]
                summary[f"{pool}/{ds}"][split] = {"regret": {s: agg(sub, s, "regret") for s in scs}}
    # diagnostics
    cl_in, cl_out = all_recs[("cl", "indoor")], all_recs[("cl", "outdoor")]
    if cl_in:
        so, stxt = stall_analysis(cl_in, ["B2", "E2D", "T0", "T1", "V2M"])
        md += ["## MP3D 中期停滞分析（Pool CL，预先定义）", stxt, ""]; summary["stall"] = so
    if cl_out:
        ho, htxt = high_regret_outdoor(cl_out)
        md += ["## Macarons++ 高 regret 状态分析（Pool CL，预先定义：r_pred ≥ 0.5 pp）", htxt, ""]; summary["high_regret"] = ho
    open(os.path.join(OUT["report"], "A_FIXED_STATE.md"), "w").write("\n".join(md))
    # failure cases
    fc = ["# Experiment A — 失败案例（自动规则，协议 §2；不渲染，只列状态与分数）", ""]
    for (pool, ds), recs in all_recs.items():
        if not recs:
            continue
        for m in ("T1", "B2"):
            if all(m in r["metrics"] for r in recs):
                fc += [f"## {pool} / {ds} — {m}", failure_cases(recs, m, ds), ""]
    open(os.path.join(OUT["report"], "A_FAILURE_CASES.md"), "w").write("\n".join(fc))
    open(os.path.join(OUT["report"], "A_G0_VALIDATION.md"), "w").write("\n".join([
        "# Experiment A — 校验（协议 §2 必过检查，自动生成）", "", val_md, "",
        "- 回归 = 新重放的 K=4 `distp1_k` 与 EXP6 `eval_fixed` 行（B2 / T0 / T1）、E2 行（E2D = D100）逐位相同；Pool CL 的参照 = Dev-3 B0 层 150 个状态，Pool FX = 全部状态。",
        "- GTD = EXP5：Pool CL 每个状态由 `ab_score.py` 在稀疏几何上重算的 GT 深度 DistP1 与 EXP5 `v2.json` 逐位相同（Pool FX 无历史值）。",
        "- `legal_idx` 与 `oracle.npz` 的一致性在打分时断言；候选集 `cand` 来自清单；选点规则 = 线上规则（`v2nv_planner.nv_choice` / `m_score.choose`，含 fallback）。",
        "- GT（oracle.npz）在该状态全部模型分数算完之后读取（`ab_score.run_episode` 的代码顺序）。"]))
    json.dump(summary, open(os.path.join(OUT["results"], "a_summary.json"), "w"), indent=1, default=float)
    figures(all_recs)
    print("done")


if __name__ == "__main__":
    main()
