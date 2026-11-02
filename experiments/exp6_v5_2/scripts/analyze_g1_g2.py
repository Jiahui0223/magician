"""EXP6 V5.2 -- G1 / G2 analysis (seen2scene env, CPU).

G2: the V5.2 rows (work_dirs/exp6_v5_2/eval_fixed/<set>_<tag>) are merged by sid with the historical fixed-state rows
(B2 / E2-D100 from E2, V4 from exp6_v4, V5 / V4C from exp6_v5) -- same states, candidates and choice rule
(exp6_v4/scripts/analyze_g12_v4.rows_to_recs -> ab_load.pick_of, incl. prefer-unvisited on MP3D); V5.2 enters as two
arms V52L0 (lambda 0) and V52L01 (lambda 0.1) with one score per candidate. Main metrics: true one-step coverage gain of
the pick (pp), Top-1, Spearman to the true gains, by phase, scene-level bootstrap CI of paired differences; share of
states whose pick differs from V5 / V4C / B2; lambda flips (L0 vs L01); potential share of the lambda = 0.1 pick.
G1: first-hit metrics aggregated from the rows (vs oracle t*), multi-surface belief metrics from
reports/belief_metrics_eval.json (eval_belief_metrics_v52.py), S-c monitor from the training run.
Output reports/g12_v52.{json,md}. Usage: python analyze_g1_g2.py [--tag s_final]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v52_common as VC  # noqa: E402

REPO = VC.REPO
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v4", "scripts"))
import analyze_g12_v4 as A4  # noqa: E402

EA = A4.EA
W5 = os.path.join(REPO, "work_dirs", "exp6_v5", "eval_fixed")
MODELS = ("B2", "E2D", "V4", "V4C", "V5", "V52L0", "V52L01")
LAB = dict(A4.LAB, V4="V4@40k", V4C="V4-Continue12k", V5="V5-Warm12k", V52L0="V5.2 λ=0 (visible)", V52L01="V5.2 λ=0.1 (visible + potential)")
COL = dict(A4.COL, V4C="#495057", V5="#7048e8", V52L0="#2f9e44", V52L01="#e8590c")
PAIRS = (("V52L0", "V5"), ("V52L01", "V5"), ("V52L01", "V52L0"), ("V52L0", "V4C"), ("V52L0", "B2"), ("V52L0", "E2D"), ("V52L01", "B2"), ("V52L01", "E2D"))
DUMMY_PRED = {**{f"sae_{g}": 0.0 for g in ("all", "unknown", "new", "known", "unknown_new")}, **{f"n_{g}": 0 for g in ("all", "unknown", "new", "known", "unknown_new")},
              "hit_agree": 0, "n_rays": 1, "rel_std_sum": 0.0, "n_allhit": 0, "hit_dis": 0, "rho_mean": None}


def load(sset, tag):
    base = {r["sid"]: r for r in A4.load(sset, "s_final_v4", "s_final_v31")}
    hist = [A4.read_rows(os.path.join(W5, f"{sset}_s_final_{a}")) for a in ("v5", "v4c")]
    v52 = A4.read_rows(os.path.join(VC.W52, "eval_fixed", f"{sset}_{tag}"))
    rows, extra = [], {}
    for sid, r in base.items():
        if sid not in v52 or not all(sid in h for h in hist):
            continue
        for h in hist:
            n = h[sid]
            assert n["scores"]["B2"] == r["scores"]["B2"], ("B2 replay differs", sid)
            for m in ("V5", "V4C"):
                if m in n["scores"]:
                    r["scores"][m], r["pred"][m] = n["scores"][m], n["pred"][m]
        x = v52[sid]
        r["scores"]["V52L0"] = [[s] for s in x["v52"]["score_l0"]]
        r["scores"]["V52L01"] = [[s] for s in x["v52"]["score_l0.1"]]
        r["pred"]["V52L0"] = r["pred"]["V52L01"] = DUMMY_PRED
        extra[sid] = x
        rows.append(r)
    return rows, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s_final")
    a = ap.parse_args()
    A4.MODELS, A4.PAIRS, A4.LAB = MODELS, PAIRS, LAB
    EA.MODELS, EA.LAB, EA.COL, EA.PAIRS = MODELS, LAB, COL, PAIRS
    EA.OUT, EA.FIG = os.path.join(VC.E52, "reports"), os.path.join(VC.E52, "figures")
    EA.ST.setup()
    S, X, G1 = {}, {}, {}
    for d in ("valsel", "dev3"):
        rows, extra = load(d, a.tag)
        recs = A4.rows_to_recs(d, rows)
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
        diff = {f"{m}_vs_{b}": float(np.mean([r[f"{m}_pick"] != r[f"{b}_pick"] for r in recs])) for m in ("V52L0", "V52L01") for b in ("V5", "V4C", "B2", "V4")}
        flips = [r["V52L0_pick"] != r["V52L01_pick"] for r in recs]
        share, frac_free, frac_known = [], [], []
        for r in recs:
            x = extra[r["sid"]]
            j = r["V52L01_pick"]
            share.append(x["v52"]["parts_l0.1"]["pot_share"][j])
            dg = x["v52"]["diag"][j]
            frac_free.append(dg["mass_free"] / max(dg["mass_dom"], 1e-9)); frac_known.append(dg["mass_known"] / max(dg["mass_dom"], 1e-9))
        # gains of the lambda flips: does the potential change picks for the better?
        fl = [r for r in recs if r["V52L0_pick"] != r["V52L01_pick"]]
        X[d] = {"pick_differs": diff, "lambda_flip_rate": float(np.mean(flips)), "n_flips": len(fl),
                "flip_gain_delta_pp": float(np.mean([r["V52L01_gain"] - r["V52L0_gain"] for r in fl])) if fl else None,
                "flip_better": int(sum(r["V52L01_gain"] > r["V52L0_gain"] for r in fl)), "flip_worse": int(sum(r["V52L01_gain"] < r["V52L0_gain"] for r in fl)),
                "pot_share_of_pick_mean": float(np.mean(share)), "pot_share_of_pick_p90": float(np.quantile(share, 0.9)),
                "visible_mass_on_D100_free_frac_mean": float(np.mean(frac_free)), "visible_mass_on_S10_known_frac_mean": float(np.mean(frac_known)),
                "score_s_per_state_mean": float(np.mean([extra[r["sid"]]["timing"]["v52_score_s"] for r in recs])),
                "candidates_per_state_mean": float(np.mean([extra[r["sid"]]["n_legal"] for r in recs]))}
        fh = {}
        for x in extra.values():
            fh = VC.merge_fh(fh, x["firsthit"])
        G1[d] = VC.fh_summary(fh)
    out = {"tag": a.tag, "models": MODELS, "summary": S, "v52": X, "g1_firsthit": G1}
    bp = os.path.join(VC.E52, "reports", "belief_metrics_eval.json")
    if os.path.exists(bp):
        out["g1_belief"] = json.load(open(bp))["metrics"]
    sc = os.path.join(VC.W52, "runs", "v52", "eval.jsonl")
    if os.path.exists(sc):
        out["sc_monitor"] = [json.loads(x) for x in open(sc)]
    json.dump(out, open(os.path.join(VC.E52, "reports", "g12_v52.json"), "w"), indent=1, default=float)
    f3 = EA.f3
    L = [f"# V5.2 G1 / G2（固定状态 365 态，自动生成 `scripts/analyze_g1_g2.py`）", "",
         "主指标：被选候选的真实一步覆盖增益（pp，越大越好）、Top-1、Spearman（候选分数 vs 真实增益）；同状态 / 同候选 / 同选点规则；真实增益只作指标。", ""]
    for d, nm in (("valsel", "MP3D VALSEL"), ("dev3", "Macarons++ Dev-3（B0 + B2）")):
        L += [f"## {nm}（{S[d]['n']} 态）", "", "| 阶段 | 状态 | 指标 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
        for sub in ("early", "middle", "late", "all"):
            for met in ("gain", "oracle", "top1", "rho"):
                L.append(f"| {sub} | {S[d][sub]['n_states']} | {met} | " + " | ".join(f3(S[d][sub]["means"].get(f"{m}_{met}")) for m in MODELS) + " |")
        L += ["", "| 配对差（全部状态，场景级 CI） | " + " | ".join(f"{x}−{y}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("gain", "top1"):
            L.append(f"| {met} | " + " | ".join(f"{f3(S[d]['all']['paired'][f'{x}-{y}|{met}'])}（低 {S[d]['all']['paired'][f'{x}-{y}|{met}']['lower']} / 高 {S[d]['all']['paired'][f'{x}-{y}|{met}']['higher']}）" for x, y in PAIRS) + " |")
        x = X[d]
        L += ["", f"- 选择不同的比例：" + "，".join(f"{k} {v:.3f}" for k, v in x["pick_differs"].items()),
              f"- λ 翻转（λ=0 与 λ=0.1 选点不同）：{x['lambda_flip_rate']:.3f}（{x['n_flips']} 态；翻转后增益更高 {x['flip_better']} / 更低 {x['flip_worse']}，平均 Δ {f3(x['flip_gain_delta_pp'])} pp）",
              f"- λ=0.1 被选候选中潜力项占总分比例：均值 {x['pot_share_of_pick_mean']:.3f}，p90 {x['pot_share_of_pick_p90']:.3f}",
              f"- 可见首中质量落在 D100 已确认 FREE 的比例（被 q_valid 去掉）：{x['visible_mass_on_D100_free_frac_mean']:.3f}；落在 S10 已知 BAND 的比例：{x['visible_mass_on_S10_known_frac_mean']:.3f}",
              f"- 打分耗时：每状态 {x['score_s_per_state_mean']:.2f} s（平均 {x['candidates_per_state_mean']:.1f} 个候选）", ""]
        L += ["### G1 首中（oracle t*，候选视角全部射线）", "", "| 射线组 | 射线 | NLL | 最高区间准确率 | 命中率 | no-hit ECE | 众数误差 vox | 期望误差 vox |", "|---|---|---|---|---|---|---|---|"]
        for grp, v in G1[d].items():
            L.append(f"| {grp} | {v['rays']:,} | {v['nll']:.3f} | {v['top_acc']:.3f} | {v['hit_rate']:.3f} | {v['nohit_ece']:.3f} | {v['mode_err_vox']:.1f} | {v['exp_err_vox']:.1f} |")
        L.append("")
    open(os.path.join(VC.E52, "reports", "g12_v52.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({d: {m: {k: S[d]["all"]["means"][f"{m}_{k}"]["mean"] for k in ("gain", "top1")} for m in MODELS} for d in S}, indent=1, default=float))


if __name__ == "__main__":
    main()
