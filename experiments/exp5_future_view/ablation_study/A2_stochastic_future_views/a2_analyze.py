"""EXP5 Ablation A2 -- stochastic future view hypotheses: per-sample quality, diversity, influence of the samples on the
choice, K ablation (seen2scene env, CPU). Protocol §4.
Output A2_stochastic_future_views/{results,tables}/
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ABL, "common"))
sys.path.insert(0, os.path.join(ABL, "A1_prediction_scoring"))
import ab_common as A  # noqa: E402
import ab_load as AL  # noqa: E402
import a1_analyze as A1  # noqa: E402
import t13_analyze as T13  # noqa: E402

import numpy as np  # noqa: E402

RES, TAB = os.path.join(HERE, "results"), os.path.join(HERE, "tables")
TAU = A1.TAU
MODELS = ("V2O", "V2M")
RT_PER_M = {"indoor": 7.0, "outdoor": 1.0}       # MP3D runtime units per metre (S_RT); Macarons++ stays in scene units
UNIT = {"indoor": "m", "outdoor": "scene units"}


def nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def regret_of(g, C, pk):
    gc = g[C]
    loc = {int(c): i for i, c in enumerate(C)}
    gp = gc[loc[pk]]
    return float(gc.max() - gp), float(gp >= gc.max() - TAU)


def state_model_row(r, dataset, m):
    C, g = r["C"], r["gain"]
    sk = r["distp1_k"][m]                                    # [n_legal, 4]
    K = sk.shape[1]
    pk4 = r["picks"][m]
    reg4, top4 = regret_of(g, C, pk4)
    k1 = [AL.pick_of(dataset, r, sk[:, k], r["rec"], m)[0] for k in range(K)]
    loo = [AL.pick_of(dataset, r, np.delete(sk, k, axis=1).mean(1), r["rec"], m)[0] for k in range(K)]
    k1r = [regret_of(g, C, p) for p in k1]
    s_pk = sk[pk4]
    dom = float(s_pk.max() / s_pk.sum()) if s_pk.sum() > 0 else None
    cm = [r["cand"][m][int(c)] for c in C]
    per = [p for x in cm for p in x["per_sample"]]
    cv = [float(np.std(sk[c]) / np.mean(sk[c])) for c in C if np.mean(sk[c]) > 0]
    err = [abs(np.log1p(sk[c].mean()) - np.log1p(r["scores"]["GTD"][c])) for c in C]
    sd = [float(np.std(sk[c])) for c in C]
    u = RT_PER_M[dataset]
    row = {"dataset": dataset, "sid": r["sid"], "scene": r["scene"], "split": r["split"], "start": r["start"], "t": r["t"],
           "src": r["src"], "model": m, "n_cand": len(C),
           # quality (samples x candidates of the candidate set)
           "brier": nanmean([x.get("brier") for x in cm]), "fair_es": nanmean([x.get("fair_es") for x in cm]),
           "false_surface_sample": nanmean([p["false_surface"] for p in per]),
           "missed_surface_sample": nanmean([p["missed_surface"] for p in per]),
           "precision_new_sample": nanmean([p["precision_new"] for p in per]),
           "depth_rel_err": nanmean([p["depth_rel_err"] for p in per]),
           "chamfer_to_gt_new": nanmean([c / u for x in cm for c in x["chamfer_to_gt_new_rt"] if c is not None]),
           # diversity
           "hit_disagree_rays": nanmean([x.get("hit_disagree_rays") for x in cm]),
           "hit_disagree_pair": nanmean([x.get("hit_disagree_pair") for x in cm]),
           "cond_depth_div": nanmean([x["div_cond_depth_rt"] / u if x["div_cond_depth_rt"] is not None else None for x in cm]),
           "surface_chamfer_div": nanmean([x["div_new_surface_chamfer_rt"] / u if x["div_new_surface_chamfer_rt"] is not None else None for x in cm]),
           "new_vox_jaccard_div": nanmean([x["div_new_vox_jaccard"] for x in cm]),
           "spread_vox": nanmean([x.get("spread_vox") for x in cm]),
           "score_cv": nanmean(cv), "score_sd": nanmean(sd),
           "n_new_rays_mean": nanmean([p["new_rays"] for p in per]),
           # influence on the choice
           "k1_agree": float(np.mean([p == pk4 for p in k1])), "k1_distinct": len(set(k1)),
           "loo_pivotal": float(any(p != pk4 for p in loo)), "loo_n_pivotal": int(sum(p != pk4 for p in loo)),
           "pick_dominance": dom, "regret_k4": reg4, "top1_k4": top4,
           "regret_k1_mean": float(np.mean([x[0] for x in k1r])), "top1_k1_mean": float(np.mean([x[1] for x in k1r])),
           "nregret_k4": reg4 / g[C].max() if g[C].max() > TAU else None,
           "pred_err_logratio": nanmean(err)}
    # does the pick depend on one high sample? dominant sample = argmax over the picked candidate's samples
    if dom is not None:
        kd = int(np.argmax(s_pk))
        row["pick_dominant_sample_is_pivotal"] = float(loo[kd] != pk4)
        pc = r["cand"][m][pk4]["per_sample"]
        row["pick_dom_precision_new"] = pc[kd]["precision_new"]
        row["pick_other_precision_new"] = nanmean([pc[k]["precision_new"] for k in range(K) if k != kd])
    return row


def kabl_rows(r, dataset):
    p = os.path.join(A.state_dir(dataset, r["sid"]), "kabl.npz")
    if not r["primary_t"] or not os.path.exists(p):           # Table B2 on the brief's times only (DEVIATIONS #5)
        return []
    z = np.load(p)
    tmj = json.load(open(os.path.join(A.state_dir(dataset, r["sid"]), "kabl_timing.json")))
    out = []
    n = len(r["legal_idx"])
    for m in MODELS:
        for K in A.K_ABL:
            picks, regs, tops = [], [], []
            for rr in range(A.N_SEEDSETS):
                s = z[f"{m}_r{rr}"][:, :K].mean(1)
                pk = AL.pick_of(dataset, r, s, r["rec"], m)[0]
                rg, tp = regret_of(r["gain"], r["C"], pk)
                picks.append(pk)
                regs.append(rg)
                tops.append(tp)
            pairs = [float(picks[i] == picks[j]) for i in range(len(picks)) for j in range(i + 1, len(picks))]
            tm = tmj[f"{m}_K{K}"]
            out.append({"dataset": dataset, "sid": r["sid"], "scene": r["scene"], "split": r["split"], "start": r["start"],
                        "t": r["t"], "src": r["src"], "model": m, "K": K, "regret": float(np.mean(regs)),
                        "top1": float(np.mean(tops)), "stability": float(np.mean(pairs)),
                        "time_per_cand_s": float(sum(tm[k] for k in ("projection_s", "cond_s", "sampling_s", "decode_score_s")) / n)})
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    summary = {}
    for dataset in ("indoor", "outdoor"):
        sts = AL.load_states(dataset)
        rows = [state_model_row(r, dataset, m) for r in sts for m in MODELS]
        krow = [x for r in sts for x in kabl_rows(r, dataset)]
        A1.write_csv(os.path.join(RES, f"a2_states_{dataset}.csv"), rows)
        if krow:
            A1.write_csv(os.path.join(RES, f"a2_kabl_{dataset}.csv"), krow)
        keys = [k for k in rows[0] if k not in ("dataset", "sid", "scene", "split", "start", "t", "src", "model")]
        S = {"n_states": len(sts), "unit_distance": UNIT[dataset], "models": {}, "paired_V2M_minus_V2O": {}, "kabl": {},
             "pooled": {}}
        for m in MODELS:
            sub = [x for x in rows if x["model"] == m]
            S["models"][m] = {k: AL.boot(AL.hier_mean(sub, k)) for k in keys}
            # pooled (candidate-level) relations: diversity vs quality, score spread vs prediction error
            cand = []
            for r in sts:
                for c in r["C"]:
                    x = r["cand"][m][int(c)]
                    ch = [v for v in x["chamfer_to_gt_new_rt"] if v is not None]
                    cand.append({"div": x["div_new_surface_chamfer_rt"], "err": float(np.mean(ch)) if ch else None,
                                 "hd": x.get("hit_disagree_rays"), "brier": x.get("brier"), "es": x.get("fair_es"),
                                 "sd": float(np.std(r["distp1_k"][m][int(c)])),
                                 "perr": abs(r["scores"][m][int(c)] - r["scores"]["GTD"][int(c)])})
            pair = lambda a, b: [(x[a], x[b]) for x in cand if x[a] is not None and x[b] is not None]  # noqa: E731
            S["pooled"][m] = {
                "rho_surface_div_vs_chamfer_err": T13.spearman(*zip(*pair("div", "err"))) if pair("div", "err") else None,
                "rho_hit_disagree_vs_brier": T13.spearman(*zip(*pair("hd", "brier"))) if pair("hd", "brier") else None,
                "rho_hit_disagree_vs_es": T13.spearman(*zip(*pair("hd", "es"))) if pair("hd", "es") else None,
                "rho_score_sd_vs_abs_pred_minus_gtd": T13.spearman(*zip(*pair("sd", "perr"))),
                "n_candidates": len(cand)}
            # high-score failures: pick with normalised regret >= 0.5
            fail = [x for x in sub if x["nregret_k4"] is not None and x["nregret_k4"] >= 0.5]
            S["models"][m]["high_score_failures"] = {
                "states": len(fail), "of": len(sub),
                "dominant_sample_pivotal_rate": nanmean([x.get("pick_dominant_sample_is_pivotal") for x in fail]),
                "dominance_mean": nanmean([x["pick_dominance"] for x in fail]),
                "dominance_mean_all_states": nanmean([x["pick_dominance"] for x in sub]),
                "k1_agree_mean": nanmean([x["k1_agree"] for x in fail])}
        for k in keys:
            ma = AL.hier_mean([x for x in rows if x["model"] == "V2M"], k)
            mb = AL.hier_mean([x for x in rows if x["model"] == "V2O"], k)
            S["paired_V2M_minus_V2O"][k] = AL.boot({s: ma[s] - mb[s] for s in ma if s in mb})
        for m in MODELS:
            for K in A.K_ABL:
                sub = [x for x in krow if x["model"] == m and x["K"] == K]
                if sub:
                    S["kabl"][f"{m}_K{K}"] = {k: AL.boot(AL.hier_mean(sub, k)) for k in ("top1", "regret", "stability", "time_per_cand_s")}
                    S["kabl"][f"{m}_K{K}"]["states"] = len(sub)
        summary[dataset] = S
        print(dataset, len(sts), "states;", "kabl rows", len(krow))
        for m in MODELS:
            x = S["models"][m]
            print(f"  {m}: brier {A1.fmt(x['brier'])} FS {A1.fmt(x['false_surface_sample'])} hitdis {A1.fmt(x['hit_disagree_rays'])} "
                  f"chamfer-div {A1.fmt(x['surface_chamfer_div'])} k1_agree {A1.fmt(x['k1_agree'])} pivotal {A1.fmt(x['loo_pivotal'])} "
                  f"regret k4 {A1.fmt(x['regret_k4'])} k1 {A1.fmt(x['regret_k1_mean'])}")
    json.dump(summary, open(os.path.join(RES, "a2_summary.json"), "w"), indent=1, default=float)
    write_tables(summary)


def write_tables(S):
    f = A1.fmt
    L = []
    for dataset, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        x = S[dataset]
        u = x["unit_distance"]
        L += [f"### Table B1 — 随机未来视角质量与多样性（{nm}，{x['n_states']} 个状态；距离单位 {u}）", "",
              "| 模型 | Hit Brier ↓ | 假表面率（单样本）↓ | 漏检率（单样本）↓ | 新命中精度 ↑ | 深度相对误差 ↓ | Fair ES ↓ | Hit 分歧率 | 条件深度多样性 | 新表面 Chamfer 多样性 | 到 GT 新表面 Chamfer ↓ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, lab in (("V2O", "V2 Original"), ("V2M", "V2-Mix")):
            y = x["models"][m]
            L.append(f"| {lab} | {f(y['brier'])} | {f(y['false_surface_sample'])} | {f(y['missed_surface_sample'])} | "
                     f"{f(y['precision_new_sample'])} | {f(y['depth_rel_err'])} | {f(y['fair_es'], 2)} | {f(y['hit_disagree_rays'])} | "
                     f"{f(y['cond_depth_div'])} | {f(y['surface_chamfer_div'])} | {f(y['chamfer_to_gt_new'])} |")
        L += ["", f"### 样本对选点的影响（{nm}）", "",
              "| 模型 | K=1 选点与 K=4 一致率 | 4 个 K=1 选点的不同数 | 存在关键样本的状态比例 | 被选候选最大样本占比 | Regret K=4 (pp) | Regret K=1 均值 (pp) | Top-1 K=4 | Top-1 K=1 均值 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m, lab in (("V2O", "V2 Original"), ("V2M", "V2-Mix")):
            y = x["models"][m]
            L.append(f"| {lab} | {f(y['k1_agree'])} | {f(y['k1_distinct'], 2)} | {f(y['loo_pivotal'])} | {f(y['pick_dominance'])} | "
                     f"{f(y['regret_k4'])} | {f(y['regret_k1_mean'])} | {f(y['top1_k4'])} | {f(y['top1_k1_mean'])} |")
        if x["kabl"]:
            L += ["", f"### Table B2 — K 消融（{nm}，4 组独立种子集；耗时为共享 GPU 上的非独占计时）", "",
                  "| 模型 | K | Top-1 ↑ | Regret (pp) ↓ | 选点稳定性 ↑ | 每候选耗时 (s) ↓ |", "|---|---|---|---|---|---|"]
            for m in ("V2O", "V2M"):
                for K in A.K_ABL:
                    y = x["kabl"].get(f"{m}_K{K}")
                    if y:
                        L.append(f"| {m} | {K} | {f(y['top1'])} | {f(y['regret'])} | {f(y['stability'])} | {f(y['time_per_cand_s'])} |")
        L.append("")
    open(os.path.join(TAB, "A2_tables.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
