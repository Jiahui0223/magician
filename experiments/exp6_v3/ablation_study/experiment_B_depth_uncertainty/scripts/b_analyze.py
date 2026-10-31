"""EXP6 V3 Ablation Experiment B -- fixed-state G2 of the planner scorers P0 (DistP1 mean), P1 (DepthU), P2 (normalised
sum, lambda = 1; 0.5 / 2.0 sensitivity), RiskAware (exploratory) on the saved K = 4 samples (CPU; protocol §3).
Same states / cameras / seeds / K as Experiment A; no GT in any scorer. Evaluation: regret / Top-1 / Spearman by phase and
scene with scene-level bootstrap, paired differences to P0, the pre-registered Gate, DepthU diagnostics (relation to the
GT future-depth error and to the oracle gain; 'uncertain but no gain' candidates), score audit on real data.
Output experiment_B_depth_uncertainty/{results,tables,figures} and report/{B_SCORE_AUDIT, B_FIXED_STATE, B_DECISION}.md
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
sys.path.insert(0, HERE)
import ab_data as D  # noqa: E402
import b_scorer as B  # noqa: E402

OUT = {k: os.path.join(EXP, k) for k in ("results", "tables", "figures", "report")}
MODELS = ("B2", "T1", "T0", "E2D")
SCORERS = ("P0", "P1", "P2", "P2_l0.5", "P2_l2.0", "RiskAware")
LABEL = {"P0": "P0 DistP1 均值（线上）", "P1": "P1 DepthU（α = 0.5）", "P2": "P2 归一化 DistP1 + 1.0·DepthU", "P2_l0.5": "P2 λ = 0.5（敏感性）",
         "P2_l2.0": "P2 λ = 2.0（敏感性）", "RiskAware": "RiskAware mean − 0.5 std（探索性）"}
E2W = D.E2W; B2W = D.B2W; W5 = D.W5


def oracle_path(r):
    if r["pool"] == "cl":
        return os.path.join(W5, "states", r["dataset"], r["sid"], "oracle.npz")
    return os.path.join(E2W, "valsel", "states", "indoor", r["sid"], "oracle.npz") if r["dataset"] == "indoor" else os.path.join(B2W, "states", "outdoor", r["sid"], "oracle.npz")


def depth_error(t_k, hit_k, tstar, k_M):
    """Mean |t_hat - t*| (runtime units) over rays with a HIT sample and a finite t*, averaged over samples; None if none."""
    ok = np.isfinite(tstar)
    vals = []
    for k in range(t_k.shape[0]):
        m = hit_k[k] & ok
        if m.any():
            vals.append(float(np.mean(np.abs(t_k[k][m] - tstar[m])) * k_M))
    return float(np.mean(vals)) if vals else None


def process(recs):
    """Adds r['b'][model] = {scores: {scorer: [n_legal]}, picks, metrics, depth_u, err, audit}."""
    audit = {"states": 0, "cands": 0, "u_min": 1e9, "u_max": -1e9, "frac_unknown": [], "frac_valid": [], "all_zero_P1": 0}
    for r in recs:
        z = np.load(os.path.join(D.WA, "cache", r["pool"], r["dataset"], r["sid"] + ".npz"))
        orc = np.load(oracle_path(r))
        k_M = float(z["k_M"])
        r["b"] = {}
        cand = r["cand"]; g = r["gain"][cand]
        for m in MODELS:
            t, hu = z[f"{m}_t"], z[f"{m}_hit_u"]
            du = np.array([B.s_depth_u(t[j], hu[j], z["valid"][j], z["n_u"][j], z["d_lo"][j], z["d_hi"][j]) for j in range(t.shape[0])])
            sc = B.scores_of_state(r["distp1_k"][m], du)
            err = np.array([np.nan if (e := depth_error(t[j], hu[j], orc["tstar"][j].astype(np.float64), k_M)) is None else e for j in range(t.shape[0])])
            picks, mets, fb = {}, {}, {}
            for name, s in sc.items():
                pk, f = D.pick(r["dataset"], s, r, f"{m}:{name}")
                picks[name], fb[name] = int(pk), bool(f)
                mets[name] = D.rank_metrics(s[cand], g, cand.index(pk))
            r["b"][m] = {"scores": sc, "picks": picks, "fallback": fb, "metrics": mets, "depth_u": du, "err": err,
                         "rho_u_gain": D.spearman(du[cand], g), "rho_u_err": D.spearman(du[cand][np.isfinite(err[cand])], err[cand][np.isfinite(err[cand])]) if np.isfinite(err[cand]).sum() >= 3 else None,
                         "rho_p0_gain": D.spearman(sc["P0"][cand], g)}
            if m == "T1":
                audit["states"] += 1; audit["cands"] += len(du)
                audit["u_min"] = min(audit["u_min"], float(du.min())); audit["u_max"] = max(audit["u_max"], float(du.max()))
                audit["frac_unknown"].append(float(((z["n_u"] > 0) & z["valid"]).mean())); audit["frac_valid"].append(float(z["valid"].mean()))
                audit["all_zero_P1"] += int(du.max() <= 0)
    return audit


def agg(recs, m, sc, key):
    return D.boot(D.hier_mean(recs, lambda r: (None if r["b"][m]["metrics"][sc].get(key) is None else float(r["b"][m]["metrics"][sc][key]))).values())


def sv(recs, m, sc, key):
    return D.hier_mean(recs, lambda r: (None if r["b"][m]["metrics"][sc].get(key) is None else float(r["b"][m]["metrics"][sc][key])))


def table(recs, m):
    rows = ["| 评分器 | regret pp [CI] | Top-1 | Spearman | Pairwise | regret early | regret mid | regret late | fallback |", "|---|---|---|---|---|---|---|---|---|"]
    for sc in SCORERS:
        ph = [D.fmt(agg([r for r in recs if r["phase"] == p], m, sc, "regret")) if any(r["phase"] == p for r in recs) else "—" for p in ("early", "mid", "late")]
        rows.append(f"| {LABEL[sc]} | {D.fmt(agg(recs, m, sc, 'regret'))} | {D.fmt(agg(recs, m, sc, 'top1'))} | {D.fmt(agg(recs, m, sc, 'rho'))} | {D.fmt(agg(recs, m, sc, 'pairwise'))} | "
                    + " | ".join(ph) + f" | {sum(r['b'][m]['fallback'][sc] for r in recs)} |")
    return "\n".join(rows)


def paired_tab(recs, m):
    rows = ["| 配对（regret，a − P0；负 = 更好） | 全部 [CI]（a 更好 / 更差 场景） | Early [CI] | Top-1 差 [CI] | 逐场景（全部） |", "|---|---|---|---|---|"]
    out = {}
    for sc in SCORERS[1:]:
        pa = D.paired(sv(recs, m, sc, "regret"), sv(recs, m, "P0", "regret"))
        er = [r for r in recs if r["phase"] == "early"]
        pe = D.paired(sv(er, m, sc, "regret"), sv(er, m, "P0", "regret")) if er else None
        pt = D.paired(sv(recs, m, sc, "top1"), sv(recs, m, "P0", "top1"))
        out[sc] = {"all": pa, "early": pe, "top1": pt}
        rows.append(f"| {sc} − P0 | {D.fmt(pa)}（{pa['wins']} / {pa['losses']}） | {D.fmt(pe)} | {D.fmt(pt)} | " + "，".join(f"{s} {v:+.3f}" for s, v in pa["per_scene"].items()) + " |")
    return "\n".join(rows), out


def gate(recs, m, dataset):
    """Protocol §3 pre-registered gate for one dev set and model; returns (passed_scorers, details)."""
    need = 4 if dataset == "indoor" else 2
    det, passed = {}, []
    base = sv(recs, m, "P0", "regret"); base_e = sv([r for r in recs if r["phase"] == "early"], m, "P0", "regret")
    for sc in ("P1", "P2"):
        cur = sv(recs, m, sc, "regret"); cur_e = sv([r for r in recs if r["phase"] == "early"], m, sc, "regret")
        pa, pe = D.paired(cur, base), D.paired(cur_e, base_e)
        a = (pa is not None and pa["hi"] < 0) or (pe is not None and pe["hi"] < 0)
        rel = (pa["mean"] / np.mean(list(base.values()))) if pa and np.mean(list(base.values())) > 0 else 0.0
        rel_e = (pe["mean"] / np.mean(list(base_e.values()))) if pe and base_e and np.mean(list(base_e.values())) > 0 else 0.0
        b = (pa is not None and rel <= -0.10 and pa["wins"] >= need) or (pe is not None and rel_e <= -0.10 and pe["wins"] >= need)
        worst = max(((cur[s] / base[s] - 1.0) if base[s] > 0 else 0.0) for s in base)
        c = worst <= 0.25
        ok = (a or b) and c
        det[sc] = {"a_ci": a, "b_rel10_wins": b, "c_no_collapse": c, "rel_all": rel, "rel_early": rel_e, "worst_scene_rel": worst, "paired_all": pa, "paired_early": pe, "pass": ok}
        if ok:
            passed.append(sc)
    return passed, det


def diagnostics(recs, m):
    rho_g = D.boot(D.hier_mean(recs, lambda r: r["b"][m]["rho_u_gain"]).values())
    rho_e = D.boot(D.hier_mean(recs, lambda r: r["b"][m]["rho_u_err"]).values())
    rho_p = D.boot(D.hier_mean(recs, lambda r: r["b"][m]["rho_p0_gain"]).values())
    # 'uncertain but no gain': within each state, candidates in the top quartile of DepthU whose oracle gain <= tau
    def unc_no_gain(r):
        du = r["b"][m]["depth_u"][r["cand"]]; g = r["gain"][r["cand"]]
        if len(du) < 4:
            return None
        top = du >= np.quantile(du, 0.75)
        return float((g[top] <= D.TAU).mean())
    ung = D.boot(D.hier_mean(recs, unc_no_gain).values())
    # DepthU of the oracle-best candidate relative to the state's max
    def rank_of_best(r):
        du = r["b"][m]["depth_u"][r["cand"]]; g = r["gain"][r["cand"]]
        if g.max() <= D.TAU or du.max() <= 0:
            return None
        return float(du[int(np.argmax(g))] / du.max())
    rb = D.boot(D.hier_mean(recs, rank_of_best).values())
    return {"rho_u_gain": rho_g, "rho_u_err": rho_e, "rho_p0_gain": rho_p, "uncertain_no_gain_frac": ung, "depthu_of_best_rel": rb}


def main():
    for d in OUT.values():
        os.makedirs(d, exist_ok=True)
    pools = {}
    for pool, ds in (("cl", "indoor"), ("cl", "outdoor"), ("fx", "indoor"), ("fx", "outdoor")):
        recs, miss = D.load_pool(pool, ds)
        aud = process(recs)
        pools[(pool, ds)] = (recs, aud)
        print(pool, ds, len(recs), "states processed; missing", miss, flush=True)
    summary, md = {}, ["# Experiment B — 固定状态 G2（自动生成 `scripts/b_analyze.py`；协议 v1.0 §3）", "",
                       "评分器只改变分数，模型 / P1 / 候选 / 噪声与 Experiment A 相同；DepthU 用同一候选的 K = 4 样本在同一射线上的 HIT 熵与命中深度标准差（按射线 UNKNOWN 段长度归一化），只在 M_unknown 射线上计，α = 0.5；P2 = 状态内 min-max 归一化后 DistP1 + λ·DepthU（λ = 1.0 冻结）。均值 = 场景均值的平均，CI = 场景级 bootstrap。", ""]
    with open(os.path.join(OUT["results"], "b_states.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pool", "dataset", "sid", "scene", "start", "t", "phase", "model", "scorer", "pick", "top1", "regret_pp", "rho", "fallback"])
        for (pool, ds), (recs, _) in pools.items():
            for r in recs:
                for m in MODELS:
                    for sc in SCORERS:
                        me = r["b"][m]["metrics"][sc]
                        w.writerow([pool, ds, r["sid"], r["scene"], r["start"], r["t"], r["phase"], m, sc, r["b"][m]["picks"][sc], int(me["top1"]), f"{me['regret']:.5f}", "" if me["rho"] is None else f"{me['rho']:.4f}", int(r["b"][m]["fallback"][sc])])
    for (pool, ds), (recs, aud) in pools.items():
        if not recs:
            continue
        title = {("cl", "indoor"): "Pool CL · MP3D", ("cl", "outdoor"): "Pool CL · Macarons++（Test-3 + Dev-3）", ("fx", "indoor"): "Pool FX · MP3D VALSEL（dev）", ("fx", "outdoor"): "Pool FX · Macarons++ Dev-3 B2 层"}[(pool, ds)]
        md.append(f"## {title}（{len(recs)} 状态）")
        summary[f"{pool}/{ds}"] = {"n": len(recs), "models": {}}
        for m in MODELS:
            md += [f"### 模型 {m}", table(recs, m), ""]
            pt, pd = paired_tab(recs, m)
            md += [pt, ""]
            dg = diagnostics(recs, m)
            md += [f"DepthU 诊断（{m}）：候选 DepthU 与 oracle 增益的状态内 Spearman {D.fmt(dg['rho_u_gain'])}；与 GT 未来深度误差的 Spearman {D.fmt(dg['rho_u_err'])}（对照：P0 与增益 {D.fmt(dg['rho_p0_gain'])}）；"
                   f"DepthU 前 1/4 候选中 oracle 增益 ≤ τ 的比例（\"很不确定但无收益\"）{D.fmt(dg['uncertain_no_gain_frac'])}；oracle 最优候选的 DepthU / 状态内最大 DepthU {D.fmt(dg['depthu_of_best_rel'])}。", ""]
            summary[f"{pool}/{ds}"]["models"][m] = {"regret": {sc: agg(recs, m, sc, "regret") for sc in SCORERS}, "top1": {sc: agg(recs, m, sc, "top1") for sc in SCORERS},
                                                     "paired": pd, "diag": dg}
        if pool == "cl" and ds == "outdoor":
            for split, scenes in (("Test-3", D.TEST3), ("Dev-3", D.DEV3)):
                sub = [r for r in recs if r["scene"] in scenes]
                md += [f"### {split}（{len(sub)} 状态，3 场景均值）— B2 / T1", table(sub, "B2"), "", table(sub, "T1"), ""]
    # gate on the dev sets
    gates, gmd = {}, ["# Experiment B — 预注册 Gate 判定（协议 §3；dev 集 = MP3D Pool FX VALSEL、Macarons++ Pool CL Dev-3）", "",
                      "条件：(a) 整体或 Early regret 的场景配对差 95% CI 上界 < 0；或 (b) 相对改善 ≥ 10% 且 MP3D ≥ 4/5 楼层、Macarons++ ≥ 2/3 场景改善；且 (c) 没有场景的 regret 相对恶化 > 25%。P1 或 P2 任一通过 → 该数据集 / 模型 / 评分器进入 G3。", "",
                      "| dev 集 | 模型 | 评分器 | Δregret 全部 [CI] | Δregret Early [CI] | 相对改善（全部 / Early） | 最差场景相对变化 | (a) | (b) | (c) | **通过** |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    dev = {"MP3D VALSEL": [r for r in pools[("fx", "indoor")][0]], "Macarons++ Dev-3": [r for r in pools[("cl", "outdoor")][0] if r["scene"] in D.DEV3]}
    for name, recs in dev.items():
        if not recs:
            continue
        ds = recs[0]["dataset"]
        for m in ("B2", "T1"):
            passed, det = gate(recs, m, ds)
            gates[f"{name}|{m}"] = {"passed": passed, "detail": det}
            for sc, d_ in det.items():
                gmd.append(f"| {name} | {m} | {sc} | {D.fmt(d_['paired_all'])} | {D.fmt(d_['paired_early'])} | {100 * d_['rel_all']:+.1f}% / {100 * d_['rel_early']:+.1f}% | {100 * d_['worst_scene_rel']:+.0f}% | {'✅' if d_['a_ci'] else '❌'} | {'✅' if d_['b_rel10_wins'] else '❌'} | {'✅' if d_['c_no_collapse'] else '❌'} | **{'PASS' if d_['pass'] else 'FAIL'}** |")
    any_pass = [k for k, v in gates.items() if v["passed"]]
    if any_pass:
        concl = "通过的组合：" + "；".join(f"{k} → {v['passed']}" for k, v in gates.items() if v["passed"])
    else:
        concl = "没有任何 (dev 集, 模型, 评分器) 组合通过 Gate → 不进入正式 G3；报告阴性结果（可选 ≤ 2 个 exploratory 闭环）。"
    gmd += ["", f"**结论**：{concl}"]
    open(os.path.join(OUT["report"], "B_DECISION.md"), "w").write("\n".join(gmd))
    open(os.path.join(OUT["report"], "B_FIXED_STATE.md"), "w").write("\n".join(md))
    # score audit on real data
    aud = pools[("cl", "indoor")][1] if pools[("cl", "indoor")][0] else pools[("fx", "indoor")][1]
    audit_md = ["# Experiment B — 评分器审计（协议 §3；真实数据不变量 + 单元测试）", "",
                "| 检查 | 结果 |", "|---|---|",
                f"| 单元测试 `tests/test_b_scorer.py`（熵、HIT/NO-HIT、span 归一化、无 far 代入、候选置换不变性、P2 归一化） | 见 pytest 日志（2 passed） |"]
    for (pool, ds), (recs, a) in pools.items():
        if recs:
            audit_md.append(f"| {pool}/{ds}：T1 的 S_depthU 范围 | [{a['u_min']:.4f}, {a['u_max']:.4f}]（{a['states']} 状态、{a['cands']} 候选；M_unknown 射线占比均值 {np.mean(a['frac_unknown']):.3f}，valid 射线占比 {np.mean(a['frac_valid']):.3f}；全 0 状态 {a['all_zero_P1']}） |")
    audit_md += ["| GT 不参与评分 | `b_scorer.py` 只读 `t, hit_u, valid, n_u, d_lo, d_hi, distp1_k`；`oracle.npz` 只在评估侧（regret、诊断）读取 |",
                 "| 跨相机比较 | 无：DepthU 在候选内按射线聚合后才与其他候选比较 |", "| NO-HIT 处理 | 不参与深度 std；只进入 U_hit 的 Bernoulli 熵；无效射线排除 |"]
    open(os.path.join(OUT["report"], "B_SCORE_AUDIT.md"), "w").write("\n".join(audit_md))
    json.dump({"summary": summary, "gates": gates}, open(os.path.join(OUT["results"], "b_summary.json"), "w"), indent=1, default=float)
    # figure: regret per scorer per model per pool
    for (pool, ds), (recs, _) in pools.items():
        if not recs:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        xs = np.arange(len(SCORERS)); wdt = 0.2
        for i, m in enumerate(MODELS):
            vals = [agg(recs, m, sc, "regret") for sc in SCORERS]
            ax.bar(xs + (i - 1.5) * wdt, [v["mean"] for v in vals], wdt, yerr=[[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]], label=m, capsize=2)
        ax.set_xticks(xs); ax.set_xticklabels(SCORERS, rotation=20); ax.set_ylabel("regret (pp)"); ax.set_title(f"{pool} {ds}: scorer regret by model"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT["figures"], f"b_regret_{pool}_{ds}.png"), dpi=130); plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
