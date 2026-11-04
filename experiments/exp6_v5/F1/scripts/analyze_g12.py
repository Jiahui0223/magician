"""EXP6 V5 / F1 -- G1 / G2 analysis of the Dev-3 fixed states (seen2scene env, CPU; generated, do not edit by hand).
Per-state G2 records (pick by the Task 2 rule, true one-step coverage gain of the pick in pp, oracle, Top-1 within
TAU, regret, Spearman of the candidate scores vs the true gains, MAE groups) from exp6_v4/scripts/analyze_g12_v4.rows_to_recs
with the F1 model list; G1 extras from the F1 rows (hit TP / FP / FN / TN on UNKNOWN rays, Brier and reliability bins of
p_hit = mean over K = 4, depth-error histogram of UNKNOWN true-hit rays). Subsets: stratum B0 / B2 / pooled x phase
early (t <= 20) / late (t > 20) / all. Point estimates: G2 and MAE = state-equal means; hit / Brier metrics = pooled
counts. Paired differences vs B0 and A vs B (same step): per scene the mean over its states of the per-state difference,
scene-level bootstrap 95% CI (3 scenes: exploratory) and a state-level bootstrap (states are not independent: secondary).
Outputs results/g1_predictions.csv, results/g2_ranking.csv, results/g12_summary.json, reports/G1.md, reports/G2.md."""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v4", "scripts"))
import analyze_g12_v4 as A4  # noqa: E402

EA = A4.EA
W = os.path.join(R, "work_dirs", "exp6_v5", "F1", "eval_fixed", "dev3")
MODELS = ("B0", "A2k", "A5k", "B2k", "B5k")
PAIRS = (("A2k", "B0"), ("A5k", "B0"), ("B2k", "B0"), ("B5k", "B0"), ("B2k", "A2k"), ("B5k", "A5k"))
EB = (-4.0, 2.0, 12)
G2K = ("gain", "top1", "regret", "rho")
G1K = ("mae_unknown", "mae_unknown_new", "mae_known", "hit_agree", "rel_std", "spread_err_rho")


def load_rows():
    rows = []
    for st in ("B0", "B2"):
        for f in sorted(glob.glob(os.path.join(W, st, "*.jsonl"))):
            rows += [json.loads(l) for l in open(f)]
    return rows


def g1_extra(r, m):
    p = r["pred"][m]
    tp, fp, fn, tn = (p.get(f"{k}_unk", 0) for k in ("tp", "fp", "fn", "tn"))
    rel = [(p.get(f"rel_n_{i}", 0), p.get(f"rel_pos_{i}", 0)) for i in range(5)]
    h = np.asarray(p["h2_unk"]) if "h2_unk" in p else np.zeros((10, EB[2]), int)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "brier_sum": p.get("brier_sum_unk", 0.0), "n_unk": p.get("n_rays_unk", 0), "rel": rel, "h2": h}


def pooled(xs):
    if not xs:
        return {k: None for k in ("hit_precision_unk", "hit_recall_unk", "hit_f1_unk", "false_nohit_rate_unk", "false_hit_rate_unk", "gt_hit_prior_unk",
                                  "pred_hit_rate_unk", "brier_unk", "ece_unk", "n_unknown_rays", "abs_err_p50_bin", "abs_err_p90_bin", "reliability")}
    tp, fp, fn, tn = (sum(x[k] for x in xs) for k in ("tp", "fp", "fn", "tn"))
    n = sum(x["n_unk"] for x in xs)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    rel = np.sum([np.asarray(x["rel"], float) for x in xs], 0)                # [5, (n, pos)]
    pv = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    ece = float(np.sum(rel[:, 0] * np.abs(pv - rel[:, 1] / np.maximum(rel[:, 0], 1))) / max(1.0, rel[:, 0].sum()))
    h = np.sum([x["h2"] for x in xs], 0).sum(0)                              # error histogram
    cen = 10 ** (EB[0] + (np.arange(EB[2]) + 0.5) * (EB[1] - EB[0]) / EB[2])
    cdf = np.cumsum(h) / max(1, h.sum())
    q = lambda a: float(cen[min(np.searchsorted(cdf, a), len(cen) - 1)]) if h.sum() else None  # noqa: E731
    return {"hit_precision_unk": prec, "hit_recall_unk": rec, "hit_f1_unk": 2 * prec * rec / max(1e-12, prec + rec),
            "false_nohit_rate_unk": fn / max(1, tp + fn), "false_hit_rate_unk": (fp / (fp + tn)) if fp + tn else None,
            "gt_hit_prior_unk": (tp + fn) / max(1, tp + fp + fn + tn), "pred_hit_rate_unk": (tp + fp) / max(1, tp + fp + fn + tn),
            "brier_unk": sum(x["brier_sum"] for x in xs) / max(1, n), "ece_unk": ece, "n_unknown_rays": n,
            "abs_err_p50_bin": q(0.5), "abs_err_p90_bin": q(0.9), "reliability": rel.tolist()}


def boot(v, n=10000, seed=0):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def subset(recs, stratum, phase):
    out = [x for x in recs if stratum in (None, x["src"])]
    if phase == "early":
        out = [x for x in out if x["t"] <= 20]
    elif phase == "late":
        out = [x for x in out if x["t"] > 20]
    return out


def main():
    rows = load_rows()
    assert rows, "no F1 G1/G2 rows"
    assert all(r.get("repro_B0_eq_E2") for r in rows), "B0 regression failed in some rows"
    present = tuple(m for m in MODELS if all(m in r["scores"] for r in rows))
    A4.MODELS = present
    EA.ST.setup()
    recs = A4.rows_to_recs("dev3", rows)
    byid = {r["sid"]: r for r in rows}
    for x in recs:
        for m in present:
            x[f"{m}_g1x"] = g1_extra(byid[x["sid"]], m)
            tm = byid[x["sid"]].get("timing", {}).get(m) or {}
            x[f"{m}_time_s"] = float(sum(v for v in tm.values() if isinstance(v, (int, float))))   # projection + cond + sampling + decode/score
    with open(os.path.join(F1, "results", "g2_ranking.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "scene", "start", "t", "stratum", "model"] + list(G2K) + ["oracle", "pick"])
        for x in recs:
            for m in present:
                w.writerow([x["sid"], x["scene"], x["start"], x["t"], x["src"], m] + [x.get(f"{m}_{k}") for k in G2K] + [x[f"{m}_oracle"], x[f"{m}_pick"]])
    with open(os.path.join(F1, "results", "g1_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "scene", "start", "t", "stratum", "model"] + list(G1K) + ["tp_unk", "fp_unk", "fn_unk", "tn_unk", "brier_sum_unk", "n_unknown_rays", "time_s"])
        for x in recs:
            for m in present:
                g = x[f"{m}_g1x"]
                w.writerow([x["sid"], x["scene"], x["start"], x["t"], x["src"], m] + [x.get(f"{m}_{k}") for k in G1K] +
                           [g["tp"], g["fp"], g["fn"], g["tn"], g["brier_sum"], g["n_unk"], x.get(f"{m}_time_s")])
    S = {}
    for sname, stv in (("pooled", None), ("B0_states", "B0"), ("B2_states", "B2")):
        S[sname] = {}
        for ph in ("early", "late", "all"):
            sub = subset(recs, stv, ph)
            blk = {"n_states": len(sub), "scenes": sorted({x["scene"] for x in sub})}
            for m in present:
                d = {k: float(np.nanmean([x[f"{m}_{k}"] for x in sub if x.get(f"{m}_{k}") is not None])) if sub else None for k in G2K + G1K}
                d.update(pooled([x[f"{m}_g1x"] for x in sub]))
                d["time_s_mean"] = float(np.nanmean([x.get(f"{m}_time_s", np.nan) for x in sub])) if sub else None
                blk[m] = d
            pairs = []
            for a_, b_ in PAIRS:
                if a_ not in present or b_ not in present:
                    continue
                pr = {"a": a_, "b": b_}
                for k in ("gain", "top1", "regret", "mae_unknown", "hit_agree"):
                    per_state = [(x["scene"], x[f"{a_}_{k}"] - x[f"{b_}_{k}"]) for x in sub if x.get(f"{a_}_{k}") is not None and x.get(f"{b_}_{k}") is not None]
                    sc = {}
                    for s, v in per_state:
                        sc.setdefault(s, []).append(v)
                    scm = {s: float(np.mean(v)) for s, v in sc.items()}
                    pr[k] = {"d": float(np.mean(list(scm.values()))) if scm else None, "ci_scene": boot(list(scm.values())),
                             "ci_state": boot([v for _, v in per_state]), "wins": int(sum(v > 0 for v in scm.values())), "losses": int(sum(v < 0 for v in scm.values())),
                             "per_scene": scm, "n_states": len(per_state)}
                pairs.append(pr)
            blk["pairs"] = pairs
            S[sname][ph] = blk
    json.dump(S, open(os.path.join(F1, "results", "g12_summary.json"), "w"), indent=1, default=float)
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731

    def fdp(q, scale=1.0):
        if not q or q["d"] is None:
            return "—"
        c, cs = q["ci_scene"], q["ci_state"]
        return f"{q['d'] * scale:+.3f} [场景 {c[0] * scale:+.3f}, {c[1] * scale:+.3f}; 状态 {cs[0] * scale:+.3f}, {cs[1] * scale:+.3f}] {q['wins']}/{q['losses']}"
    hdr = ("*自动生成（`scripts/analyze_g12.py`）。Dev-3 固定状态（E1.2 的 B0 层与 B2 层），所有模型都用 S10 输入与 S10 DistP1。"
           "误差 p50 / p90 是半个数量级宽的直方图 bin 中心，分辨率很粗，不用于比较模型。B0 的分数与 E2 记录逐位一致（每一行都检查）。差值的 CI 分两种：场景级（只有 3 个场景，仅作探索）和状态级（状态之间并不独立，只作次要参考）。*")
    L1 = ["# G1 — 未来深度 / 命中诊断（固定状态配对）", "", hdr, ""]
    L2 = ["# G2 — 候选排序与真实一步覆盖增益（固定状态配对）", "", hdr, "",
          "- 增益 = 所选候选的真实一步覆盖增益（百分点）；Top-1 = 所选候选与 oracle 最优在 TAU 以内；",
          "- Regret = oracle 增益 − 所选增益；ρ = 候选分数与真实增益的 Spearman 相关；",
          "- 选择规则与 Task 2 相同（未访问优先）。", ""]
    for sname in ("pooled", "B0_states", "B2_states"):
        for ph in ("early", "late", "all"):
            blk = S[sname][ph]
            ttl = f"## {sname} · {'t ≤ 20' if ph == 'early' else 't > 20' if ph == 'late' else '全部'}（{blk['n_states']} 个状态）"
            L1 += [ttl, "", "| 模型 | unknown MAE | unknown∩new MAE | known MAE | 命中一致率 | UNKNOWN 命中 P / R / F1 | 假无命中率 | 假命中率 | GT 命中先验 | Brier | ECE | 误差 p50 / p90（直方图 bin） | 每状态评分耗时 s |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            L2 += [ttl, "", "| 模型 | 增益 (pp) | Top-1 | Regret (pp) | ρ |", "|---|---|---|---|---|"]
            for m in present:
                d = blk[m]
                L1.append(f"| {m} | {f3(d['mae_unknown'])} | {f3(d['mae_unknown_new'])} | {f3(d['mae_known'])} | {f3(d['hit_agree'])} | "
                          f"{f3(d['hit_precision_unk'])} / {f3(d['hit_recall_unk'])} / {f3(d['hit_f1_unk'])} | {f3(d['false_nohit_rate_unk'])} | {f3(d['false_hit_rate_unk'])} | "
                          f"{f3(d['gt_hit_prior_unk'])} | {f3(d['brier_unk'])} | {f3(d['ece_unk'])} | {f3(d['abs_err_p50_bin'])} / {f3(d['abs_err_p90_bin'])} | {f3(d['time_s_mean'])} |")
                L2.append(f"| {m} | {f3(d['gain'])} | {f3(d['top1'])} | {f3(d['regret'])} | {f3(d['rho'])} |")
            L1 += ["", "| 配对 A − B | Δ unknown MAE | Δ 命中一致率 |", "|---|---|---|"]
            L2 += ["", "| 配对 A − B | Δ 增益 (pp) | Δ Top-1 | Δ Regret (pp) |", "|---|---|---|---|"]
            for pr in blk["pairs"]:
                L1.append(f"| {pr['a']} − {pr['b']} | {fdp(pr['mae_unknown'])} | {fdp(pr['hit_agree'])} |")
                L2.append(f"| {pr['a']} − {pr['b']} | {fdp(pr['gain'])} | {fdp(pr['top1'])} | {fdp(pr['regret'])} |")
            L1.append("")
            L2.append("")
    open(os.path.join(F1, "reports", "G1.md"), "w").write("\n".join(L1) + "\n")
    open(os.path.join(F1, "reports", "G2.md"), "w").write("\n".join(L2) + "\n")
    print("states", len(recs), "models", present)


if __name__ == "__main__":
    main()
