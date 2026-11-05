"""EXP6 V3 / ablation E -- offline test of a "refutation memory" re-scoring (CPU; diagnostic only; no closed loop, no planner
change). Inputs: e_predict.py outputs (predicted novel cells of every legal candidate, union over the K samples with the
max weight per cell; per-candidate predicted value), the episode's service records (the agent's own S10 observations) and the
e_oracle.py true one-step gains (used ONLY to score the re-ranking afterwards).
Refutation (online-computable: uses only the agent's own past predictions and observations): after the executed view of step
t is observed (frame t+1), every cell that was predicted novel for that executed view but has no observed S10 point of
frame t+1 within TOL = 2.0 is marked refuted ("looked there, did not see it"). At step t the re-scored value of a candidate
= sum of the weights of its predicted cells that are not refuted by any executed view before t.
Compared with the original predicted union value (same cells, no refutation) and with the per-candidate planner mean
value (pred_value): Top-1 against the true one-step coverage gain, the true gain of the argmax (pp), Spearman rho,
per phase (colosseum low-plateau episodes: plateau / pre; others: t < 20 / t >= 20). Also the persistence of predicted
empty cells (share of the executed view's predicted cells that had already been predicted for an earlier executed view).
Outputs results/refute_summary.json, results/REFUTE_TABLES.md."""
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_E_stability_plateau")
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
TOL = 2.0
M20 = (1 << 20) - 1
NAME = {"V3": "H0", "V3-H3": "H3", "V3-D1-H3-NORISK": "D1", "V3-YBEST-H3": "YBEST-H3"}


def dec(ids):
    return np.stack([(ids >> 40) & M20, (ids >> 20) & M20, ids & M20], 1).astype(np.float64) - (1 << 19) + 0.5


def onset(c):
    return next((t for t in range(101) if c[min(t + 20, 100)] - c[t] < 0.02), None)


def one(fpred):
    mid = os.path.basename(os.path.dirname(fpred))
    base = os.path.basename(fpred).replace("_r0_predict.jsonl", "")
    scene, k = base.rsplit("_s", 1)
    k = int(k)
    rec = json.load(open(os.path.join(EP, mid, f"{scene}_s{k}.json")))
    c = rec["coverage_official"]
    ep = json.load(open(os.path.join(R, rec["episode_file"])))
    svc = os.path.join(os.path.dirname(os.path.dirname(os.path.join(R, rec["episode_file"]))), "svc", ep["tag"])
    z = np.load(fpred.replace(".jsonl", "_cells.npz"))
    idx = {(int(a), int(b)): (int(o), int(n)) for a, b, o, n in z["index"]}
    rows = {json.loads(l)["t"]: json.loads(l) for l in open(fpred)}
    refuted, seen_pred = set(), set()
    on = onset(c)
    low = c[100] < 0.70 if scene == "colosseum" else None
    out = []
    for t in sorted(rows):
        r = rows[t]
        ca = r["cands"]
        dc = np.asarray([x["true_dcov_pp"] for x in ca])
        best = dc.max()
        v_orig, v_ref = [], []
        for j in range(len(ca)):
            off, n = idx[(t, j)]
            ids, w = z["ids"][off:off + n], z["w"][off:off + n]
            keep = np.fromiter((x not in refuted for x in ids.tolist()), bool, len(ids))
            v_orig.append(float(w.sum()))
            v_ref.append(float(w[keep].sum()))
        v_orig, v_ref, pv = np.asarray(v_orig), np.asarray(v_ref), np.asarray([x["pred_value"] for x in ca])
        ex = r["executed"]
        off, n = idx[(t, ex)]
        ex_ids = z["ids"][off:off + n]
        persist = float(np.mean([x in seen_pred for x in ex_ids.tolist()])) if n else np.nan
        row = {"scene": scene, "policy": NAME.get(mid, mid), "start": k, "t": t, "low": low, "phase": "pre" if on is None or t < on else "plateau", "best": float(best),
               "persist": persist, "refuted_share_exec": float(1 - v_ref[ex] / v_orig[ex]) if v_orig[ex] > 0 else np.nan}
        for nm, v in (("pred", pv), ("union", v_orig), ("refute", v_ref)):
            j = int(np.argmax(v)) if v.max() > 0 else None
            row[f"top1_{nm}"] = (dc[j] >= best - 1e-9) if j is not None else None
            row[f"gain_{nm}"] = float(dc[j]) if j is not None else None
            row[f"rho_{nm}"] = float(spearmanr(v, dc)[0]) if len(v) >= 3 and np.ptp(v) > 0 and np.ptp(dc) > 0 else np.nan
        out.append(row)
        # online update after the executed view is observed (frame t+1)
        p1 = os.path.join(svc, f"svc_{scene}_s{k}_t{t + 1}.npz")
        if os.path.exists(p1) and n:
            obs = np.load(p1)["sample_points"].astype(np.float64).reshape(-1, 3)
            if len(obs):
                d = cKDTree(obs).query(dec(ex_ids), k=1, distance_upper_bound=TOL)[0]
                refuted.update(ex_ids[~np.isfinite(d)].tolist())
            seen_pred.update(ex_ids.tolist())
    return out


def agg(sel):
    if not sel:
        return None
    m = lambda q: float(np.nanmean([np.nan if r.get(q) is None else float(r[q]) for r in sel]))  # noqa: E731
    return {"steps": len(sel), **{q: m(q) for q in ("best", "top1_pred", "top1_union", "top1_refute", "gain_pred", "gain_union", "gain_refute",
                                                    "rho_pred", "rho_union", "rho_refute", "persist", "refuted_share_exec")}}


def main():
    rows = []
    for f in sorted(x for x in glob.glob(os.path.join(W, "oracle", "*", "*_r0_predict.jsonl")) if "/smoke/" not in x):
        if os.path.exists(f.replace(".jsonl", "_cells.npz")):
            rows += one(f)
    groups = [("colosseum 低平台 episode 的平台阶段", lambda r: r["scene"] == "colosseum" and r["low"] and r["phase"] == "plateau"),
              ("colosseum 低平台 episode 的平台前", lambda r: r["scene"] == "colosseum" and r["low"] and r["phase"] == "pre"),
              ("colosseum 成功 episode t < 20", lambda r: r["scene"] == "colosseum" and r["low"] is False and r["t"] < 20),
              ("colosseum 成功 episode t ≥ 20", lambda r: r["scene"] == "colosseum" and r["low"] is False and r["t"] >= 20),
              ("pisa / alhambra t < 20", lambda r: r["scene"] != "colosseum" and r["t"] < 20), ("pisa / alhambra t ≥ 20", lambda r: r["scene"] != "colosseum" and r["t"] >= 20),
              ("全部 t < 20", lambda r: r["t"] < 20), ("全部 t ≥ 20", lambda r: r["t"] >= 20)]
    S = {g: agg([r for r in rows if f(r)]) for g, f in groups}
    S["_episodes"] = sorted({(r["scene"], r["policy"], r["start"]) for r in rows})
    json.dump(S, open(os.path.join(EX, "results", "refute_summary.json"), "w"), indent=1, default=str)
    f2 = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.2f}"  # noqa: E731
    L = ["# REFUTE_TABLES — “反证记忆”离线重打分（`scripts/refute_analysis.py`；诊断，不是闭环结果）", "",
         f"- episode：{len(S['_episodes'])} 条（{', '.join(f'{a} {b} s{c}' for a, b, c in S['_episodes'])}）。",
         "- 原始 = 规划器的预测均值（pred_value）；并集 = 同一预测按样本并集（每 cell 取最大权重）的值；反证 = 并集值去掉被此前已执行视图“看过却没看到”的 cell。",
         "- Top-1 与所选真实增益都对照 GT 一步覆盖增益（pp）；GT 只用于评分，不参与重打分。", "",
         "| 步集合 | 步数 | 最佳一步增益 pp | Top-1：原始 / 并集 / 反证 | 所选真实增益 pp：原始 / 并集 / 反证 | ρ：原始 / 并集 / 反证 | 执行视图预测 cell 的持续率 | 执行视图预测值中被反证的比例 |",
         "|---|---|---|---|---|---|---|---|"]
    for g, _ in groups:
        a = S[g]
        if a is None:
            continue
        L.append(f"| {g} | {a['steps']} | {f2(a['best'])} | {f2(a['top1_pred'])} / {f2(a['top1_union'])} / {f2(a['top1_refute'])} | "
                 f"{f2(a['gain_pred'])} / {f2(a['gain_union'])} / {f2(a['gain_refute'])} | {f2(a['rho_pred'])} / {f2(a['rho_union'])} / {f2(a['rho_refute'])} | {f2(a['persist'])} | {f2(a['refuted_share_exec'])} |")
    open(os.path.join(EX, "results", "REFUTE_TABLES.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-10:]))


if __name__ == "__main__":
    main()
