"""Camera Pose C2 -- Stage A analysis (CPU): oracle best one-step gain of C0 vs C2a / C2b / C2c (and ALL, diagnostic) on the
90 MP3D VALSEL development states, the pre-registered Stage A gate (C2/protocol.md §2) and the diagnostics.
Eligible set = the planner's prefer-unvisited rule on each variant's legal list (fine-lattice indices). Output
C2/reports/stage_a.{json,md}. Usage: python c2_analyze_oracle.py"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C2 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(C2)))
sys.path.insert(0, HERE)
import c2_poses as CP  # noqa: E402

ORC = os.path.join(REPO, "work_dirs", "exp_camera_pose", "C2", "oracle")
VARS = ("C0", "C2a", "C2b", "C2c", "ALL")


def eligible(cands, hist):
    h = {tuple(x) for x in hist[:-1]}
    e = [c for c in cands if tuple(c["f"]) not in h]
    return e or cands


def phase(t):
    return "early" if t <= 25 else ("middle" if t <= 55 else "late")


def main():
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ORC, "*.json")))]
    rows = []
    for s in S:
        r = {"sid": s["sid"], "tag": s["tag"], "house": s["scene"].split("_")[0], "t": s["t"], "phase": phase(s["t"]), "cov_t": s["cov_t"],
             "gen": s["n_generated"], "oob": s["n_out_of_bounds"], "blk": s["n_blocked"]}
        for v in VARS:
            L = [c for c in s["candidates"] if c["family"] in CP.VARIANTS[v]]
            E = eligible(L, s["executed_history_fine"])
            b = max(E, key=lambda c: (c["gain"], -E.index(c)))
            r[v] = {"n_legal": len(L), "n_elig": len(E), "best": 100 * b["gain"], "best_family": b["family"], "best_trans": b["trans_m"], "best_dyaw": b["dyaw"],
                    "best_dpitch": b["dpitch"], "best_on_lattice": b["on_original_lattice"], "best_f": b["f"]}
        rows.append(r)
    out = {"n_states": len(rows), "variants": {}}
    c0 = np.asarray([r["C0"]["best"] for r in rows])
    out["C0_oracle_mean_pp"] = float(c0.mean())
    thr = max(0.10, 0.10 * float(c0.mean()))
    out["gate_threshold_pp"] = thr
    for v in VARS:
        b = np.asarray([r[v]["best"] for r in rows])
        d = b - c0
        o = {"oracle_mean_pp": float(b.mean()), "delta_mean_pp": float(d.mean()), "delta_rel": float(d.mean() / c0.mean()), "delta_median_pp": float(np.median(d)),
             "share_improved": float((d > 1e-12).mean()), "n_legal_mean": float(np.mean([r[v]["n_legal"] for r in rows])),
             "n_elig_mean": float(np.mean([r[v]["n_elig"] for r in rows])), "best_trans_mean_m": float(np.mean([r[v]["best_trans"] for r in rows])),
             "best_from_new_share": float(np.mean([r[v]["best_family"] != "C0" for r in rows])),
             "best_family": {f: float(np.mean([r[v]["best_family"] == f for r in rows])) for f in ("C0", "C2a", "C2b", "C2c")}}
        srt = np.sort(d)[::-1]
        k = max(1, int(round(0.1 * len(d))))
        o["top10pct_share_of_total"] = float(srt[:k].sum() / d.sum()) if d.sum() > 0 else None
        o["by_phase_delta_pp"] = {p: float(np.mean([x for x, r in zip(d, rows) if r["phase"] == p])) for p in ("early", "middle", "late")}
        o["by_house_delta_pp"] = {h: float(np.mean([x for x, r in zip(d, rows) if r["house"] == h])) for h in sorted({r["house"] for r in rows})}
        new = [r[v] for r in rows if r[v]["best_family"] != "C0"]
        if new:
            o["new_best_dyaw_hist"] = {str(int(x)): int(sum(1 for z in new if round(z["best_dyaw"]) == x)) for x in sorted({round(z["best_dyaw"]) for z in new})}
            o["new_best_dpitch_hist"] = {str(int(x)): int(sum(1 for z in new if round(z["best_dpitch"]) == x)) for x in sorted({round(z["best_dpitch"]) for z in new})}
            o["new_best_trans_mean_m"] = float(np.mean([z["best_trans"] for z in new]))
            o["new_best_off_lattice_share"] = float(np.mean([not z["best_on_lattice"] for z in new]))
        c0t = float(np.mean([r["C0"]["best_trans"] for r in rows]))
        o["gate"] = {"mean_delta_ok": bool(d.mean() >= thr), "robust_ok": bool((d > 1e-12).mean() >= 0.25 and (o["top10pct_share_of_total"] or 1.0) <= 0.60),
                     "cost_ok": bool(o["best_trans_mean_m"] <= c0t + 0.25)}
        o["gate"]["pass"] = all(o["gate"].values()) if v not in ("C0", "ALL") else None
        out["variants"][v] = o
    gen = sum(r["gen"] for r in rows)
    out["generation"] = {"generated": gen, "out_of_bounds_rate": sum(r["oob"] for r in rows) / gen, "collision_rejected_rate": sum(r["blk"] for r in rows) / gen}
    # joint vs single: per state, does C2c beat both C2a and C2b?
    out["joint_vs_single"] = {"C2c_gt_max_C2a_C2b_share": float(np.mean([r["C2c"]["best"] > max(r["C2a"]["best"], r["C2b"]["best"]) + 1e-9 for r in rows])),
                              "C2c_minus_max_C2a_C2b_mean_pp": float(np.mean([r["C2c"]["best"] - max(r["C2a"]["best"], r["C2b"]["best"]) for r in rows]))}
    out["stage_a_pass_variants"] = [v for v in ("C2a", "C2b", "C2c") if out["variants"][v]["gate"]["pass"]]
    os.makedirs(os.path.join(C2, "reports"), exist_ok=True)
    json.dump({"summary": out, "rows": rows}, open(os.path.join(C2, "reports", "stage_a.json"), "w"), indent=1, default=float)
    L = [f"# C2 Stage A —— 固定状态 Oracle（MP3D VALSEL {len(rows)} 态；候选集 = 规划器优先未访问规则后的可选集；GT 只用于评估与运动约束）", "",
         f"C0 oracle 均值 {out['C0_oracle_mean_pp']:.3f} pp（应等于历史 1.099）；放行阈值 mean Δ ≥ {thr:.3f} pp。生成 {gen} 个 ALL 候选：越界 {out['generation']['out_of_bounds_rate']:.3f}，GT 碰撞剔除 {out['generation']['collision_rejected_rate']:.3f}。", "",
         "| 变体 | 合法 / 可选候选 | Oracle best（pp） | Δ vs C0（pp） | 相对 | 中位 Δ | 改善的状态 | top-10% 份额 | 最优来自新候选 | 最优平移（m） | 门槛 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in VARS:
        o = out["variants"][v]
        top10 = "—" if o["top10pct_share_of_total"] is None else f"{o['top10pct_share_of_total']:.2f}"
        gt = "—" if o["gate"]["pass"] is None else ("✅ 通过" if o["gate"]["pass"] else "❌ " + ", ".join(k for k, x in o["gate"].items() if k != "pass" and not x))
        L.append(f"| {v} | {o['n_legal_mean']:.1f} / {o['n_elig_mean']:.1f} | {o['oracle_mean_pp']:.3f} | {o['delta_mean_pp']:+.3f} | {o['delta_rel']:+.1%} | {o['delta_median_pp']:+.3f} | {o['share_improved']:.2f} | "
                 f"{top10} | {o['best_from_new_share']:.2f} | {o['best_trans_mean_m']:.2f} | {gt} |")
    L += ["", "**分阶段 Δ（pp）**：" + "；".join(f"{v}: " + ", ".join(f"{p} {x:+.3f}" for p, x in out["variants"][v]["by_phase_delta_pp"].items()) for v in ("C2a", "C2b", "C2c")), "",
          "**分房屋 Δ（pp）**：" + "；".join(f"{v}: " + ", ".join(f"{h} {x:+.3f}" for h, x in out["variants"][v]["by_house_delta_pp"].items()) for v in ("C2a", "C2b", "C2c")), "",
          f"**联合 vs 单独**：C2c 最优严格大于 max(C2a, C2b) 的状态占 {out['joint_vs_single']['C2c_gt_max_C2a_C2b_share']:.2f}，均值差 {out['joint_vs_single']['C2c_minus_max_C2a_C2b_mean_pp']:+.3f} pp。", ""]
    for v in ("C2a", "C2b", "C2c", "ALL"):
        o = out["variants"][v]
        if "new_best_dyaw_hist" in o:
            L.append(f"- {v} 最优为新候选时：|Δyaw| 分布 {o['new_best_dyaw_hist']}，|Δpitch| 分布 {o['new_best_dpitch_hist']}，平均平移 {o['new_best_trans_mean_m']:.2f} m，落在原格外 {o['new_best_off_lattice_share']:.2f}；该变体下最优候选的族分布 {o['best_family']}")
    L += ["", f"**Stage A 通过的变体**：{out['stage_a_pass_variants'] or '无'}"]
    open(os.path.join(C2, "reports", "stage_a.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({v: {k: out["variants"][v][k] for k in ("oracle_mean_pp", "delta_mean_pp", "share_improved", "top10pct_share_of_total", "best_trans_mean_m", "gate")} for v in VARS}, indent=1, default=float))
    print("C0 mean", out["C0_oracle_mean_pp"], "pass", out["stage_a_pass_variants"])


if __name__ == "__main__":
    main()
