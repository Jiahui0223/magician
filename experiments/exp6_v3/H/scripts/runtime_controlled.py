"""EXP6 V3 / H -- reports/RUNTIME_CONTROLLED.md (CPU; generated). barts_s0 re-run of H0-H3, one episode per idle L40 (gpu115,
nothing else running on the node's GPUs), same chain as G3: per-step planner decision time (median / P90), cumulative
planning time after 1 / 5 / 20 / 100 steps, policy share of the step (H1-H3: time outside the frozen score_state), planner
peak GPU memory, H3 extra second-layer views per step; reproducibility vs the Table 1 records (executed keys / official
coverage) -- H0 vs the V3 record, H1-H3 vs V3-H1 / V3-H2 / V3-H3."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(H)))
TW = os.path.join(R, "work_dirs", "exp6_v3", "H", "timing")
REF = {"h0": "V3", "h1": "V3-H1", "h2": "V3-H2", "h3": "V3-H3"}


def main():
    L = ["# RUNTIME_CONTROLLED — 受控计时（barts_s0；每个策略独占一张空闲的 L40；自动生成，`scripts/runtime_controlled.py`）", "",
         "| 策略 | 每步决策 中位数 / P90（s） | 累计规划时间 @1 / @5 / @20 / @100 步（s） | 策略额外开销 中位数（s / 步） | 规划器峰值显存（GB） | H3 每步第二层唯一视角数 | 整个 episode 墙钟（s） | 与 Table 1 记录逐位一致（执行动作 / 官方覆盖） |",
         "|---|---|---|---|---|---|---|---|"]
    out = {}
    for p, mid in REF.items():
        eps = glob.glob(os.path.join(TW, p, "V2M", "episodes", "ep_V2M_barts_s0_r0.json"))
        if not eps:
            L.append(f"| {p} | 未完成 | | | | | | |")
            continue
        e = json.load(open(eps[0]))
        st = [json.loads(l) for l in open(glob.glob(os.path.join(TW, p, "V2M", "planner_logs", "*.steps.jsonl"))[0])]
        dec = np.asarray([x["decision_s"] for x in st])
        cum = np.cumsum(dec)
        pol = [x["timing"].get("policy_s") for x in st if x.get("timing") and x["timing"].get("policy_s") is not None]
        h3 = [x["all_scores"]["h3"]["n_c2_unique"] for x in st if x.get("all_scores", {}).get("h3")]
        ref = json.load(open(os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes", mid, "barts_s0.json")))
        rk, rc = e["executed_keys"] == ref["executed_keys"], e["coverage"] == ref["coverage_official"]
        out[p] = {"decision_s_median": float(np.median(dec)), "decision_s_p90": float(np.quantile(dec, 0.9)), "cum": [float(cum[i]) for i in (0, 4, 19, 99)],
                  "policy_s_median": float(np.median(pol)) if pol else 0.0, "peak_mem_gb": max(x["peak_mem_gb"] for x in st), "c2": float(np.mean(h3)) if h3 else None,
                  "episode_s": e.get("seconds_this_process"), "reproduces": [rk, rc]}
        o = out[p]
        c2s = "—" if o["c2"] is None else f"{o['c2']:.1f}"
        L.append(f"| {p.upper()} | {o['decision_s_median']:.2f} / {o['decision_s_p90']:.2f} | " + " / ".join(f"{x:.0f}" for x in o["cum"]) +
                 f" | {o['policy_s_median']:.2f} | {o['peak_mem_gb']:.1f} | {c2s} | {o['episode_s']:.0f} | {rk} / {rc} |")
    json.dump(out, open(os.path.join(H, "results", "runtime_controlled.json"), "w"), indent=1)
    if "h0" in out and "h3" in out:
        L += ["", f"**H3 相对 H0**：每步决策时间 ×{out['h3']['decision_s_median'] / out['h0']['decision_s_median']:.2f}；100 步累计规划时间 {out['h0']['cum'][3]:.0f} → {out['h3']['cum'][3]:.0f} s。"]
    open(os.path.join(H, "reports", "RUNTIME_CONTROLLED.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
