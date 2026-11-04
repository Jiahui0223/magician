"""EXP6 V5 / F1 -- reports/RUNTIME.md + results/runtime.json (CPU; generated). Controlled cost: barts_s0 re-run of B0 / A2k / A5k
/ B5k, one episode per idle L40 (timing_runs.sh, 22:56-22:58, nothing else on the node's GPUs): planner decision time per
step (planner-reported: encode + condition + flow sampling + DistP1 scoring), its components on the first steps, simulator
wall time, planner peak GPU memory; parameter counts from the training runs. Reproducibility: executed keys / official
coverage of every re-run vs the G3 record of the same model (B0: vs the Table 1 record)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
W = os.path.join(R, "work_dirs", "exp6_v5", "F1")
PAR = {"B0": 24822114, "A2k": 24822114, "A5k": 24822114, "B5k": 24988647}
COND = {"B0": 76224, "A2k": 76224, "A5k": 76224, "B5k": 242757}     # Ray32/Conv1D RayEncoder vs V5RayDecoder parameters


def main():
    out = {}
    for m in ("B0", "A2k", "A5k", "B5k"):
        ep = json.load(open(os.path.join(W, "g3", "timing", m, "V2M", "episodes", "ep_V2M_barts_s0_r0.json")))
        ref = json.load(open(os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes", "B0", "barts_s0.json"))) if m == "B0" else \
            json.load(open(os.path.join(W, "g3", "episodes", m, "barts_s0.json")))
        st = [x for x in ep["steps"] if x.get("planner")]
        dec = np.asarray([x["planner"]["decision_s"] for x in st if x["planner"].get("decision_s") is not None])
        out[m] = {"decision_s_median": float(np.median(dec)), "decision_s_mean": float(dec.mean()), "decision_s_p90": float(np.quantile(dec, 0.9)),
                  "episode_s": ep.get("seconds_this_process"), "planner_peak_mem_gb": (ep.get("planner_bye") or {}).get("peak_mem_gb"),
                  "params_total": PAR[m], "params_condition_module": COND[m],
                  "reproduces_record": {"executed_keys": ep["executed_keys"] == ref["executed_keys"], "official_coverage": ep["coverage"] == ref["coverage_official"]}}
    json.dump(out, open(os.path.join(F1, "results", "runtime.json"), "w"), indent=1)
    L = ["# RUNTIME — 模型开销（自动生成，`scripts/runtime_report.py`）", "",
         "**受控计时**：",
         "- barts_s0 重跑，每个模型独占一张空闲的 L40（22:56–22:58，同节点没有其他 GPU 任务）；",
         "- 链路与 G3 相同（`sim_out_v5.py`，S10 推理 + S10 评分）。",
         "- **为什么不用 G3 记录里的时间**：G3 主运行与训练、其他 worker 共用 GPU，那里的时间偏大，不能比较。", "",
         "| 模型 | 总参数 | 条件模块参数 | 每步决策 中位数 / 均值 / P90（s） | 整个 episode（s） | 规划器峰值显存（GB） | 与 G3 记录逐位一致（执行动作 / 官方覆盖） |",
         "|---|---|---|---|---|---|---|"]
    for m, r in out.items():
        rr = r["reproduces_record"]
        L.append(f"| {m} | {r['params_total']:,} | {r['params_condition_module']:,} | {r['decision_s_median']:.3f} / {r['decision_s_mean']:.3f} / {r['decision_s_p90']:.3f} | "
                 f"{r['episode_s']:.0f} | {r['planner_peak_mem_gb']:.2f} | {rr['executed_keys']} / {rr['official_coverage']} |")
    b0, b5 = out["B0"], out["B5k"]
    L += ["", f"**F1-B（V5 Query）相对 B0 的开销**：",
          f"- 每步决策时间 ×{b5['decision_s_median'] / b0['decision_s_median']:.2f}（中位数 {b0['decision_s_median']:.3f} → {b5['decision_s_median']:.3f} s）；",
          f"- 规划器峰值显存 {b0['planner_peak_mem_gb']:.2f} → {b5['planner_peak_mem_gb']:.2f} GB；",
          f"- 参数 +{b5['params_total'] - b0['params_total']:,}（条件模块 {b0['params_condition_module']:,} → {b5['params_condition_module']:,}）。", "",
          "**训练开销**（`reports/TRAINING.md`）：",
          "- F1-A：约 0.35 s/步，4 × L40，14.3 GB / rank；",
          "- F1-B：第一阶段 0.49 s/步、3.6 GB / rank；第二阶段 0.78 s/步、12.4 GB / rank；",
          "- 5k 步墙钟（含每 1000 步的 EMA 评估）：A 49 min（21:29–22:18），B 78 min（21:29–22:47）。"]
    open(os.path.join(F1, "reports", "RUNTIME.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[6:12]))


if __name__ == "__main__":
    main()
