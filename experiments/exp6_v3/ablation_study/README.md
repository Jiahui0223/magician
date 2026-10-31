# EXP6 V3 — Ablation Study（Experiment A GT-Depth Oracle Decomposition；Experiment B Depth Uncertainty Planner；Experiment C GT-Depth Closed-Loop）

任务书：`../EXP6_V3_ABLATION_ONLY_PLAN.md`（2026-10-31；V4 暂停，先做诊断）+ Experiment C 任务书（2026-10-31 15:20，用户指令）。协议：`protocol.md` v1.0（冻结）。状态板：`STATUS.md`（宏观板 `../STATUS.md` 的 P6 段指向这里）。

| 目录 | 内容 |
|---|---|
| `shared/ab_score.py` | 两个状态池 × 四个模型（B2 / E2D / T0 / T1）的 D100 + 稀疏重放、K=4 原始样本、DistP1、GT 深度 DistP1、回归比对 → `work_dirs/exp6_v3/ablation_study/cache/<pool>/<dataset>/<sid>.{npz,json}` |
| `shared/ab_data.py` | 读取两个池（EXP5 历史 + 新缓存）为统一记录；选择规则、指标、bootstrap |
| `experiment_A_gtdepth_oracle/` | `scripts/a_analyze.py`（A1 / A2 / A3 表、拆分、停滞 / 高 regret 诊断、图）、`report/` |
| `experiment_B_depth_uncertainty/` | `scripts/b_scorer.py`（P0 / P1 / P2 / RiskAware 纯函数）、`scripts/b_analyze.py`、`report/` |
| `experiment_C_gtdepth_closed_loop/` | 2026-10-31 15:20 任务书：C1 GT depth + DistP1、C2 GT 单步增益 greedy 的完整 G3 闭环（自己的 `protocol.md`、`scripts/`、`tests/`、`report/C_{G0_VALIDATION,GTDEPTH_G3,ORACLE_GAIN_G3,FAILURE_CASES,FINAL_REPORT}.md`）；输出 `work_dirs/exp6_v3/ablation_study/experiment_C/` |
| `tests/` | 选择规则 / 指标 / DepthU 纯函数单测；1 状态冒烟 |
| `reports/` | `V3_ABLATION_SUMMARY.md`、`V3_ABLATION_COMPARISON_TABLES.md`、`V3_ABLATION_DEVIATIONS.md`、`V3_ABLATION_RESOURCES.md` |

复用：EXP5 A1 / A2 的状态、oracle、t\*、GTD、V2O / V2M 样本（`experiment_A_gtdepth_oracle/report/CACHE_REUSE_AUDIT.md`）。历史数字一律标 `historical EXP5, not new V3 result`。
