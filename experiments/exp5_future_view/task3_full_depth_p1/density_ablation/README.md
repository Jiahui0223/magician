# Task 3 / E1.2 — BAND Density Ablation（固定状态，2026-10-28）

| 文件 | 内容 |
|---|---|
| `protocol.md` | v1.0，运行前冻结（组合、分层、阶段、指标、闭环筛选与推荐规则） |
| `reports/T3_E12_DENSITY_AUDIT.md` | 代码与数据审计、修改范围、运行计划 |
| `reports/T3_E12_INPUT_DENSITY.md` | E1.2-A 输入密度消融（几何、预测、排序、筛选、成本） |
| `reports/T3_E12_SCORING_DENSITY.md` | E1.2-B 评分密度消融与新颖性掩码 |
| `reports/T3_E12_PHANTOM_SCORING.md` | E1.2-C 伪表面定义审计与评分错误归因 |
| `reports/T3_E12_FINAL_DECISION.md` | 结论、Q1–Q6、E2 密度推荐（D100） |
| `reports/T3_E2_PILOT_PLAN.md` | E2 Pilot 方案（待批准，未训练） |
| `scripts/` | `e12_density.py`（嵌套密度 P1）、`e12_paths.py`、`e12_states_b2.py`、`e12_oracle_b2.py`、`e12_fixed.py`、`e12_analyze.py`、`e12_viz.py`、`e12_truth_audit.py` |
| `analysis/` | `e12_summary.json`、`e12_states.json`、`tables.md`、`e12_truth_audit.json` |
| `figures/` | Figure 1 / 1b / 2 / 2b / 6 与按规则选取状态的 Figure 3 / 4 / 5 |

大文件在 `work_dirs/exp5_future_view/task3_full_depth_p1/e12/`（B2 状态清单与 oracle、固定状态结果、日志）。

## 结论
- 几何在 50% 基本饱和。
- 输入密度：Macarons++ 越密越好，MP3D 无差别；50% 和 75% 都没有通过冻结的闭环筛选，**不跑闭环**。
- 评分密度：稀疏 vs 非稀疏的阶跃，室内有益、室外有害。
- **E2 推荐密度 D100**（尚未证明最优）。

## 变更记录
- 2026-10-28：创建；固定状态实验、分析、图、报告完成；E2 方案待批准。
