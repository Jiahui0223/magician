# EXP6 V3 Task 2 — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 19:15（Task 2 全部完成；无运行中任务）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| — | **无运行中任务**：G3 队列 17:51 → 19:00（100 条闭环 + 60 条覆盖率，0 失败）；分析与报告 19:10 完成；GPU 分配空闲 | — | — | — | — | — |

## P0 审计与协议
| 编号 | 内容 | 状态 |
|---|---|---|
| P0.1 | 缓存审计：Experiment A 缓存含 B2 / E2D / T0 / T1 每候选每样本的解码深度 `t`（CL 1,461 + FX 365 状态，4.0 GB）、EXP5 / E2 oracle（增益、t\*、候选采样点）；资源核实 gpu117 3429774（11-01 14:50）、gpu115 3435033（11-01 10:09）空闲，3435072 排队不用 | ✅ 17:00 |
| P0.2 | DistP1 审计（`exp1_coverage_scoring/report/E1_DISTP1_AUDIT.md`）：新颖性尺度（1 体素膨胀 vs eps 5–13 体素，BAND 只沿 10% 射线）、z0 上限（MP3D 4.7–13 m，Macarons++ 射程的 22–71%）、无入射角补偿、区域裁剪；覆盖率口径 = 历史 10% 采样点的 eps 邻域 | ✅ 17:30 |
| P0.3 | `protocol.md` v1.0（评分器族、dev / eval 划分、G2 Screening 与选择规则、G3、E2 R0–R4）；`configs/FROZEN_SHA256_E1.json` | ✅ 17:10 冻结 |

## E1 Experiment 1 — Coverage-Consistent Scoring
| 编号 | 内容 | 状态 |
|---|---|---|
| E1.1 | `common/t2_scorers.py` + 单测（S0 = 冻结 DistP1 逐位；面积 / 入射角 / 新颖性 / 一致性合成检查）6 passed；2 状态冒烟：GT S0 = Experiment A 逐位、B2 S0 = 缓存逐位，2 s / 状态，峰值 11.8 GB | ✅ 17:08 |
| E1.2 | 固定状态打分 1,676 状态（8 卡，17:10–17:43 + 7 状态补算 17:48；≈ 4.5 GPU-h 含 D100 重放）；S0 复现：GT 1,653 / 1,676 逐位（其余 ≤ 1.8e-14）、B2 / T1 1,652 逐位 | ✅ 17:48 |
| E1.3 | `e1_analyze.py` → `E1_GT_SCORING.md`、`E1_MODEL_SCORING.md`、`E1_FAILURE_CASES.md`；Gate：Macarons++ S1c 两模型通过（B2 −12.6%、T1 −24.5%）；MP3D T1 通过 S2h / S3 / S2S3（−24%），B2 无（偏差 T2-4）→ 阶段报告一 | ✅ 17:55 |
| E1.4 | G3：正式 Macarons++ B2 + S1c 0.715 / 0.890（vs B2 0.730 / 0.906，退化）、T1 + S1c 0.719 / 0.879（vs T1 0.712 / 0.883，无差异）；探索性 MP3D **T1 + S2S3 0.354 / 0.530**（vs T1 0.331 / 0.508，+0.023 / +0.022，停滞 5 → 2）、B2 + S2S3 0.323 / 0.472（退化）→ `E1_G3_RESULTS.md`、`E1_FINAL_REPORT.md` | ✅ 19:05 |

## E2 Experiment 2 — MP3D Long-Range Exploration
| 编号 | 内容 | 状态 |
|---|---|---|
| E2.0 | 规划审计 `E2_PLANNING_AUDIT.md` | ✅ 17:20 |
| E2.1 | R1 评分区域 counterfactual（Pool CL 368 状态）：GT 深度 Top-1 n_win 5 / 7 / 9 / 整层 = 0.757 / 0.845 / 0.894 / 0.724 → `E2_ROI_COUNTERFACTUAL.md` | ✅ 18:05 |
| E2.2 | R3 候选空间 counterfactual：两步可达集最大增益 2.3×（停滞态 0.022 → 0.121 pp），98% 在已知格 → `E2_CANDIDATE_COUNTERFACTUAL.md` | ✅ 17:40 / 18:05 |
| E2.3 | R2 frontier：离线 ✅ 17:35；G3 MP3D bonus 0.272 / 0.345、trigger 0.282 / 0.348（vs B2 0.334 / 0.530，0/5 楼层，停滞 9 / 8）→ 阴性 → `E2_FRONTIER_EXPLORATION.md`、`E2_G3_RESULTS.md` | ✅ 19:05 |
| E2.4 | R4 组合：无 ≥ 2 个有正向证据的模块 → 不建 | ✅（不执行）|
| E2.5 | `E2_FAILURE_CASES.md`（6 例 + 图）、`E2_FINAL_REPORT.md`、`reports/TASK2_{COMBINED_REPORT, COMPARISON_TABLES, DEVIATIONS, RESOURCES}.md`；提交 `[V3][task2]` × 4 | ✅ 19:15 |
