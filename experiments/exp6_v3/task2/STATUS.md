# EXP6 V3 Task 2 — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 17:58（E1 固定状态完成；G3 队列运行中）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| G3 | Task 2 闭环队列 `work_dirs/exp6_v3/task2/g3_jobs.txt`：正式 B2 + S1c、T1 + S1c（Macarons++ 各 30 条）→ 探索性 T1 + S2S3、B2 + S2S3（MP3D 各 10 条）→ R2 B2 + frontier bonus / trigger（MP3D 各 10 条）→ 室外覆盖率 60 条 | gpu117 + gpu115 各 4 卡 | 3429774 / 3435033 | `work_dirs/exp6_v3/task2/g3/logs/` | 17:57 | 约 19:15 |
| R1 | `e2_r1_region.py`（Pool CL 368 状态，CPU，重跑） | CPU | — | `work_dirs/exp6_v3/task2/exp2/logs/r1_region.log` | 17:52 | 约 18:05 |

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
| E1.4 | G3：正式 Macarons++ B2 + S1c / T1 + S1c；探索性 MP3D T1 + S2S3 / B2 + S2S3（`configs/G3_ARMS.json`）→ `E1_G3_RESULTS.md`、`E1_FINAL_REPORT.md` | 🔄 17:57 |

## E2 Experiment 2 — MP3D Long-Range Exploration
| 编号 | 内容 | 状态 |
|---|---|---|
| E2.0 | 规划审计 `E2_PLANNING_AUDIT.md` | ✅ 17:20 |
| E2.1 | R1 评分区域 counterfactual（离线 GT；Pool CL，偏差 T2-1） | 🔄 17:52 重跑 |
| E2.2 | R3 候选空间 counterfactual（`e2_r3_candidates.py`，368 状态，4 卡 10 min） | ✅ 17:40 |
| E2.3 | R2 frontier 图 + 进展项（`common/t2_frontier.py`，单测 1 passed）；离线评估 `e2_r2_offline.py` ✅ 17:35（λ = 119.1 / θ = 75.3 预登记）；G3 MP3D bonus / trigger 各 10 条在队列 | 🔄 |
| E2.4 | R4 组合（仅有证据的模块） | ⬜ |
| E2.5 | 报告 + 总报告 + 提交 | ⬜ |
