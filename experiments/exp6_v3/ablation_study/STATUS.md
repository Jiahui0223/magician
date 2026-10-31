# EXP6 V3 Ablation — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 16:50（Experiment C 全部完成）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| — | **无运行中任务**：Experiment C 的闭环（16:24）、覆盖率回放（16:27）、分析与报告（16:45）均已完成；GPU 分配空闲（gpu117 3429774 至 11-01 14:50、gpu115 3435033 至 11-01 10:09） | — | — | — | — | — |

## AB0 审计与协议 — ✅
| 编号 | 内容 | 状态 |
|---|---|---|
| AB0.1 | 资源核实：3429774（gpu117，空闲）、3435033（gpu115，空闲）、3435072（排队，不用）；git 工作区干净（`40ceaa7`） | ✅ 09:30 |
| AB0.2 | 缓存审计 `experiment_A_gtdepth_oracle/report/CACHE_REUSE_AUDIT.md`：Pool CL 1,461 状态 oracle / t\* / GTD / V2 样本全部可逐位复用，svc 含完整深度 → D100 重放可行；Pool FX 365 状态缺 GTD 与原始样本 | ✅ 09:45 |
| AB0.3 | `protocol.md` v1.0 冻结（A 三路径、B 评分器 α=0.5 / λ=1.0、预注册 Gate、停滞 / 高 regret 定义、必过检查） | ✅ 09:55 |

## A Experiment A — ✅（10:35）
| 编号 | 内容 | 状态 |
|---|---|---|
| A1 | `shared/ab_score.py` 实现 + 1 状态冒烟 + 单测 | ✅ 09:45：三路冒烟通过，B2 / E2D / T0 / T1 在 EXP6 / E2 已有状态上逐位一致；GTD 与 EXP5 逐位一致；`tests/` 5 passed（指标 / 选择规则 / DepthU） |
| A2 | Pool FX 打分（VALSEL 9 ep + Dev-3 B2 15 ep；回归 vs EXP6 / E2 行）| 🔄 09:46 起；09:53 已 90 + 121 状态；✅ 10:00 完成 90 + 125 状态；209 / 215 逐位，6 个差 1e-12（偏差 #1：数值等同） |
| A3 | Pool CL 打分（室内 40 + 室外 120 来源 episode；Dev-3 B0 回归） | ✅ 10:22：368 + 1,093 状态；Dev-3 B0 回归 147 / 150 逐位 + 3 个 1e-12（偏差 #1）；GTD 与 EXP5 逐位核对见 `A_G0_VALIDATION.md`；8 卡 ≈ 4.8 GPU-h |
| A4 | `a_analyze.py`：A1 / A2 / A3 表、拆分、停滞、高 regret、CI、图 → `A_FIXED_STATE.md`、`A_FAILURE_CASES.md`、`A_G0_VALIDATION.md` | ✅ 10:30：CL MP3D regret B2 0.112 / T1 0.114 / GTD 0.064（Δ/r_pred ≈ 0.43）；CL Macarons++ B2 0.468 / **T1 0.319** / GTD 0.160（B2 Δ/r_pred 0.66 → T1 0.50）；停滞 53 状态 G\* 中位 0.019 pp（候选空间 / 区域限制）；高 regret 状态 B2 20% / T1 16%，GTD 选对 57% / 48% |
| A5 | 阶段报告一 | ✅ 10:35 |

## B Experiment B — ✅（阴性结果，10:40）
| 编号 | 内容 | 状态 |
|---|---|---|
| B1 | `b_scorer.py` 纯函数 + 单测 + `B_SCORE_AUDIT.md` | ✅ 10:00（2 passed；审计随 B2 生成） |
| B2 | 固定状态 G2（P0 / P1 / P2 / RiskAware；B2、T1 主，T0 / E2D 对照）→ `B_FIXED_STATE.md` | ✅ 10:33：P1 使 regret +65% ~ +200%（所有场景）；P2（λ=1）MP3D B2 −3.6% / T1 +40%，Macarons++ B2 +20% / T1 +5%；唯一正信号 T1 Macarons++ λ=0.5 −0.023 [−0.051, −0.001]（非冻结设置） |
| B3 | Gate 判定 → `B_DECISION.md`；通过则 G3 闭环 → `B_CLOSED_LOOP.md` | ✅ 10:40：8 / 8 组合 FAIL → 不进入 G3，也不跑 exploratory 闭环（`B_CLOSED_LOOP.md` = Not Run） |
| B4 | 阶段报告二 | ✅ 10:45 |

## R 总报告 — ✅（10:50）
`reports/V3_ABLATION_SUMMARY.md`、`V3_ABLATION_COMPARISON_TABLES.md`（表 1–7，含 G3 参照与 MAG-G / MAG-B）、`V3_ABLATION_DEVIATIONS.md`（#1）、`V3_ABLATION_RESOURCES.md`（≈ 4.8 GPU-h）；提交 `[V3][ablation]`（A：`c170827`；B + 总报告：见最新提交）。

## C Experiment C — GT-Depth Closed-Loop（任务书 2026-10-31 15:20；`experiment_C_gtdepth_closed_loop/`）
| 编号 | 内容 | 状态 |
|---|---|---|
| C0.1 | 协议审计（t23 / v2_sim / t25 / s1_oracle / ab_score 的 GT 路径、候选、选择规则、覆盖定义）+ 资源核实（3429774 gpu117、3435033 gpu115 空闲；3435072 排队不用）| ✅ 15:35 |
| C0.2 | `protocol.md` v1.0 起草（C1 / C2 定义、GT 旁路 cq 文件、诊断、统计、G0 九项、资源）| ✅ 15:40（G0 通过后冻结 `configs/FROZEN_SHA256.json` 15:45）|
| C0.3 | 代码：`c_common / c_gt / c_in_sim / c_in_planner / c_out_sim / c_out_planner / c_scoring / c_out_cov / run_queue_c.sh`；CPU 单测 5 passed | ✅ 15:40 |
| C0.4 | **G0**（`tests/g0_run.sh` + `g0_check.py` → `report/C_G0_VALIDATION.md`）：9 / 9 PASS（t\* 与 EXP5 逐位、GTD 与 Experiment A 逐位、增益与 oracle.npz 逐位、无泄漏、合法性、单调、确定性、B2 回放逐位（室内 5 步 / 室外 3 步）、冒烟 + 在线单调覆盖 = t25 回放）。第 1 轮唯一问题：子进程 P1 全量哈希每步 15 s → 改为 GPU 校验和（不影响分数）；室外最后一步执行一致性检查补上后第 2 轮 9 / 9 | ✅ 15:44 |
| C1 | C1 GT depth + DistP1 全部 40 条闭环（10 MP3D × 150 步 + 30 Macarons++ × 100 步，含每步诊断）| ✅ 16:10（15:45 起，8 卡；室内均 387 s / 条、室外均 241 s / 条 ≈ 3.1 GPU-h；1 次失败 = 偏差 C-1（bridge 起点 0 的 t=0 覆盖为 0 使状态哈希崩溃，修复后从头重跑 ✅）；室外覆盖率回放 16:12 起另开 2 worker）|
| C2 | C2 GT gain greedy 全部 40 条 | ✅ 16:24（室内均 112 s / 条、室外均 196 s / 条，0 失败）|
| C3 | 室外覆盖率（`c_out_cov.py`，60 条）| ✅ 16:27（官方曲线回放 60 / 60 逐位相等；在线单调覆盖 = 回放）|
| C4 | `c_analyze.py`：9 组配对统计、曲线、停滞 / 诊断汇总 → `report/{C_GTDEPTH_G3, C_ORACLE_GAIN_G3, C_FAILURE_CASES, C_FINAL_REPORT}.md`；更新 `reports/V3_ABLATION_*.md`（表 8–10、总结 §7–8、偏差 C-1、资源）| ✅ 16:45：C1 − B2 AUC +0.023 / +0.011（CI 含 0）、Final −0.017 / −0.012 → 退化；C2 − C1 MP3D Final +0.101 [+0.043, +0.195]、Macarons++ AUC +0.030 [+0.007, +0.052] → DistP1 是主要损失；C2 MP3D Final 仍只 0.615 → 候选 / 视野上限 |
| C5 | 阶段回复（C1 完成后 16:25）→ 自动继续 C2 → 总报告 → `[V3][ablation-c]` 提交 | ✅ |
