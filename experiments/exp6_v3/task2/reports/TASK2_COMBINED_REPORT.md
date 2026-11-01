# EXP6 V3 Task 2 — Coverage-Aware NBV Planning：总报告

> 状态：**COMPLETED**（2026-10-31 17:00–19:10；用户任务书 17:00）。协议 `../protocol.md` v1.0；状态板 `../STATUS.md`；偏差 `TASK2_DEVIATIONS.md`（T2-1…T2-4）；资源 `TASK2_RESOURCES.md`（≈ 15 GPU-h，无训练）；统一表 `TASK2_COMPARISON_TABLES.md`（表 1–5）。Experiment 1：`../exp1_coverage_scoring/report/E1_{DISTP1_AUDIT, GT_SCORING, MODEL_SCORING, G3_RESULTS, FAILURE_CASES, FINAL_REPORT}.md`；Experiment 2：`../exp2_long_range/report/E2_{PLANNING_AUDIT, ROI_COUNTERFACTUAL, FRONTIER_EXPLORATION, CANDIDATE_COUNTERFACTUAL, G3_RESULTS, FAILURE_CASES, FINAL_REPORT}.md`。不训练、不改 Task0 / Task1 / Ablation A / B / C 的冻结结果；V4 未启动。评估楼层 / Test-3 非盲，全部结论 exploratory。

## 1. 一句话结论
| 问题 | 结论 |
|---|---|
| E1：更贴近真实覆盖的评分器能否改善排序与闭环？ | 排序能：GT 深度下 MP3D 用覆盖率尺度的新颖性（S2h）把被选动作的真实增益从 1.038 升到 1.079 pp（上限 1.099，Top-1 0.63 → 0.77），Macarons++ 用去掉 z0 上限的面积权重（S1）从 1.781 升到 1.869 pp（上限 1.999）。**闭环只在一处兑现**：MP3D 上 V3-T1 + S2S3 的 AUC 0.354 / Final 0.530（vs T1 0.331 / 0.508；停滞 5 → 2 条），与 GT 深度规划器 C1 持平（探索性）。室外正式 G3（S1c）两个模型都没有改善（B2 退化 −0.015 AUC，T1 +0.007 不显著）。 |
| E2：MP3D 的远距离探索 | 区域裁剪是 GT 深度一步排序错误的主要来源（n_win 5 → 9：Top-1 0.757 → 0.894；32% 的最优增益在区域外）；两步视野能把可得增益翻倍（停滞态 0.022 → 0.121 pp），但两步最优位姿几乎都在已知格，是"移动 + 转身"而非进入新区域；**frontier 启发式（bonus / trigger）在闭环上严重退化**（AUC 0.272 / 0.282 vs 0.334，Final 0.345 / 0.348 vs 0.530）。无组合规划器可建。 |
| 一步 vs 闭环 | 第三次独立证实（Experiment C R4、E1 室外、E1 室内 B2）：固定状态的一步排序改善不可靠地转化为 AUC / Final；只有当改动同时减少停滞（T1 + S2S3：停滞 5 → 2）时才看到闭环收益。 |

## 2. Experiment 1 摘要（AUC / Final / 被选动作真实增益）
| 阶段 | MP3D | Macarons++ |
|---|---|---|
| 审计 | 新颖性尺度（1 体素 vs eps 7.8–12.8 cm）、z0 上限（4.7–13 m）、无 cosθ、区域裁剪 | 同；z0 = 射程的 22–71% |
| E1-GT（dev） | S0 1.038 → **S2h 1.079** pp（上限 1.099） | S0 1.781 → **S1a 1.869** / S1aS2S3 1.889 pp（上限 1.999） |
| E1-Pred（dev） | T1 0.944 → S2S3 0.982；B2 0.882 → S2S3 0.906（一楼层明显恶化） | T1 1.627 → S1a 1.724；B2 1.458 → S1c 1.522 |
| Gate | T1 通过 S2h / S3 / S2S3；B2 无 → 无正式 G3 | S1c 两模型通过 → 正式 G3 |
| G3 | **T1 + S2S3 0.354 / 0.530**（+0.023 / +0.022 vs T1；探索性）；B2 + S2S3 0.323 / 0.472（退化） | B2 + S1c 0.715 / 0.890（−0.015 / −0.016，退化）；T1 + S1c 0.719 / 0.879（+0.007 / −0.003，无差异） |
| 成本 | 新评分 +0.5 s / 步 | +0.3 s / 步 |

## 3. Experiment 2 摘要（MP3D）
| 模块 | 证据 | 结论 |
|---|---|---|
| R0 回放 | S0 经补丁逐位复现 B2（室内 5 步 / 室外 3 步） | 通过 |
| R1 评分区域（离线，GT） | n_win 5 / 7 / 9 / 整层：Top-1 0.757 / 0.845 / 0.894 / 0.724；剩余表面在区域内 0.26 / 0.44 / 0.60 | 区域是主要限制；需在模型输入侧扩大（另设实验） |
| R2 frontier（无 GT） | 闭环 AUC 0.272（bonus）/ 0.282（trigger）vs 0.334；Final 0.345 / 0.348 vs 0.530；停滞 9 / 8 条；trigger 55% 步触发 | 阴性；本形式不可用 |
| R3 候选空间（离线，GT） | 两步可达集最大增益 2.3×（停滞态 5.5×），但 98% 在已知格 | 问题是两步视野，不是可达性 |
| R4 组合 | 无 ≥ 2 个有正向证据的模块 | 不建 |

## 4. 完整性与协议
- 复现：S0 = Experiment A / 缓存 1,653 / 1,676 逐位（其余 ≤ 1.9e-14）；B2 闭环回放逐位；R1 / R3 的 oracle 复现 ≤ 1e-12；单测 7 passed。
- 冻结：协议 v1.0 与 E1 代码在正式打分前冻结（`configs/FROZEN_SHA256_E1.json`），G3 臂由 Gate 自动决定后冻结（`configs/G3_ARMS.json`），R2 的 λ / θ 在闭环前预登记；没有后验调参。
- 偏差：T2-1（R1 不含 VALSEL：oracle 点集不同）、T2-2（冒烟 claim 漏算 7 状态，已补）、T2-3（函数签名改动，数值不变）、T2-4（MP3D 无两模型都通过的变体 → 探索性闭环）。
- GT 边界：GT 只用于 oracle 增益、t\* 诊断、R1 / R3 离线反事实、失败案例与 G3 覆盖率；可部署评分器（`t2_scorers`）只读预测深度、P1 几何与历史 10% 采样点；frontier 图只读记录；CC-DistP1 在 B2 / T1 推理时不访问 GT 网格（规划器 LeakGuard 保留，拒绝 0）。
- 资源：≈ 15 GPU-h（固定状态 4.9、G3 9.6、R3 0.4、G0 0.1），既有分配，无新作业，未取消任何作业；输出 12 GB 在 `work_dirs/exp6_v3/task2/`。

## 5. 对 V4 讨论的输入（不作决定）
1. **评分器**：室内把 T1 的 DistP1 换成 S2S3 是目前唯一在闭环上有正向证据的零训练改动（AUC +0.023、Final +0.022、停滞减半），但需要在未用过的楼层上盲测一次；室外保持原 DistP1。
2. **规划区域 / 视野**：R1 + R3 指向"扩大模型输入区域 + 两步前瞻"，而不是 frontier 启发式；这两项都不需要新训练（V2 / V3 整图卷积，n_win 7–9 显存可行）。
3. **模型侧**：本 Task 没有新增关于 3D U-Net / Cross-Attention / Scratch vs Warm-start 的证据；A / B / C 的结论不变（预测侧闭环上限 ≈ 0 Final）。
