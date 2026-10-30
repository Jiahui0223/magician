# EXP5 Task 1 正式实验清单 v1.0

> 2026-10-22，与 `protocol.md` v1.0 同时冻结。
> - 每个实验有独立的目录、配置、日志、结果和报告。
> - **状态**只追加更新：PLANNED → RUNNING → DONE / FAIL / STOP，并注明日期。
> - 运行输出的根目录是 `work_dirs/exp5_future_view/task1/`，下文用 `W/` 表示。

## Task 1.1 数据
| ID | 内容 | 配置 | 输出 | 状态 |
|---|---|---|---|---|
| D-SPLIT | S-c 起点留出，以及 LOSO pisa 的划分 | `configs/data_v1.json` | `frozen_values.json` | PLANNED |
| D-TESTS | protocol §2.6 的 T-R1、T-P1、T-P2、T-L1、T-L2、T-G1 | 同上 | `W/data/checks/` | PLANNED |
| D-BUILD | P1 重放（T-L5）+ 投影 + 粗网格：18,000 个状态，全部合法候选 | 同上 | `W/data/proj/<scene>/<traj>.npz` | PLANNED |
| D-LABEL | embree 标签，加 T-L3 / T-L4 网格链检查 | 同上 | `W/data/labels/<scene>/<traj>.npz`、`W/data/checks/` | PLANNED |
| D-REPORT | 数据清单、可视化、`reports/T1.1_report.md` | — | `W/data/data_manifest.json` | PLANNED |

## Task 1.2 训练
| ID | 训练集 | 配置 | 输出 | 状态 |
|---|---|---|---|---|
| M-SMOKE | 冒烟阶梯（protocol §3.4） | `configs/train_main.json` | `W/training/smoke/` | PLANNED |
| **M-MAIN** | Train-6 去掉 S-c | `configs/train_main.json` | `W/training/main/` | PLANNED |
| M-LOSO | 5 个场景（不含 pisa）去掉 S-c | `configs/train_loso_pisa.json` | `W/training/loso_pisa/` | PLANNED |
| M-SCALE3 | **条件执行**（protocol §3.6）：3 个场景，不含 pisa | `configs/train_scale3.json`（只在触发时生成） | `W/training/scale3/` | 触发时才运行 |
| M-REPORT | 学习曲线、S-c 判定、LOSO 诊断、`reports/T1.2_report.md` | — | — | PLANNED |

## Task 1.3 评分与诊断
| ID | 内容 | 输出 | 状态 |
|---|---|---|---|
| S-TESTS | T-S1：DistP1 入口逐位等价（S-a 73 个状态 × T1-C 世界）；T-S2：EMPTY 违例 | `W/fixed_state/checks/` | PLANNED |
| S-GAIN-C | S-c 校准状态的真实增益（magician 环境，评估侧） | `W/fixed_state/sc_gain/` | PLANNED |
| S-CAL | δ 规则、λ 选择、U 有效性判定 → `frozen_values.json` | `W/fixed_state/calibration/` | PLANNED |
| S-FS-a | S-a 的 73 个状态：V2 各评分、Frontier、真实深度上限、B0 归档 | `W/fixed_state/sa/` | PLANNED |
| S-FS-b | S-b 的 1,500 个 B0 状态：V2 与 B0 的一步比较 | `W/fixed_state/sb/` | PLANNED |
| S-G1ES | G1 ES 匹配检查，以及 V2 在 B′ 48 个状态上的 ES | `W/fixed_state/g1es/` | PLANNED |
| S-LAT | U_flow / U_feat 诊断 | `W/fixed_state/latent/` | PLANNED |
| S-TIME | 计时：B0 / B2 / V2-K4 / V2-K1，30 个状态，独占 GPU | `W/timing/` | PLANNED |
| S-REPORT | `reports/T1.3_report.md` | — | PLANNED |

## Task 1.4 闭环（Dev-3 × 5 个起点 × 100 步 × 3 个噪声重复）
| ID | 名称 | 刷新 | K | 评分 | 轨迹数 | 输出 | 状态 |
|---|---|---|---|---|---|---|---|
| L-SMOKE | 冒烟 + 续跑测试（E1 配置） | R10 | 4 | DistP1 | 2 × 12 步 + 1 次续跑 | `W/closed_loop/smoke/` | PLANNED |
| **E1** | V2G-R10 | R10 | 4 | DistP1 | 45 | `W/closed_loop/E1_V2G_R10_K4/` | PLANNED |
| E2 | V2G-Nat | Native | 4 | DistP1 | 45 | `W/closed_loop/E2_V2G_Nat_K4/` | PLANNED |
| E3 | V2Uraw-R10 | R10 | 4 | U_raw | 45 | `W/closed_loop/E3_V2Uraw_R10_K4/` | PLANNED |
| E4 | V2DistU-R10 | R10 | 4 | DistU | 45 | `W/closed_loop/E4_V2DistU_R10_K4/` | PLANNED |
| E5 | V2GU-R10 | R10 | 4 | r(DistP1) + λ·r(DistU) | 45 | `W/closed_loop/E5_V2GU_R10_K4/` | PLANNED |
| E6 | V2G-R10-K1 | R10 | 1 | DistP1 | 45 | `W/closed_loop/E6_V2G_R10_K1/` | PLANNED |
| L-POST | 每条轨迹的评估器重放和 oracle 诊断 | — | — | — | 270 | 各组的 `replay/`、`oracle_diag/` | PLANNED |
| L-REPORT | `TASK1_FINAL_REPORT.md` | — | — | — | — | — | PLANNED |

- E3 / E4 / E5 只有在 protocol §4.4 判定 U 为 FAIL 时才不运行，并写明原因。不因诊断结果不理想而取消。
- 所有组用同一个 M-MAIN checkpoint，不为不同评分重新训练。

## 引用的基线（不重跑，只核对 sha）
| ID | 来源 |
|---|---|
| B0 | `work_dirs/exp4_multiworld_belief/ablation_study/abl_1/T1C_FRindep_DistP1_Greedy_R10_K4/dev100/` |
| B2 | `work_dirs/exp4_multiworld_belief/ablation_study/abl_2/T1C_FRindep_DistP1_Greedy_R10_K1/chain_k*/dev100/` |
| B1 | `work_dirs/exp4_multiworld_belief/ablation_study/abl_2/Frozen_FRindep_DistP1_Greedy_R10_K4/dev100/` |
| MAGICIAN | `work_dirs/exp3b_macaronspp_transfer/task7a_baseline_v2/attempt2/` |
