# EXP5 — 低覆盖诊断 + 相机条件的随机未来视角预测（V2）

> 2026-10-22 建立。
> - **任务书**：仓库根目录 `EXP4_NextStage_Task0_Task1_V2_Research_Plan.md`，sha256 `981c63ad9823cb14ac9ee1b8a3060547a3d41a0c6d966f48e349a01bc3e64ab9`。该文件不复制、不修改；如果任务书更新，这里记录新的 sha。
> - 计划书里的推荐目录是 `exp4_multiworld_belief/next_stage/`。按你的要求改为单独的 EXP5 目录：Task 0 → `task0/`，Task 1 → `task1/`。
> - 目录名 `exp5_future_view` 是我起的，可以改。
> - 原来空的 `exp5_active_geometry/` 已按你的要求删除。

## 目标
- **Task 0**：用 MP3D 室内场景和 Macarons++ 室外场景，在匹配的极低覆盖条件下诊断冻结 Seen2Scene 和 T1-C。
- **Task 1（V2）**：对每个合法候选相机，直接预测 K 个可能的未来深度和命中情况，接上冻结的 DistP1 + Greedy，在 Dev-3 上跑 100 步闭环。

## 进度（每次状态变化立即更新）

| 任务 | 状态 | 文件 | 最近更新 |
|---|---|---|---|
| Task 0 室内 / 室外低覆盖验证 | **DONE**（2026-10-24）：T0.1–T0.7 全部完成。完整 MP3D 已就绪（1.17 TB，已校验）；室内 157 个固定状态（M / R 协议）+ 室外 85 个 + 室内 22 × 4 条 100 步闭环。室内闭环中 V2K4 的 AUC@100 为 0.467，比随机参考（0.350）、Frozen（0.331）和 T1-C（0.311）都高；两个世界模型因已观测 EMPTY 里的幻觉增益而停滞（事后诊断）；T1-C 没有损害室内；原版 Seen2Scene 在室内极低覆盖时并不更好。8 个问题的回答与后续方向（A 规划侧修正、B V2 室内微调；未执行，等你决定）见 `task0/TASK0_FINAL_REPORT.md`。**扩展 T0.8–T0.11（10-25 DONE）**：同协议下 V2K4 + Greedy 与 MAGICIAN Beam 相当（AUC 0.467 / 0.475），beam 在站点图上只带来 +0.021；V2K4 加上"只走未访问位姿"规则后最好（0.509）；论文的 MP3D 协议无法用发布代码复现。见 `task0/reports/T0.11_final_comparison.md`。**T0.12 网格对齐 + T0.12-C 离线诊断（10-25 DONE）**：在 MAGICIAN 式的 1.0 m 网格上，MAG-B 的 AUC@100 为 0.254，V2K4 为 0.214，CI 含 0，差距主要来自一个房屋；原因是 V2 的规划区域只有约 7.8 m，看不到远处，放大到整层的探针证实了这一点；C1 / C2 / A / B 等你决定。总结见 `task0/reports/T0.12_summary.md`。历史记录：2026-10-22 WAIVED / DEFERRED，2026-10-23 RESUMED | `task0/TASK0_FINAL_REPORT.md`、`task0/reports/` | 2026-10-25 |
| Task 3 / E1 Full-Depth P1（冻结 V2-Mix，只把 P1 的已知表面改为由全部有效深度像素融合） | **DONE**（2026-10-28）：已知表面有效率提升 15–25 个百分点，伪表面率约翻倍（细结构）。闭环 MP3D ΔAUC +0.009 [−0.026, +0.059]（持平），Macarons++ ΔAUC −0.038 [−0.066, −0.012]（退化）→ **No-Go for direct replacement**。固定状态归因：稠密评分在 MP3D 有益、在 Macarons++ 有害；稠密输入对冻结模型帮助有限。**B2 闭环**（稠密输入 / 稀疏评分）：MP3D ΔAUC +0.012、ΔFinal +0.061（CI 不含 0），Macarons++ ΔAUC +0.001、ΔFinal +0.022，按冻结标准仍为 No-Go，但无退化。**E1.2 密度消融**（固定状态）：几何在 50% 饱和，50% / 75% 未优于 100%，未跑闭环；评分端室内稠密有益、室外有害。E2 推荐 D100 + 稀疏评分，方案待批准。见 `task3_full_depth_p1/reports/`、`task3_full_depth_p1/density_ablation/reports/` | `task3_full_depth_p1/` | 2026-10-28 |
| Ablation A1 / A2（预测–评分对齐；随机未来视角） | **DONE**（2026-10-28）：在 368 + 1,093 个固定状态上做同几何、同噪声的配对比较。V2-Mix 的一步排序显著好于 V2 Original：MP3D regret 0.210 → 0.119 pp，Top-1 0.41 → 0.56。GT 深度 + DistP1 的 Top-1 也只有约 0.69：室内主要因为规划区域裁剪，室外部分因为评估不计分的表面。K = 4 相对 K = 1 只在 V2-Mix 室内有小幅收益；样本分歧有信息量但未被利用；失败多为 4 个样本一致的系统偏差。见 `ablation_study/reports/ABLATION_SUMMARY.md` | `ablation_study/` | 2026-10-28 |
| Task 2 V2-Mix + 最终对照 | **DONE**（2026-10-27）：用 MP3D 和 Macarons++ 联合微调得到 V2-Mix（室内选择集 ES 15.5 → 6.0）。在共享动作网格上与 MAGICIAN 对比：MP3D 室内 AUC@150 0.322（MAG-B 0.292），Cov@150 0.469 对 0.377（显著），Comp@5cm 46.7% 对 37.6%；Macarons++ Test-3 AUC@100 0.723 对 0.656 / 0.685，最终覆盖率相近。相对 V2 Original 的闭环提升小且不显著；瓶颈在规划。见 `task2/reports/TASK2_FINAL_REPORT.md` | `task2/` | 2026-10-27 |
| Task 1 V2 | **DONE**（2026-10-23）：T1.1–T1.4 全部完成。V2 + DistP1 的 AUC@100 为 0.713（B0 0.683、MAGICIAN 0.697），Cov@100 为 0.856，episode 墙钟时间约为 B0 的 1/40；DistU ≈ DistP1，没有互补；跨场景泛化是主要瓶颈。见 `task1/TASK1_FINAL_REPORT.md` | `task1/TASK1_FINAL_REPORT.md`、`task1/reports/` | 2026-10-23 |

**GPU-h**：Task 1 ≈95（明细见 `task1/README.md`）；Task 0 ≈235 + T0.8–T0.11 ≈15.5 + T0.12 ≈7.3 + T0.12-C ≈3.3（明细见 `task0/TASK0_FINAL_REPORT.md` §9、`task0/reports/T0.11_final_comparison.md` §5、`task0/reports/T0.12_summary.md` §6）。

## 约定
- **数据**：Test-3 不碰；Train-6 只用于训练、选择和校准；Dev 只用于评估。
- **基线**：EXP4 的 B0（abl_1 A2）、B1、B2 和 MAGICIAN T7a 只引用已冻结的结果并记录哈希，不重跑。
- **冻结代码**：不修改任何 EXP3b / EXP4 的冻结代码和结果。
- **偏差**：只追加到 `DEVIATIONS.md`，不改已冻结的文字。
- **git**：可复用的代码放在仓库包里，提交前缀 `[exp5][类型] 描述`；实验目录和 `work_dirs/` 不进 git。
- **运行输出**：`work_dirs/exp5_future_view/`，到实际运行时才建立。

## 决策与修改记录
见 `decision_log.md`。
