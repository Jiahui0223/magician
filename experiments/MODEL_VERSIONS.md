# 模型版本总览：V2（B2 / E2）· V3 · V3.1 · V4 · V4C · V5 · V5.2 · V5.2-M（= V5.3）

> **命名备注**：`V5.3 = V5.2-M Scratch70k（同一模型，讨论名称）`。目录、checkpoint、脚本仍用原名 `exp6_v5_2_scratch70k` / `V5_2M` / `v52m_*`，不另建 V5.3。

整理于 2026-11-02。本文件只记录各版本的结构、继承关系、checkpoint 位置和历史已报告结果（各实验报告中的数字，不是论文发表结果），不含新实验。数字全部摘自各实验自己的报告，出处见表注。
- 文件级的 checkpoint 明细（新路径、原路径、step、参数、SHA256、严格加载结果）：`experiments/CHECKPOINTS.md`。该文件由 `experiments/checkpoint_registry.py` 生成。
- checkpoint 统一入口：`work_dirs/checkpoints/<实验>/`。这里全部是指向原文件的相对软链接，原文件未移动、未删除，旧路径继续有效。
- 评估场景（Macarons++ Dev-3 / Test-3、MP3D VALSEL）都**不是盲测**。

## 1. 结构与继承
| 版本 | 实验目录 | 结构要点 | 参数（实测） | 初始化 | 训练数据 / 步数 | 选择 | 规划 P1（输入 / 评分） | 主 checkpoint（`work_dirs/checkpoints/…`） |
|---|---|---|---|---|---|---|---|---|
| V2 Original | `exp5_future_view/task1` | 3D U-Net（Encoder3D 32/64/96）+ 32 点射线编码器 + Conv1D + 2D rectified-flow U-Net，K 个未来深度 → DistP1 | 24,822,114 | 随机 | Outdoor Train-6 去 S-c，稀疏 P1，60k | EMA@60k | 稀疏 / 稀疏 | `exp6_v2/B2/lineage/v2orig_ema60000.pt` |
| V2-Mix（= B0 / **B2** 权重） | `exp5_future_view/task2`；整理入口 `exp6_v2/B2` | 同 V2 | 24,822,114 | V2 Original EMA@60k | MP3D train + Train-6（稀疏），20k | EMA@17.5k（ES 比值） | B0：稀疏 / 稀疏；**B2：D100 / S10** | `exp6_v2/B2/v2mix_ema17500.pt` |
| **E2-D100** | `exp5_future_view/task3_full_depth_p1/e2_pilot`；整理入口 `exp6_v2/E2` | 同 V2 | 24,822,114 | V2-Mix EMA@17.5k | MP3D + Train-6，D100 分片，5k | EMA@5k | D100 / S10 | `exp6_v2/E2/e2d100_ema5000.pt`（对照 E2-S10：`exp6_v2/E2/control_e2s10/`） |
| V3-T0（V3-VAE） | `exp6_v3/task0_vae` | VAE 潜空间（编码 / 解码）+ 潜空间射线编码器 + 2D flow | 23,539,622 | 随机 | Mixed scratch（MP3D + Train-6，D100 + TSDF 缓存），40k | EMA@40k | D100 / S10 | `exp6_v3/T0_vae/v3t0_ema40000.pt` |
| V3-T1（V3-VAE-XAttn） | `exp6_v3/task1_xattn` | V3-T0 + 射线交叉注意力 | 23,730,986 | 随机 | 同 T0，40k | EMA@40k | D100 / S10 | `exp6_v3/T1_xattn/v3t1_ema40000.pt` |
| V3.1 | `exp6_v3.1`（work_dirs 中为 `exp6_v3_1`） | V3-T0 结构 | 23,539,622 | 随机 | Outdoor-only scratch（Train-6 去 S-c，D100 + TSDF），40k | EMA@40k | D100 / S10 | `exp6_v3_1/V3_1/v31_ema40000.pt` |
| V4 | `exp6_v4` | V2 模块 + 射线对齐交叉注意力（ray_xattn） | 24,847,544 | 随机 | Outdoor-only scratch（D100 分片），40k | EMA@40k | D100 / S10 | `exp6_v4/V4/v4_ema40000.pt` |
| V4C（V4-Continue12k） | `exp6_v5` | 同 V4 | 24,847,544 | V4 EMA@40k | 继续 12k（V5 的对照） | EMA@12k | D100 / S10 | `exp6_v5/V4C/v4c_ema12000.pt` |
| V5（V5-Warm12k） | `exp6_v5` | V4 的 U-Net + Flow，加几何感知 ray-to-scene memory + 多表面 belief 解码器（去掉 32 点采样） | 24,988,647 | V4 EMA@40k（部分模块） | Outdoor，12k | EMA@12k | D100 / S10 | `exp6_v5/V5/v5_ema12000.pt` |
| V5.2 pilot | `exp6_v5_2` | Idea 2：射线几何 belief + 条件首中 hazard，直接计算可见 / 潜力收益；无 Flow、无 DistP1 | 2,668,710 | V5 EMA@12k（warm-start） | Outdoor，6k | EMA@6k | D100 / S10（打分器自带） | `exp6_v5_2/V5_2/v52_ema6000.pt` |
| V5.2-M（**= V5.3**，讨论名称，同一模型） | `exp6_v5_2_scratch70k` | Idea 2 大容量：3D U-Net → Ray Query → 2D Spatial U-Net → 1D 深度 decoder，FREE 在 hazard 之前生效 | 26,999,143 | 随机 | Outdoor Train-6，70k | best_train_holdout = EMA@50k | 同 V5.2 | `exp6_v5_2_scratch70k/V5_2M/v52m_best_ema50000.pt`（另有 `v52m_ema70000.pt`） |

## 2. G3 闭环（历史已报告结果：AUC / Final）

评估范围：MP3D VALSEL 10 episodes × 150 steps；Macarons++ Dev-3 + Test-3（6 scenes，30 episodes × 100 steps）——即 Dev-3（barts、pantheon、colosseum）与 Test-3（alhambra、bridge、neuschwanstein）各 3 个场景 × 5 个起点。

| 版本 | MP3D（10 ep × 150） | Macarons++ Dev-3 + Test-3（30 ep × 100） | 出处 |
|---|---|---|---|
| MAGICIAN Beam（MAG-B） | 0.292 / 0.377 | 0.685 / 0.891 | `exp6_v5/reports/FINAL_REPORT.md` §2 |
| V2 Original | 0.315 / 0.441 | 0.724 / 0.898 | 同上 |
| **B2** | 0.334 / 0.530 | **0.730 / 0.906** | 同上；`exp5…/task3_full_depth_p1/reports/T3_E1_B2_CLOSED_LOOP.md` |
| **E2-D100** | **0.364 / 0.553** | 0.711 / 0.880 | 同上；`e2_pilot/reports/E2_G3_REPORT.md` |
| V3-T0 / V3-T1 | 0.311 / 0.463 · 0.331 / 0.508 | 0.702 / 0.872 · 0.712 / 0.883 | `exp6_v3/reports/g3.md` |
| V3.1 | 0.335 / 0.499 | 0.697 / 0.865 | `exp6_v5/reports/FINAL_REPORT.md` §2 |
| V4@40k | 0.327 / 0.495 | 0.710 / 0.878 | 同上 |
| V4C | 0.342 / 0.484 | 0.713 / 0.882 | 同上 |
| V5 | 0.333 / 0.501 | 0.711 / 0.891 | 同上 |
| V5.2 pilot λ0 | 0.238 / 0.310 | 0.608 / 0.781 | `exp6_v5_2/reports/FINAL_REPORT.md` |
| V5.2-M（= 讨论名 V5.3） | 只有 smoke（冻结门槛未通过，未跑 Full） | 无 30 ep 结果；smoke 只含 Test-3 的 4 个 episode（alhambra s0/s1、bridge s0/s1）：AUC λ0 / λ0.1 = 0.727 / 0.735，与本列不可直接比较 | `exp6_v5_2_scratch70k/FINAL_REPORT.md`、`G3_SMOKE.md` |

## 3. 审查发现（不影响已有结果，供决定是否处理）
1. **命名冲突**：`exp5_future_view/task1/configs/closed_loop_v1.json` 中的 "B2" 基线行和 "E2" 臂（E2_V2G_Nat_K4）是 EXP5 Task 1 / EXP4 的旧名，与 Task 3 的 B2（D100 输入 / S10 评分）、E2-D100 无关。各 README 已注明。
2. **目录名不统一**：V3.1 在 `experiments/` 中叫 `exp6_v3.1`，在 `work_dirs/` 中叫 `exp6_v3_1`。checkpoint 注册表沿用 `exp6_v3_1`。没有改名，因为旧脚本依赖这个路径。
3. **`final.pt` 与最后一个 EMA 字节相同**：V3 / V3.1 / V4 / V5 / V4C / V5.2 的 `checkpoints/*final*.pt` 与 `runs/*/ema_<最后一步>.pt` 相同。注册表只登记一次，并在 MANIFEST 中标为 alias。
4. **V5.2-M 的 best 与 step_050000**：内容来源相同，但文件字节不同，因为是分别保存的，两者都登记了。
5. **未登记的文件**：G0 测试运行（`runs/resume_A`、`runs/resume_B`、`runs/g0`、smoke）不是正式权重，没有登记。V2 Original 的中间 EMA 仍在 `work_dirs/weights/W20_exp5_v2/W20-4_*`，EXP4 世界模型在 W10 / W11，均未纳入 exp6_* 注册表。
6. **启动脚本不在版本库**：B2 / E2 的历史 G3 / G1G2 启动脚本只在 `work_dirs/` 中。`exp6_v2` 已把它们收为版本化、参数化的入口（`shared/g3_queue.sh`、`shared/g12_fixed.sh`），默认输出到新目录。V3–V5.2-M 的启动脚本本来就在各自 `experiments/` 目录中。
7. **V5.3 命名**：`V5.3 = V5.2-M Scratch70k（同一模型，讨论名称）`。实验目录（`experiments/exp6_v5_2_scratch70k`）、checkpoint（`work_dirs/checkpoints/exp6_v5_2_scratch70k/V5_2M/`）和脚本名称保持不变，没有建立 V5.3 目录，也没有重复登记权重。
