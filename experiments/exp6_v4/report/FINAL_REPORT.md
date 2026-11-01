# EXP6 V4 / V3.1 第一阶段最终报告 —— Outdoor-only 从零训练 + 双域 G1 / G2 / G3（2026-10-31 19:55 → 11-01 10:51）

> 任务书 `../../exp6_v3/EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md`；审计 `../AUDIT_BASELINES.md`、`../CHECKPOINT_LINEAGE.md`；协议 `../FROZEN_PROTOCOL.md` v1.0（20:46 冻结，无偏差）；G0 `G0_VALIDATION.md`；训练 `OUTDOOR_TRAINING.md`、`../../exp6_v3.1/report/OUTDOOR_TRAINING.md`；评估 `G3_OUTDOOR.md`、`G3_INDOOR_ZEROSHOT.md`、`G1_G2_EVAL.md`；统一表 `../reports/V4_COMPARISON_TABLES.md`（Table A–D）；资源 `../reports/V4_RESOURCES.md`；偏差 `../reports/V4_DEVIATIONS.md`。全部数字来自真实运行；评估场景非盲（Test-3 与 5 个 MP3D 楼层被历次实验使用）。

## 1. 一句话结论
| 问题 | 结论 |
|---|---|
| 只用 Macarons++ Train-6、零 MP3D 样本、40k 步从零训练，V4 与 V3.1 能到什么水平？ | **本域（Macarons++）**：V4 0.710 / 0.878 / 0.556、V3.1 0.697 / 0.865 / 0.551（AUC@100 / Final / Cov@20），都略低于 V2 Original（0.724 / 0.898）与 B2（0.730 / 0.906），差距 −0.014 ～ −0.034 AUC，CI 多含 0，按 Task 3 §4 判"退化"；与同族 Mixed 模型 V3-T1 / T0 无差异。**MP3D zero-shot**：V3.1 0.335 / 0.499、V4 0.327 / 0.495，达到 Mixed 训练的 V3-T1（0.331 / 0.508）与 B2（0.334 / 0.530）的 AUC 水平、Final 低 0.03（CI 含 0），高于 MAGICIAN（Final +0.12），低于 E2-D100（0.364 / 0.553）。 |
| V4（Ray-Aligned X-Attn）vs V3.1（VAE）？ | 闭环**无显著差异**（Macarons++ +0.013 / +0.013，MP3D −0.008 / −0.003，胜负 3/3、2/3）。V4 训练 **17.3 GPU-h**（0.47 s/步）vs V3.1 **57.6 GPU-h**（1.29 s/步）；推理时延同级。固定状态上 V4 的一步预测质量明显更差（室内未知射线 MAE 0.381 vs 0.237，被选增益 0.570 vs 0.883 pp），但没有兑现为闭环差距。 |
| X-Attn 是否造成改善？ | **不能下结论**：严格对照 U0（V2 架构 + D100 + 40k + Outdoor-only，无 X-Attn）未训练；V2 Original 的口径（稀疏 P1、60k）不同。V4 − V2 Original：Macarons++ −0.014 AUC，MP3D Final +0.054 [+0.015, +0.117]（5/0）—— 混杂了 P1 密度与预算。 |
| Outdoor-only vs Mixed（同架构 V3-T0 / V3.1）？ | MP3D zero-shot 上 **V3.1 优于 Mixed 的 T0**（ΔAUC +0.024 [+0.002, +0.052]，4/1，"meaningful on one"），Macarons++ 无差异（−0.006）：V3 原先的 Mixed 训练没有带来室内收益。 |

## 2. 做了什么（事实）
| 项 | V4 | V3.1 |
|---|---|---|
| 结构 | V2 3D U-Net（F0 32 @8×、F1 64 @16×）+ 原 32 点射线采样 + Conv1d 射线编码器 + **Ray-Aligned X-Attn**（每条射线对自身 32 个采样点 token 的 4 头注意力，query = 射线上下文 + Fourier(方向, 相机位置)，mask = 网格内，W_o std 0.005 残差）+ 原 2D Rectified Flow；参数 24,847,544（X-Attn 25,430） | V3-T0：Simple 3D VAE（[32, 64] → μ / logvar 16）+ 原 LatentRayEncoder + 原 2D Flow；参数 23,539,622；初始公共模块 sha 与 T0 相同 |
| 数据 | Macarons++ Train-6 去 S-c：138 条轨迹 / 98,887 视角 / 13,800 状态（E2 D100 分片）；MP3D 样本 0；S-c 只监控 | 同左 + 4× TSDF 缓存 |
| 训练 | 40k 步，64 视角 / 8 状态 / 步，V2 Original 配方（lr 2e-4）；gpu115，4.34 h ≈ 17.3 GPU-h，0.472 s/步，18.4 GB/卡；每步有效监督射线均值约 124k | 40k 步，同全局 batch（accum 2，整步全局射线数归一化），V3 配方（lr 1e-4，λ_rec 1，β 0.02）；gpu117，14.39 h ≈ 57.6 GPU-h，1.287 s/步，23.4 GB/卡；active units 16/16，KL/unit 0.38（无后验坍缩） |
| checkpoint | 固定 EMA@40k（训练前冻结）`work_dirs/exp6_v4/checkpoints/final.pt` sha `1b31afff…` | `work_dirs/exp6_v3_1/checkpoints/final.pt` sha `13df0a45…` |
| 开发指标（S-c，只监控） | FM 0.0196，fair ES 3.74 vox，Brier 0.016 | FM 0.0275，ES 6.19，Brier 0.027，重建精度 0.994 / 0.996 / 0.998 |
| G0 | 12 门全部通过（`G0_VALIDATION.md`）；开关关闭时与 V2 逐位相同；冻结规划器回放 4/4 逐位 | 同 |
| 评估 | 冻结 G3（B2 口径，n_win 5，K 4）：Macarons++ 30 × 100、MP3D zero-shot 10 × 150；G1 / G2 固定状态 365 态（B2 回放逐位） | 同 |
| 总资源 | 训练 17.3 + G3 3.3 + G1/G2 0.9 | 训练 57.6 + G3 ≈ 3.4（两节点）+ G1/G2 0.9 |；G0 ≈ 1.5；**合计 ≈ 85 GPU-h（L40S）** |

## 3. G3 统一表（AUC / Final / Cov@20；[CI] 见 `G3_*.md`）
| 组 | 训练来源 | MP3D zero-shot？ | MP3D AUC@150 | Final | Cov@20 | 停滞 | Macarons++ AUC@100 | Final | Cov@20 |
|---|---|---|---|---|---|---|---|---|---|
| MAG-G | 官方 | 官方 | 0.278 | 0.377 | 0.169 | 7 | 0.704 | 0.905 | 0.507 |
| MAG-B | 官方 | 官方 | 0.292 | 0.377 | 0.165 | 8 | 0.685 | 0.891 | 0.504 |
| V2 Original | Outdoor-only 60k（稀疏 P1） | 是 | 0.315 | 0.441 | 0.179 | 2 | 0.724 | 0.898 | 0.578 |
| B0 | Mixed FT（稀疏） | 否 | 0.322 | 0.469 | 0.195 | 3 | 0.729 | 0.884 | 0.622 |
| B2 | Mixed FT（D100） | 否 | 0.334 | 0.530 | 0.197 | 4 | 0.730 | 0.906 | 0.572 |
| E2-D100 | Mixed FT + 5k | 否 | **0.364** | **0.553** | 0.195 | 2 | 0.711 | 0.880 | 0.570 |
| V3-T0 | Mixed scratch 40k | 否 | 0.311 | 0.463 | 0.183 | 7 | 0.702 | 0.872 | 0.568 |
| V3-T1 | Mixed scratch 40k + X-Attn | 否 | 0.331 | 0.508 | 0.184 | 5 | 0.712 | 0.883 | 0.566 |
| **V3.1** | **Outdoor-only scratch 40k** | **是** | **0.335** | 0.499 | 0.185 | 3 | 0.697 | 0.865 | 0.551 |
| **V4** | **Outdoor-only scratch 40k + Ray X-Attn** | **是** | 0.327 | 0.495 | 0.187 | 4 | 0.710 | 0.878 | 0.556 |

关键配对差（Task 3 §4）：V4 − V3.1 两域无差异；V3.1 − T0 MP3D +0.024 AUC [+0.002, +0.052]（有意义）；V4 / V3.1 − B2 两域退化（Final −0.03 ～ −0.04，CI 多含 0，室外 V3.1 0/6）；V4 / V3.1 − E2-D100 MP3D 退化（Final −0.05 ～ −0.06，CI 不含 0）；V4 / V3.1 − MAG-B MP3D Final +0.12（4/1、5/0），Macarons++ Final −0.01 ～ −0.03。

## 4. 解读（证据边界内）
1. **室外本域没有收益**：同一数据域上，40k 步的新架构没有超过 60k 步的 V2 Original 或 Mixed 微调的 B2；V3.1 还略差（0/6 场景的 Final）。这与 V3 的结论一致（Mixed scratch 的 T0 / T1 也低于 B2）。
2. **室内 zero-shot 很强**：没有见过任何 MP3D 数据的 V3.1 / V4 在 MP3D 闭环上与见过 228 条 MP3D 轨迹的 V3-T1 / B2 同水平。结合 V3.1 > T0，说明**在 V3 架构族里 Mixed 训练中的 MP3D 数据没有转化为室内闭环收益**，而 E2-D100 的室内优势来自 V2 Original → V2-Mix → D100 微调这一链。因此第二阶段"从 Outdoor-only 权重出发的 MP3D 微调"有明确的目标（E2-D100 0.364 / 0.553）和合理的起点。
3. **V4 的效率**：与 V3.1 闭环相当，训练成本 1/3.3、每步快 2.7 倍；但一步预测质量更差且样本更不分散（可能的因果：X-Attn 残差让条件更"确定"、Flow 更快收敛到低方差解），这是继续研究 V4 时首先要诊断的点（固定状态上的 ES / 离散度 vs 闭环）。
4. **X-Attn 因果**：没有 U0 不能宣称。V4 vs V2 Original 的差混杂了 P1 密度（D100 vs 稀疏）和预算（40k vs 60k）。

## 5. Go / No-Go 建议（供决定；本轮不自动启动任何一项）
| 项 | 建议 | 依据 / 成本 |
|---|---|---|
| 第二阶段 MP3D 室内微调（V4-FT、V3.1-FT，`configs/indoor_finetune_DRAFT.json`：各自 EMA@40k 起，MP3D train 228 条，5k 步，lr 2e-5，微调前后双域 G3） | **Conditional Go**：目标是从 0.33 / 0.50 接近 E2-D100 的 0.364 / 0.553，并量化室外遗忘（Table E / F） | 每组约 5k × ~1.3–3 GPU-h + 双域 G3 ≈ 10 GPU-h；预计半天 |
| U0（V2 架构、`ray_xattn.enabled = false`、同数据 / 预算 / 种子） | **Go（若要对 X-Attn 下任何结论）**：唯一能把 V4 − U0 归因于 X-Attn 的对照 | ≈ 16 GPU-h 训练 + 4 GPU-h 评估 |
| 延长到 60k 步 | **No-Go（现阶段）**：V4 的 S-c ES 35k → 40k 只降 1 %（3.78 → 3.74），V3.1 降 0.2 %；室外差距更可能来自数据 / 口径而非步数 | — |
| n_win 7 / 9 规划器消融（`configs/nwin_ablation_DRAFT.json`） | 独立决定；冻结模型 + 预注册固定状态测试先行 | 固定状态 ≈ 2 GPU-h，闭环每模型每 n_win ≈ 1.5 GPU-h |
| V4 固定状态质量诊断（K = 4 离散度、X-Attn 注意力熵 vs 闭环） | 建议在 U0 之后或与之并行 | CPU + 少量 GPU |

## 6. 没有做的 / 边界
- 未训练 U0；未启动 MP3D 微调；未跑 n_win ≠ 5 的闭环；未改 DistP1 / 候选 / 2D Flow / Loss。
- 所有闭环数字都是非盲对照；MAG-G / MAG-B 来自 Task 2 的同动作协议闭环（不重跑）。
- V2 Original 为稀疏口径，与其余 D100 口径不同，表中单独标注。

## 7. 代码 / 提交
`seen2scene/futureview/model.py`（RayXAttn + 开关）、`experiments/exp6_v4/`、`experiments/exp6_v3.1/`；提交 `f57be9a` `[V4][setup]`、`4cef0f8`（冻结）、`6a6843f` / `64c59c5` `[V4][train]`、`5784e2e` `[V3.1][train]`、`ef3dd06` / `e9aa310` `[EXP6][eval]` 及本报告的提交；权重与闭环输出在 `work_dirs/exp6_v4/`、`work_dirs/exp6_v3_1/`（不入库）。
