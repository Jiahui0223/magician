# V3_HISTORY — EXP6 V3（T0 / T1）与 V3.1 的历史结论复核（F1 §4，只读）

> 2026-11-03 只读整理：没有训练、没有重算。所有数字都摘自下列原始报告，路径相对于 `experiments/`。原报告没有的数字写"未提供"，不推断，也不补算。
> 一句话结论：T0 / T1 在同一条件下训练（单种子）。T1 − T0 的闭环差值小，方向基本一致，但 CI 多数包含 0，生成质量（G1）没有变化，G2 在两个数据集上方向相反。**这些证据不足以证明 Scene Cross-Attention 有独立价值。**

## 1. 定义与训练条件
| 项 | V3-T0（V3-VAE） | V3-T1（V3-VAE-XAttn） | 来源 |
|---|---|---|---|
| 结构 | 从零训练的 Simple 3D Conv VAE（4× partial TSDF → μ/logvar 16 ch @8×）+ V2 结构 Ray Encoder（沿射线采样 μ）+ 2D Rectified-Flow U-Net | 与 T0 相同，另在 U-Net 瓶颈处加 **1 个** Camera-Ray → 场景潜变量 Cross-Attention（K/V ≤ 512 个视锥内 μ token，`W_o` 零初始化，+191,364 参数） | `exp6_v3/README.md:13-23`、`exp6_v3/protocol.md:18-21` |
| 配置 | `model_v3.json` 中 `xattn.enabled = false` | 同一文件，`xattn.enabled = true`（唯一差别） | `exp6_v3/configs/model_v3.json:3,51-52` |
| 训练数据 | MP3D train 228 条轨迹（114 个楼层）+ Macarons++ Train-6 去掉 S-c 后的 138 条；每步 32 + 32 视角 | 相同，`data_manifest_sha256 9f432c3f…` 与 T0 一致 | `exp6_v3/protocol.md:8-14`；`T0_TRAINING.md:29-32`；`T1_TRAINING.md:11,33` |
| 步数 / 种子 / 初始化 | 40,000 步，seed 0，公共初始化 sha `6a9d6071…` | 40,000 步，seed 0，公共初始化 sha 相同；第 1 步损失 3.674514578 与 T0 逐位相同 | `exp6_v3/configs/train_v3.json:5-6`；`T1_TRAINING.md:12` |
| checkpoint 选择（冻结） | 在 VALSEL / S-c 上取 score = ES_in/ES_in(5k) + ES_out/ES_out(5k) 的最小值 → 分数单调下降，选中 **EMA@40k**（1.380） | 同一规则，同样单调下降，选中 **EMA@40k**（1.395） | `train_v3.json:47-51`；`protocol.md:33`；`T0_TRAINING.md:74`；`T1_TRAINING.md:77` |
| 运行环境 | gpu117，4 × L40S，63.7 GPU-h，代码提交 `7de02cc` | gpu115，4 × L40S，65 GPU-h，代码提交 `2021604`（代码逐文件与 `7de02cc` 相同） | `T0_TRAINING.md:3`；`T1_TRAINING.md:3` |
| 训练终点 dev（VALSEL / S-c） | FM 0.0900 / 0.0272；ES 6.37 / 6.44；hit ratio 1.006 / 0.969 | FM 0.0900 / 0.0272；ES 6.43 / 6.48；hit ratio 1.010 / 0.971；每个评估点与 T0 相差都在 ±1% 以内 | `T0_TRAINING.md:77`；`T1_TRAINING.md:80,83` |

**是否同条件：是**（同数据、同样本流、同噪声、同步数、同选择规则，T1 不读取 T0 的权重；`exp6_v3/protocol.md:46`）。剩下的差别有：**单种子**、两个不同节点、两个 commit SHA（代码内容相同）。两组的选择分数到 40k 步时都还在下降（`V3_COMBINED_REPORT.md:78`）。T1 的共享参数与 T0 的相对差为 0.29，`W_o` 的 rms 为 0.006，即 XAttn 支路不是零（`T1_TRAINING.md:87-90`）。

## 2. 原报告数值（T0 / T1）
覆盖率口径：**MP3D = C_obs 规则**（可见 GT 点落入历史 10% 采样点的 eps 邻域）。**Macarons++ 主指标 = 单调覆盖**（`t25_out_cov`），`g3.json` 中另存了 **MAGICIAN 官方覆盖**作为次口径（`exp6_v3/shared/analyze_g3.py:84-87,145`；`exp6_v3/shared/out_cov_v3.py:1`）。
G1 / G2 在固定状态上评估：MP3D VALSEL 有 90 个状态（9 个楼层），Macarons++ Dev-3 有 275 个状态（3 个场景）。G3 闭环：MP3D 10 个 episode（5 个楼层 × 2 个起点 × 150 步），Macarons++ 30 个 episode（Test-3 + Dev-3 共 6 个场景 × 5 个起点 × 100 步）。两者都不是盲集。CI 为场景级 bootstrap 95%。

**G1 未来深度**（`exp6_v3/reports/g12_s_final.md:118-137,192-212`；MAE 未做单位换算）
| 集合 | 指标 | T0 | T1 | T1 − T0 [CI] |
|---|---|---:|---:|---|
| MP3D VALSEL | MAE all / unknown / new / known | 0.138 / 0.191 / 0.862 / 0.089 | 0.139 / 0.193 / 0.860 / 0.089 | unknown +0.001 [−0.001, 0.003]；new −0.001 [−0.006, 0.003]；known 0.000 [−0.000, 0.001] |
| MP3D VALSEL | hit_agree / hit_dis（K=4） | 0.755 / 0.041 | 0.756 / 0.041 | 未提供 |
| Macarons++ Dev-3 | MAE all / unknown / new / known | 1.856 / 3.828 / 5.545 / 1.585 | 1.861 / 3.835 / 5.561 / 1.591 | unknown +0.007 [−0.014, 0.018]；new +0.016 [−0.016, 0.048]；known +0.005 [−0.003, 0.015] |
| Macarons++ Dev-3 | hit_agree / hit_dis（K=4） | 0.809 / 0.023 | 0.808 / 0.022 | 未提供 |
| 两者 | **No-Hit 专项指标**；G1 中的 Brier / ES | 未提供 | 未提供 | 未提供 |

**G2 候选排序**（`g12_s_final.md:82-83,118,131-142,156-157,192,205-216`；`exp6_v3/reports/V3_COMPARISON_TABLES.md:47-56`）
| 集合 | 指标 | T0 | T1 | T1 − T0 [CI] |
|---|---|---:|---:|---|
| MP3D VALSEL 全部 | Top-1 / regret | 0.567 / 0.170 | 0.578 / 0.155 | Top-1 +0.011 [−0.022, 0.044]；regret −0.014 [−0.038, 0.000] |
| MP3D VALSEL Early（36） | Top-1 / regret | 0.417 / 0.385 | 0.472 / 0.348 | Top-1 +0.056 [0.000, 0.139]；regret −0.036 [−0.094, −0.002] |
| Macarons++ Dev-3 全部 | Top-1 / regret | 0.548 / 0.343 | 0.552 / 0.361 | Top-1 +0.004 [0.000, 0.010]；regret +0.019 [−0.009, 0.071] |
| Macarons++ Dev-3 Early（95） | Top-1 / regret | 0.425 / 0.740 | 0.433 / 0.785 | Top-1 +0.008 [−0.025, 0.025]；regret +0.045 [−0.031, 0.161] |
| 两者 | **Selected Gain** | 未提供 | 未提供 | 未提供 |

G2 → G3 放行：T0 按规则 B 通过，T1 按规则 A 通过（`V3_COMPARISON_TABLES.md:64-65`）。

**G3 闭环**（`exp6_v3/reports/g3.md:5-19`；`V3_COMPARISON_TABLES.md:8-20,27-28`；官方覆盖取自 `exp6_v3/reports/g3.json:452-481,5865-5973`）
| 数据集 / 口径 | 指标 | T0 | T1 | T1 − T0 [CI]（胜/负） | 参照 B2 / MAG-B |
|---|---|---:|---:|---|---|
| MP3D（C_obs） | AUC@150 | 0.311 | 0.331 | +0.020 [−0.003, 0.042]（4/1） | 0.334 / 0.292 |
| MP3D（C_obs） | Final | 0.463 | 0.508 | +0.044 [−0.036, 0.121]（3/2） | 0.530 / 0.377 |
| MP3D（C_obs） | Cov@20 | 0.183 | 0.184 | +0.001 [−0.007, 0.014] | 0.197 / 0.165 |
| Macarons++（单调） | AUC@100 | 0.702 | 0.712 | +0.010 [−0.006, 0.034]（3/3） | 0.730 / 0.685 |
| Macarons++（单调） | Final | 0.872 | 0.883 | +0.011 [0.005, 0.017]（6/0） | 0.906 / 0.891 |
| Macarons++（单调） | Cov@20 | 0.568 | 0.566 | −0.002 [−0.045, 0.056] | 0.572 / 0.504 |
| Macarons++（官方，次口径） | AUC / Final | 0.682 / 0.843 | 0.692 / 0.854 | AUC +0.010 [−0.006, 0.034]；Final +0.011 [0.005, 0.018] | 0.710 / 0.880；0.669 / 0.868 |
| Macarons++（官方） | Cov@20 | 未提供 | 未提供 | 未提供 | 未提供 |

MP3D 中期停滞 episode 数（共 10 个）：T0 7，T1 5，B2 4（`V3_COMPARISON_TABLES.md:13`）。原报告的判定为：T1 − T0 在 MP3D 上"有意义"，在 Macarons++ 上"不退化"，合起来记为 "meaningful on one"。T0 − B2 退化。T1 − B2 持平（CI 含 0）。最终决定：问题 1 No-Go，问题 2 Conditional Go（`V3_COMBINED_REPORT.md:65,80`）。

## 3. T1 − T0 能否证明 Scene Cross-Attention 有独立价值
- **支持的一面**：这是设计好的单因素对照（见 §1，同条件），闭环方向一致（MP3D AUC +0.020 / Final +0.044，Macarons++ Final +0.011，6 胜 0 负）。Macarons++ 的 ΔFinal 在单调口径和官方口径下 CI 都不含 0。MP3D VALSEL Early 的 regret 和 Top-1 CI 刚好不含 0 或贴着 0。
- **限制的一面**：
  1. **单种子**：没有种子间方差估计。T1 的共享参数相对 T0 偏离 0.29，两组已经走了不同的训练轨迹，+0.01 ～ +0.02 的差值无法与训练随机性区分。
  2. **场景少**：MP3D 只有 5 个楼层，CI 包含 0。Macarons++ 6 个场景中只有 Final 一项显著，AUC 为 3 胜 3 负，Cov@20 ≈ 0。"有意义"是按 Task 3 §4 的阈值和胜场数判定的，并不要求 CI 不含 0。
  3. **机制证据弱**：G1 全部 ≈ 0，训练 dev 指标每个评估点都在 ±1% 以内。G2 在 MP3D 上有改善，在 Dev-3 上反而变差（regret +0.019，Early +0.045，CI 都含 0）。原报告的归因只做了权重诊断，没有做激活层面的分析（`V3_COMBINED_REPORT.md:78`）。
  4. **场景不独立**：Dev-3 / Test-3 都不是盲集，Dev-3 同时用于 G2 和 G3。另外，系统级 Table 1 中没有登记 T0（`cvpr_results/table1/MODEL_REGISTRY.yaml:538`），因此 All-14 官方口径下没有 T1 − T0 的配对。
- **结论**：这组结果只能说明"在从零训练、40k 步、单种子的设定下，加 XAttn 有一个小的、方向基本一致的闭环增量，可以作为假设"。它**不能**证明 Scene Cross-Attention 有独立价值；要下这个结论，至少需要多种子重复和更多场景。

## 4. CVPR Table 1 系统级对比（All-14，MAGICIAN 官方覆盖；与 §2 不是同一协议）
来源：`cvpr_results/table1/TABLE1_PAPER.md:7,10,12,18`，`reports/RESULTS_ANALYSIS.md:23,29,130,136,243`。All-14 指 14 个场景 × 5 个起点，**其中包含 Train-6 训练场景**。V3 = V3-T1。**V3.1 = V3-T0 结构，只用室外数据（Train-6 去掉 S-c）从零训练 40k 步，accum 2，固定取 EMA@40k**（`MODEL_REGISTRY.yaml:468-481`；`exp6_v3.1/configs/outdoor_train.json:3`）。V3.1 与 T0 的训练数据和域都不同，**不能当作 T0 来读**。
| 模型 | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final | Train-6 AUC@20 / AUC@100 / Final | Dev+Test（8 个场景）AUC@20 / AUC@100 / Final |
|---|---:|---:|---:|---:|---:|---|---|
| MAG-B（Beam 10×10） | 0.307 | 0.544 | 0.711 | 0.537 | 0.906 | 0.326 / 0.761 / 0.945 | 0.293 / 0.674 / 0.877 |
| B2 | 0.388 | 0.601 | 0.742 | 0.598 | 0.907 | 0.422 / 0.771 / 0.922 | 0.362 / 0.721 / 0.896 |
| **V3-T1** | 0.384 | 0.604 | 0.744 | 0.603 | 0.903 | 0.418 / 0.778 / 0.936 | 0.358 / 0.717 / 0.878 |
| **V3.1** | 0.377 | 0.594 | 0.731 | 0.593 | 0.887 | 0.421 / 0.767 / 0.913 | 0.344 / 0.704 / 0.868 |

| 配对（All-14） | ΔAUC@20 | ΔAUC@100 | ΔFinal |
|---|---|---|---|
| V3-T1 − MAG-B | +0.077 [+0.044, +0.112] 12/2 ✱ | +0.033 [+0.007, +0.064] 10/4 ✱ | −0.003 [−0.020, +0.018] 5/9 |
| V3.1 − MAG-B | +0.070 [+0.032, +0.114] 12/2 ✱ | +0.020 [−0.012, +0.055] 9/5 | −0.019 [−0.045, +0.010] 5/9 |
| V3-T1 − B2（系统级） | −0.004 [−0.015, +0.008] 4/10 | +0.001 [−0.019, +0.018] 8/6 | −0.004 [−0.026, +0.011] 8/6 |
| V3.1 − B2 / V3-T1 − V3.1 | 未提供 | 未提供 | 未提供 |

补充两点。第一，V3-T1 在 Dev-3 + Test-3 上的 30 个历史 episode 换算成官方口径是 AUC 0.692 / Final 0.854，单调口径是 0.712 / 0.883（`MODEL_REGISTRY.yaml:300-305`）。第二，V3.1 的原报告值：Macarons++ 单调口径 AUC 0.697 / Final 0.865 / Cov@20 0.551；MP3D zero-shot 0.335 / 0.499 / 0.185；V3.1 − T0 在 MP3D 上 ΔAUC +0.024 [+0.002, +0.052]（4/1）（`exp6_v4/report/FINAL_REPORT.md:11,36`）。**以上都是系统级比较，不能作为 XAttn 消融的证据。**

## 5. 与 F1 的关系
- **V3 的自训练 VAE 不是预训练的 Seen2Scene Sparse VAE。** V3 VAE 是在 4× 稠密聚合 partial TSDF 上的 Conv3D VAE，随机初始化（seed 0），与 Flow 联合从零训练，不加载任何外部 VAE 权重。V3 的任务书明确废弃了"预训练 Sparse VAE / Adapter / 从 V2 微调"这条旧路线（`exp6_v3/EXP6_V3_FROM_SCRATCH_PLAN.md:6,12`；`exp6_v3/protocol.md:32`）。
- **本支线在 F1 中不重启，也不是 F1 的依赖。** F1 的训练、对照和闭环都不需要 V3 / V3.1 的权重或代码；本文件只作为历史参照。

## 来源文件
- `exp6_v3/README.md`、`exp6_v3/protocol.md`、`exp6_v3/STATUS.md`、`exp6_v3/EXP6_V3_FROM_SCRATCH_PLAN.md`
- `exp6_v3/configs/{train_v3.json, model_v3.json, FROZEN_SHA256.json}`
- `exp6_v3/task0_vae/report/T0_TRAINING.md`、`exp6_v3/task1_xattn/report/T1_TRAINING.md`
- `exp6_v3/reports/{V3_COMBINED_REPORT.md, V3_COMPARISON_TABLES.md, g12_s_final.md, g3.md, g3.json}`、`exp6_v3/shared/{analyze_g3.py, out_cov_v3.py}`
- `exp6_v3.1/README.md`、`exp6_v3.1/configs/outdoor_train.json`、`exp6_v4/report/FINAL_REPORT.md`
- `cvpr_results/table1/{TABLE1_PAPER.md, reports/RESULTS_ANALYSIS.md, MODEL_REGISTRY.yaml, PROTOCOL.md}`
