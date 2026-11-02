# EXP6 V5.2-M Scratch70k —— 最终报告

自动生成于 2026-11-02 09:21:58（`scripts/final_report_v52m.py`，数字全部读自阶段输出；评估场景非盲）。人工解读见文末（若有）。

## 0. 运行事实

- 状态：**running**（阶段 train）；完成的同步 optimizer step **35,000 / 70,000**；seen samples 2,240,000 views（全局 batch 64 views / 8 states）。
- 训练轮次 1：r000_1793615980 8 rank 0→… exit …；训练 GPU-h ≈ 45.6。
- Slurm：头节点交互分配 3443137（gpu115），节点 B sbatch 3449325。
- Git：启动时 ，报告提交 —。
- 权重 sha256：—；best_train_holdout = step 35000（select 0.411）；主权重 —。

## 1. Train-6 S-c holdout（选择依据；自然比例校准）

| step | NLL UNK | Top UNK | p_nohit>.95 真命中率 UNK | 已知穿透 | no-hit ECE(nat) | 第二+召回 | 深遮挡召回 |
|---|---|---|---|---|---|---|---|
| 0 | 2.969 | 0.494 | 0.000 | 0.656 | 0.509 | 0.000 | 0.000 |
| 10,000 | 0.488 | 0.841 | 0.001 | 0.033 | 0.004 | 0.821 | 0.769 |
| 20,000 | 0.387 | 0.867 | 0.002 | 0.032 | 0.003 | 0.860 | 0.802 |
| 35,000 | 0.338 | 0.880 | 0.002 | 0.034 | 0.002 | 0.877 | 0.848 |

## 2. 三个问题（机械判读）

G2 最终评估尚无结果，Q1 / Q2 暂不能回答。

**Q3 G3 是否改善（smoke，4 Outdoor + 2 MP3D episode，探索性）。** smoke 有效：否。

| 域 | V5.2-M λ0 | V5.2-M λ0.1 | 旧 V5.2 λ0 | 旧 V5.2 λ0.1 | V5 | V4C | B2 | E2-D100 |
|---|---|---|---|---|---|---|---|---|
| Outdoor Test-3（AUC / Final / Cov20） | — | — | 0.667 / 0.858 / 0.512 | 0.695 / 0.865 / 0.592 | 0.731 / 0.939 / 0.603 | 0.740 / 0.937 / 0.599 | 0.760 / 0.943 / 0.615 | 0.773 / 0.944 / 0.669 |
| MP3D VALSEL | — | — | 0.837 / 0.958 / 0.645 | 0.837 / 0.966 / 0.640 | 0.850 / 0.980 / 0.643 | 0.869 / 0.981 / 0.723 | — | — |

## 3. 门槛与自动决策（FROZEN_PROTOCOL §5）

```json
{}
```

## 4. G3 用户基准（以往实验，AUC / Final）

| 模型 | Outdoor | MP3D |
|---|---|---|
| V5 | 0.711 / 0.891 | 0.333 / 0.501 |
| old V5.2 λ0 | 0.608 / 0.781 | 0.238 / 0.310 |
| B2 | 0.730 / 0.906 | — |
| E2-D100 | — | 0.364 / 0.553 |

## 5. 明细文件

- 训练：`TRAINING_LOG_SUMMARY.md`、`figures/training_dashboard.png`
- G1：`G1_FIRSTHIT.md`；G2：`G2_FIXED_STATE.md`（20k / 40k / 70k / best）
- G3：`G3_SMOKE.md`、`G3_FULL.md`；决策：`AUTO_DECISION.json`；状态：`RUN_STATUS.json`、`STATUS.md`
- 复现：`scripts/submit.sh`（续跑 / 节点 B）、`scripts/overnight_driver.py`、`scripts/slurm/*.sbatch`；冻结协议 `FROZEN_PROTOCOL.md`

**归因限制**：容量、从零训练、2D 上下文、训练分布与几何约束同时改变；结论只针对这套联合干预。

