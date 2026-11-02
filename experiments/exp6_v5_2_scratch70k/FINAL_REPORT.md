# EXP6 V5.2-M Scratch70k —— 最终报告

自动生成于 2026-11-02 15:27:03（`scripts/final_report_v52m.py`，数字全部读自阶段输出；评估场景非盲）。人工解读见文末（若有）。

## 0. 运行事实

- 状态：**completed**（阶段 done）；完成的同步 optimizer step **70,000 / 70,000**；seen samples 4,480,000 views（全局 batch 64 views / 8 states）。
- 训练轮次 1：r000_1793615980 8 rank 0→70000 exit 0；训练 GPU-h ≈ 90.1。
- Slurm：头节点交互分配 3443137（gpu115），节点 B sbatch 3449325。
- Git：启动时 ，报告提交 —。
- 权重 sha256：`step_070000.pt` e1b168945323a21c…；`best_train_holdout.pt` a540950738cc7191…；best_train_holdout = step 50000（select 0.398）；主权重 final_best。

## 1. Train-6 S-c holdout（选择依据；自然比例校准）

| step | NLL UNK | Top UNK | p_nohit>.95 真命中率 UNK | 已知穿透 | no-hit ECE(nat) | 第二+召回 | 深遮挡召回 |
|---|---|---|---|---|---|---|---|
| 0 | 2.969 | 0.494 | 0.000 | 0.656 | 0.509 | 0.000 | 0.000 |
| 10,000 | 0.488 | 0.841 | 0.001 | 0.033 | 0.004 | 0.821 | 0.769 |
| 20,000 | 0.387 | 0.867 | 0.002 | 0.032 | 0.003 | 0.860 | 0.802 |
| 40,000 | 0.332 | 0.881 | 0.003 | 0.035 | 0.002 | 0.879 | 0.852 |
| 70,000 | 0.326 | 0.888 | 0.004 | 0.037 | 0.003 | 0.887 | 0.863 |

## 2. 三个问题（机械判读）

**Q1 MP3D UNKNOWN 过度自信 no-hit 是否减少：是。** VALSEL UNKNOWN 射线中 p_nohit ∈ [0.9, 1] 的射线：旧 V5.2 239,321 条，其中实际命中 0.951；V5.2-M 406,238 条，实际命中 0.932。V5.2-M p_nohit > 0.95 占真实命中射线的比例 0.214；UNKNOWN 首中 NLL 旧 5.348 → 新 7.323，最高区间准确率 0.177 → 0.363。（Dev-3 同口径：旧 0.541（839,876 条）→ 新 0.560（978,791 条）；NLL 4.783 → 5.317。）

**Q2 Outdoor G2 Top-1 是否提升：是（相对旧 V5.2）。** Dev-3 275 态 Top-1：V5.2-M λ0 0.471 / λ0.1 0.494；旧 V5.2 λ0 0.411；V5 0.540；配对差 V5.2-M λ0 − 旧 V5.2 λ0 = +0.060 [+0.034, +0.083]（3 场景），− V5 = -0.068 [-0.113, -0.008]（3 场景）；增益（pp）配对差 − 旧 V5.2 +0.205 [+0.030, +0.434]（3 场景），− V5 -0.038 [-0.068, +0.012]（3 场景）。MP3D VALSEL Top-1：V5.2-M λ0 0.378，旧 V5.2 0.356，V5 0.467。

**Q3 G3 是否改善（smoke，4 Outdoor + 2 MP3D episode，探索性）。** smoke 有效：否。

| 域 | V5.2-M λ0 | V5.2-M λ0.1 | 旧 V5.2 λ0 | 旧 V5.2 λ0.1 | V5 | V4C | B2 | E2-D100 |
|---|---|---|---|---|---|---|---|---|
| Outdoor Test-3（AUC / Final / Cov20） | 0.727 / 0.902 / 0.643 | — | 0.667 / 0.858 / 0.512 | 0.695 / 0.865 / 0.592 | 0.731 / 0.939 / 0.603 | 0.740 / 0.937 / 0.599 | 0.760 / 0.943 / 0.615 | 0.773 / 0.944 / 0.669 |
| MP3D VALSEL | — | — | 0.837 / 0.958 / 0.645 | 0.837 / 0.966 / 0.640 | 0.850 / 0.980 / 0.643 | 0.869 / 0.981 / 0.723 | — | — |

**G3 Full**：skipped —— G3 smoke only (G2 gate not met) (smoke invalid)。

## 3. 门槛与自动决策（FROZEN_PROTOCOL §5）

```json
{
 "rule": "FROZEN_PROTOCOL §5: G3 full iff Outdoor Dev-3 G2 Top-1 (λ=0, 275 states) ≥ 0.46 AND MP3D VALSEL G2 Top-1 (λ=0, 90 states) ≥ 0.40 AND UNKNOWN-ray high-confidence false no-hit share (rays with p_nohit in [0.9, 1] that truly hit) below old V5.2 in both domains; AND the G3 smoke is valid",
 "thresholds": {
  "outdoor_top1_min": 0.46,
  "mp3d_top1_min": 0.4
 },
 "time": "2026-11-02 15:07:47",
 "outdoor_top1_l0": 0.47133597883597883,
 "mp3d_top1_l0": 0.3777777777777777,
 "false_nohit": {
  "dev3": {
   "new_false_nohit_share_p09": 0.5598110321815383,
   "new_rays_p09": 978791,
   "old_false_nohit_share_p09": 0.540791735922922,
   "old_rays_p09": 839876,
   "new_hc_false_nohit_rate_p095": 0.13544144475171957,
   "improved": false
  },
  "valsel": {
   "new_false_nohit_share_p09": 0.9322539988873517,
   "new_rays_p09": 406238,
   "old_false_nohit_share_p09": 0.951316432740963,
   "old_rays_p09": 239321,
   "new_hc_false_nohit_rate_p095": 0.21351692196322744,
   "improved": true
  }
 },
 "g2_gate": "FAIL",
 "decision": "G3 smoke only (G2 gate not met) (smoke invalid)",
 "g3_smoke_valid": false
}
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

