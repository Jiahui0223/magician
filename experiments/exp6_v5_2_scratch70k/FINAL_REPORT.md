# EXP6 V5.2-M Scratch70k —— 最终报告

自动生成于 2026-11-02 15:39:11（`scripts/final_report_v52m.py`，数字全部读自阶段输出；评估场景非盲）。人工解读见文末（若有）。

## 0. 运行事实

- 状态：**completed**（阶段 done）；完成的同步 optimizer step **70,000 / 70,000**；seen samples 4,480,000 views（全局 batch 64 views / 8 states）。
- 训练轮次 1：r000_1793615980 8 rank 0→70000 exit 0；训练 GPU-h ≈ 90.1。
- Slurm：头节点交互分配 3443137（gpu115），节点 B sbatch 3449325。
- Git：启动时 f105eed，报告提交 8ad0aa65c261。
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

**Q3 G3 是否改善（smoke，4 Outdoor + 2 MP3D episode，探索性）。** smoke 有效：是。

| 域 | V5.2-M λ0 | V5.2-M λ0.1 | 旧 V5.2 λ0 | 旧 V5.2 λ0.1 | V5 | V4C | B2 | E2-D100 |
|---|---|---|---|---|---|---|---|---|
| Outdoor Test-3（AUC / Final / Cov20） | 0.727 / 0.902 / 0.643 | 0.735 / 0.909 / 0.642 | 0.667 / 0.858 / 0.512 | 0.695 / 0.865 / 0.592 | 0.731 / 0.939 / 0.603 | 0.740 / 0.937 / 0.599 | 0.760 / 0.943 / 0.615 | 0.773 / 0.944 / 0.669 |
| MP3D VALSEL | 0.835 / 0.954 / 0.696 | 0.845 / 0.959 / 0.696 | 0.837 / 0.958 / 0.645 | 0.837 / 0.966 / 0.640 | 0.850 / 0.980 / 0.643 | 0.869 / 0.981 / 0.723 | — | — |

**G3 Full**：skipped —— G3 smoke only (G2 gate not met)。

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
 "decision": "G3 smoke only (G2 gate not met)",
 "g3_smoke_valid": false,
 "g3_smoke_valid_after_rerun": true,
 "note_after_rerun": "2026-11-02 15:34: 6 smoke items that failed for infrastructure reasons (FROZEN_PROTOCOL §9) were rerun with the same checkpoint / score config; the smoke is valid after the rerun. The decision is unchanged: G3 full is skipped because the G2 gate failed (MP3D Top-1 0.378 < 0.40; Dev-3 high-confidence false no-hit 0.560 not below old V5.2 0.541)."
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

## 6. 人工解读

（人工解读，2026-11-02 15:45；所有数字来自本报告上文和 `G1_FIRSTHIT.md` / `G2_FIXED_STATE.md` / `G3_SMOKE.md` / `eval/diag_nohit_best/summary.md`；评估场景非盲）

### 一句话结论
V5.2-M 从零训练到 70k、参数 27.0M，比旧 V5.2 明显更好：
- 多表面 belief 在两个域都大幅改善；
- Outdoor G2 Top-1 和 G3 smoke 都高于旧 V5.2。

但它仍没有超过 V5、B2、E2：
- **首中 / no-hit 头**在评估场景上越训越差：训练场景 holdout 一直在变好，评估场景 UNKNOWN NLL 却在上升；
- **MP3D**：G2 Top-1 只有 0.378，未达 0.40 门槛；
- **Dev-3**：高置信 no-hit 误判没有改善。

按冻结门槛 **G3 full 跳过**，只有 smoke。下面所有结论只对应这次联合干预，即容量、从零训练、2D 上下文、训练分布、几何约束同时改变。

### 1. 训练本身是健康的（Train-6 S-c holdout）
- 70,000 步一轮跑完，8 rank，约 11.3 h，约 90 GPU-h，没有 OOM、NaN 或重启。
- holdout UNKNOWN NLL：2.97 → 0.32；最高区间准确率 0.88；no-hit ECE 0.003；已知表面穿透 0.036；第二+表面召回 0.89；深遮挡召回 0.86。
- 50k 之后进入平台期。高置信 no-hit 误判从 0.0024 升到 0.0038，因此按预注册分数，best = 50k。

### 2. G1：表面 belief 跨场景泛化好，首中 / no-hit 头不好
**多表面 belief**（评估网格 GT，best 权重 vs 旧 V5.2）

| 指标 | MP3D | Dev-3 B0 |
|---|---|---|
| 首表面召回 | 0.783 vs 0.328 | 0.721 vs 0.316 |
| 第二+召回 | 0.483 vs 0.274 | 0.448 vs 0.306 |
| 深遮挡（间隔 ≥128 vox）召回 | 0.382 vs 0.129 | 0.336 vs 0.159 |
| 网格 ECE | 0.058 vs 0.156 | 0.057 vs 0.098 |

首表面后的虚假表面率与旧 V5.2 持平（0.20–0.23）。这是这轮最明确的正面结果：2D 上下文加深度 decoder 学到的表面 belief 能迁移到评估场景。

**首中 PMF**（固定状态所有候选射线 vs oracle t*）
- 最高区间准确率约翻倍：
  - MP3D UNKNOWN 0.363 vs 0.177；
  - Dev-3 UNKNOWN 0.482 vs 0.269。
- 但 NLL 更差，因为错的时候更自信：
  - MP3D UNKNOWN 7.32 vs 5.35；
  - Dev-3 UNKNOWN 5.32 vs 4.78。
- 里程碑走势：MP3D UNKNOWN NLL 20k 5.52 → 40k 6.73 → best 7.32。同期 Train-6 holdout NLL 从 0.39 降到 0.32。首中头在过拟合训练场景的几何统计。

### 3. Q1：MP3D UNKNOWN 过度自信 no-hit **没有实质减少**
- **按预注册的比例**：p_nohit ∈ [0.9, 1] 的 UNKNOWN 射线中实际命中的比例从 0.951 降到 0.932，机械判读为“是”。
- **按绝对量**：这类射线从 23.9 万条增加到 40.6 万条。高置信错判数约从 22.8 万增加到 37.9 万，NLL 也更差。**实质上没有减少**。
- **只读诊断**（best 权重，全部 90 态）。MP3D UNKNOWN 中 p_nohit ≥ 0.9 的射线拆成：
  - 网格内命中：49%；
  - 命中点在规划网格外（7.8 m 区域裁剪之外）：44%；
  - 真 no-hit：7%。
- 解读：
  - 大约一半的误判来自**区域裁剪**——网格外的墙模型看不到。训练用的室外 Train-6 网格覆盖整个场景，“出网格 = 无命中”在训练中成立，到室内裁剪区域却不成立。这类射线打分时本来就不计入网格外体素，对选点影响有限。
  - 另一半是**网格内真正的误判**，这才是需要解决的泛化问题。
- **Dev-3**：false no-hit 比例 0.560 vs 旧 0.541，没有改善，其中网格外命中只占 14%。

### 4. Q2：Outdoor G2 Top-1 提升（相对旧 V5.2），但不如 V5 / B2 / E2
Dev-3 275 态（λ0）：

| 模型 | Top-1 | 增益 pp |
|---|---|---|
| V5.2-M λ0 | 0.471 | 1.107 |
| V5.2-M λ0.1 | 0.494 | — |
| 旧 V5.2 | 0.411 | 0.902 |
| V5 | 0.540 | 1.145 |
| B2 | 0.581 | 1.275 |
| E2 | 0.542 | 1.214 |

配对差（3 场景 CI）：
- 相对旧 V5.2：Top-1 +0.060 [+0.034, +0.083]，增益 +0.205 [+0.030, +0.434]，显著；
- 相对 V5：Top-1 −0.068 [−0.113, −0.008]；
- 相对 B2：增益 −0.168 [−0.201, −0.123]。

MP3D VALSEL：V5.2-M 0.378 vs 旧 0.356 vs V5 0.467 vs B2 0.533；增益与旧 V5.2 持平（−0.016，CI 含 0）。

### 5. Q3：G3（只有 smoke，4 Outdoor + 2 MP3D episode，探索性）
- **Outdoor**：V5.2-M λ0 / λ0.1 为 AUC 0.727 / 0.735，Final 0.902 / 0.909。
  - 相对旧 V5.2（0.667 / 0.858）有改善；
  - 与 V5（0.731 / 0.939）的 AUC 相当，Final 更低；
  - 仍低于 B2（0.760）和 E2（0.773）；
  - Cov@20 最高（0.643），早期探索更快。
- **MP3D**（150 步，C_obs 口径）：V5.2-M 0.835 / 0.845，对比旧 V5.2 0.837、V5 0.850、V4C 0.869，没有改善。
- **λ 的作用**：λ 0.1 的影响很小（翻转率 0–7%）。
- **episode 数**：smoke 只有 6 个 episode，不能作为正式结论。按冻结门槛没有运行 G3 full，所以对“G3 是否改善”只能回答：**smoke 上 Outdoor 改善于旧 V5.2、接近 V5，MP3D 没有改善；未经过 Full 检验**。

### 6. 代价
- **打分时间**：每状态 0.77–0.88 s，规划器每步 0.9–1.4 s，是旧 V5.2（约 0.12 s）的约 8–10 倍。原因是每个候选视角都要解码整幅 64×114 射线网格。
- **训练时间**：8 卡 0.57 s/step，4 → 8 卡加速 1.74 倍（DDP 效率 87%），显存 6.8 GB / rank。

### 7. 对下一步的含义（建议，待用户决定，未执行）
1. **把表面 belief 和首中头分开看**。belief 已有强迁移性，可以考虑直接用 belief b 构造可见性，或给首中头加跨场景正则，而不是只靠更大的模型。
2. **训练分布要包含“网格外仍有表面”的情形**（裁剪区域的训练样本），或把出网格的质量单独作为“未知”而非 no-hit。这需要改协议，应作为新实验。
3. **只用 Train-6 选权重会选到过拟合点**：本轮 best 在 50k，但 20k 的评估指标更好。可以考虑在预注册的独立场景 holdout 上选权重——这同样需要新协议，不能事后改本轮的选择。

