# EXP6 — V3 Combined Report: Simple 3D VAE vs 3D VAE + Cross-Attention

> STATUS（2026-10-31 04:30）：**COMPLETED** — Phase 1–5 全部完成；G0 ✅、T0 / T1 从零训练 ✅（各 40k 步）、G1 / G2 ✅（两组均 PASS）、G3 ✅（formal）。本文件按 `../EXP6_V3_REPORT_TEMPLATE.md` 结构填写；所有数字来自真实运行（来源：`g3.md` / `g12_s_final.md` / `T*_TRAINING.md` / run manifests），每节标注 Completed / Exploratory / Not Run。
> 真相来源：`../protocol.md` v1.0（2026-10-30 06:10 冻结）、`../configs/FROZEN_SHA256.json`、run manifests（`work_dirs/exp6_v3/task*/runs/main/run_*.json`）、`task*/checkpoints/final.{pt,json}`、`work_dirs/exp6_v3/eval{,_fixed}/`。偏差日志 `V3_DEVIATIONS.md`（#1–#10）。统一表格 `V3_COMPARISON_TABLES.md`。

## 1. Executive Summary（Completed）
- **问题 1 — 从零训练的 3D VAE partial-TSDF 表示（T0）能否改善未来视角生成与闭环 NBV？** 固定状态上部分改善、闭环上**否**。T0 相对冻结 V2-Mix（B2）把 MP3D 未知射线的未来深度 MAE 降低 17%（−0.038 [−0.052, −0.025]，9/9 episode），排序 regret 更低（VALSEL 0.170 vs 0.217，Dev-3 0.343 vs 0.464，CI 含 0），G2 PASS；但闭环按冻结标准**退化**：MP3D ΔAUC −0.023 [−0.048, −0.000]，Macarons++ ΔFinal −0.035 [−0.076, −0.005]；与微调的 E2-D100 差距更大（MP3D ΔAUC −0.054）。
- **问题 2 — 同预算下一处 Camera-Ray Cross-Attention（T1）是否有可测收益？** **是，小而方向一致**。T1 − T0 闭环：MP3D ΔAUC **+0.020 [−0.003, 0.042]**（4/1 场景胜）、ΔFinal +0.044；Macarons++ ΔAUC +0.010、ΔFinal **+0.011 [0.005, 0.017]**（6/0）；固定状态 Early regret **−0.036 [−0.094, −0.002]**、Early Top-1 +0.056（VALSEL）。按冻结标准 **"meaningful on one"（MP3D）**。生成质量（G1）与 T0 逐项相同 → 增益来自早期 / 空地图阶段的候选判断，不是更准的逐视角深度。
- **总体结论**：V3 从零路线只有加上 Cross-Attention 才**追平**冻结 V2-Mix（T1 − B2：MP3D −0.003 [−0.049, 0.037]，Macarons++ −0.018 [−0.052, 0.006]），仍低于微调的 E2-D100（MP3D −0.033 [−0.083, −0.004]）；两组在 MP3D 上均优于 MAGICIAN 官方模型（T1 − MAG-B Final **+0.131 [0.044, 0.218]**，5/0），Macarons++ 上 Cov@20 +0.062 [0.017, 0.110]。**决定：V3 不替代主线（E2-D100 / B2 规划器）**；Cross-Attention 作为经证实的小增量保留为后续候选（叠加到预训练 V2 线上，未执行）。
- 负面结果与偏差：T0 闭环退化（上）；MP3D 中期停滞 episode T0 7 / T1 5 / B2 4 / E2D 2（规划区域裁剪瓶颈，与 T0.12-C 一致）；10 个工程 / 调度偏差（`V3_DEVIATIONS.md`），其中 #9 G3 先于 G1 / G2 执行（仅调度）、#10 G1 / G2 第一轮被一次性非确定性触发 STOP 后按原规则补跑通过。无协议数值修改，无 GT 泄漏，Test-3 未用于调参。

## 2. Experimental Protocol and Data Provenance（Completed）
| Split | 内容 | 轨迹 / 状态 | 候选视角 | 独立性 |
|---|---|---|---|---|
| MP3D train | 61 个官方 train 房屋 114 个楼层（E2 轨迹作业） | 228 条 × 150 状态 | 66,803（D100 分片） | 场景级与 val / test 不交 |
| MP3D VALSEL（选择 + G1 / G2） | 6 个 val 房屋 9 个楼层 | 9 条；G1 / G2 用 90 个固定状态 | 10,473 | 与 train 不交，不是闭环楼层 |
| MP3D 闭环（G3） | 5 个 val 楼层 × 2 起点 × 150 步 | 10 episode | — | T0.11 可视化挑选，非盲 |
| Macarons++ train | Train-6 去掉 S-c 起点 | 138 条 | 25,225 | |
| Macarons++ S-c（选择） | Train-6 同场景留出起点 | 42 条 | 31,027 | **同场景留出起点，不是未见场景** |
| Macarons++ Dev-3（G1 / G2） | barts / pantheon / colosseum，E1.2 的 B0 层 150 + B2 层 125 个状态 | 275 状态 | — | 开发集，非盲 |
| Macarons++ 闭环（G3） | Test-3（alhambra、bridge、neuschwanstein）+ Dev-3 × 5 起点 × 100 步 | 30 episode | — | Test-3 已在 Task 2 使用，**非盲** |
- 输入：真实 Partial TSDF（历史 0..t 的 D100 融合：BAND 全部有效像素、EMPTY 挖空不变、UNKNOWN 哨兵）→ 4× 状态感知聚合 `[sdf, band, free, dom]`；DistP1 打分始终用 10% 稀疏 P1；预测不写回。无泄漏核对：缓存构建时 417 / 417 条轨迹（62,550 个状态）的 V2 粗网格与 E2 分片逐位相同；G1 / G2 中 B2 回放 365 / 365 个状态与 E2 行逐位相同。
- GT 只用于评价、标签、碰撞 / 视线与覆盖率；教师与规划器不接触 GT。物理尺度：MP3D 体素 8.7–17.6 mm；Macarons++ 0.11–0.33 scene units（非米）。

## 3. Architecture, Losses and Scratch Training（Completed）
| 模块 | T0 V3-VAE | T1 V3-VAE-XAttn |
|---|---|---|
| VAE Conv3D 编码器（μ, logvar 16 ch @8×，latent 单元 = 8 fine 体素） | scratch，1,0xx k | 相同，公共初始化 sha256 `6a9d6071a225…` 相同 |
| 3D 解码器（仅训练，重建观测状态） | 585,188 | 相同 |
| Ray Encoder（V2 结构，沿射线采样 μ） | scratch | scratch |
| 2D Rectified-Flow U-Net（V2 结构，K=4，10 Euler） | scratch | scratch |
| Camera-Ray Cross-Attention（瓶颈 8 × 16，K/V = ≤ 512 个视锥内 μ token + Fourier 位置，`W_o` 零初始化） | 无 | 191,364 |
| 参数（推理 / 总） | 22,954,434 / 23,539,622 | 23,145,798 / 23,730,986 |
训练（协议 v1.0）：`L = L_FM + 1.0·L_recon + β(step)·KL`，β 0 → 0.02（5k 步）；AdamW lr 1e-4（预热 1k、余弦）、bf16、EMA 0.999、全局 64 视角 / 步（32 + 32）、40,000 步、种子 0；两组相同采样索引 / 噪声 / 步数；第 1 步损失逐位相同（3.674514578）。选择：score = ES_in/ES_in(5k) + ES_out/ES_out(5k) 取最小 → 两组均单调下降并选 **EMA @40k**（T0 1.380、T1 1.395）；无早停。后验健康：16 / 16 活跃通道、μ std 0.51、KL/unit 0.29。T1 的 `W_o` 终值 rms 0.006（相邻投影的 1/6），共享参数与 T0 相对差 0.29（`T1_TRAINING.md` §5）。

## 4. G0（Completed，2026-10-30 06:18）
无泄漏（417 / 417 粗网格逐位）；三态 / 对齐 / 分割 / 梯度 / XAttn 正确性单元测试 6 passed；tiny overfit T0 −83% / T1 −84%；干净 4 卡 profile T0 0.94 / T1 0.96 s/步（正式训练实测 1.34 / 1.37 s/步，含大场景）；dunnottar 单进程 1.83 s / 27.4 GB，不分块；DDP resume 通过（修复 RNG 恢复 bug，偏差 #8）。详见 `../task0_vae/report/T0_G0_TESTS.md`、`../task1_xattn/report/T1_G0_TESTS.md`。

## 5. G1 — 未来视角生成（Completed；VALSEL 90 状态 / Dev-3 275 状态；表 3）
| 指标（VALSEL，全部状态） | B2 | E2-D100 | T0 | T1 |
|---|---:|---:|---:|---:|
| 深度 MAE all / unknown / new / known | 0.150 / 0.230 / 0.848 / 0.102 | **0.132 / 0.179 / 0.831 / 0.084** | 0.138 / 0.191 / 0.862 / 0.089 | 0.139 / 0.193 / 0.860 / 0.089 |
| K=4 rel_std / hit_dis / 离散度–误差 ρ | 0.026 / 0.044 / 0.424 | 0.023 / 0.042 / 0.377 | 0.025 / 0.041 / 0.392 | 0.025 / 0.041 / 0.390 |
T0 − B2：MAE_unknown −0.038 [−0.052, −0.025]（9/0），MAE_known −0.012 [−0.018, −0.007]（9/0）；T0 − E2D：MAE_unknown +0.012 [0.006, 0.018]，MAE_new +0.031 [0.014, 0.048]；**T1 − T0 全部 ≈ 0（CI 含 0）**。Dev-3：MAE_new T0 5.545 / T1 5.561 vs B2 5.886（−0.341 [−1.022, 0.037]），其余与 B2 持平。多样性无坍缩。

## 6. G2 — 候选排序（Completed；表 3、表 4）
| 集合 / 阶段 | 指标 | B2 | E2-D100 | T0 | T1 | T1 − T0 |
|---|---|---:|---:|---:|---:|---|
| VALSEL all | regret / Top-1 / ρ / margin | 0.217 / 0.533 / 0.503 / 0.356 | 0.178 / 0.567 / 0.510 / 0.359 | 0.170 / 0.567 / 0.541 / 0.376 | **0.155 / 0.578 / 0.546 / 0.383** | regret −0.014 [−0.038, 0.000] |
| VALSEL Early（36） | regret / Top-1 | 0.500 / 0.389 | 0.404 / 0.472 | 0.385 / 0.417 | **0.348 / 0.472** | **−0.036 [−0.094, −0.002] / +0.056 [0.000, 0.139]** |
| VALSEL Middle / Late | regret | 0.038 / 0.008 | 0.037 / 0.007 | 0.036 / 0.006 | 0.036 / 0.008 | ≈ 0 |
| Dev-3 all | regret / Top-1 | 0.464 / **0.581** | 0.525 / 0.542 | **0.343** / 0.548 | 0.361 / 0.552 | +0.019 [−0.009, 0.071] / +0.004 [0.000, 0.010] |
| Dev-3 Early（95） | regret | 1.174 | 1.323 | **0.740** | 0.785 | +0.045 [−0.031, 0.161] |
G2 → G3 放行：**T0 PASS（规则 B：Dev-3 Top-1 0.548 < 0.551 差 0.003 未满足规则 A）；T1 PASS（规则 A）** → 两组 G3 均为 formal。

## 7. G3 — 闭环（Completed，formal；表 1、表 2）
| Dataset | Metric | MAG-G | MAG-B | B0 | B2 | E2-D100 | **V3-T0** | **V3-T1** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MP3D（10 ep） | AUC@150 | 0.278 | 0.292 | 0.322 | 0.334 | **0.364** | 0.311 | 0.331 |
| MP3D | Final | 0.377 | 0.377 | 0.469 | 0.530 | **0.553** | 0.463 | 0.508 |
| MP3D | Cov@20 / @50 / @100 | 0.169 / 0.266 / 0.336 | 0.165 / 0.276 / 0.360 | 0.195 / 0.298 / 0.375 | 0.197 / 0.276 / 0.398 | 0.195 / 0.297 / 0.452 | 0.183 / 0.267 / 0.382 | 0.184 / 0.282 / 0.391 |
| Macarons++（30 ep） | AUC@100 | 0.704 | 0.685 | 0.729 | **0.730** | 0.711 | 0.702 | 0.712 |
| Macarons++ | Final | 0.905 | 0.891 | 0.884 | **0.906** | 0.880 | 0.872 | 0.883 |
| Macarons++ | Cov@10 / @20 / @50 | 0.299 / 0.507 / 0.815 | 0.298 / 0.504 / 0.776 | **0.400 / 0.622 / 0.824** | 0.389 / 0.572 / 0.822 | 0.364 / 0.570 / 0.808 | 0.385 / 0.568 / 0.787 | 0.368 / 0.566 / 0.804 |
配对差与判定见表 2。要点：**T1 − T0** MP3D +0.020 AUC（4/1）/ +0.044 Final，Macarons++ +0.011 Final（6/0）→ "meaningful on one"；**T0 − B2** 退化（MP3D −0.023 AUC，Macarons++ −0.035 Final）；**T1 − B2** 持平（CI 含 0；Final 均值 −0.022 / −0.024 按阈值记退化）；**T1 − E2D** MP3D −0.033 [−0.083, −0.004]；**T1 − MAG-B** MP3D Final +0.131 [0.044, 0.218]（5/0）。行为：MP3D 中期停滞 episode MAG 7–8 / B0 3 / B2 4 / E2D 2 / T0 7 / T1 5；每步决策 ≈ 1 s（V3）vs 2.6 s（B2，历史记录，不同负载不可直接比）；峰值显存 MP3D 18.5–18.7 GB、Macarons++ 5.7 GB。

## 8. Main Comparisons and Attribution（Completed）
1. **T1 vs T0（主比较，唯一差别 = Cross-Attention）**：G1 相同 → 增量不来自生成质量；G2 Early regret −0.036、Early Top-1 +0.056（MP3D），中后期 ≈ 0；G3 MP3D 停滞 episode 7 → 5，AUC +0.020、Final +0.044；Macarons++ Final +0.011（6/0）。归因：场景记忆 token 在地图最空、射线采样到的 μ 信息最少的早期阶段提供视锥外上下文；权重诊断（`W_o` rms 0.006、共享参数差 0.29）证明支路有效而非零。
2. **T0 vs B2（表示：从零 V3-VAE vs 预训练 V2-Mix，同规划器）**：固定状态更好（MAE_unknown −17%，regret 更低），闭环更差（MP3D −0.023 AUC）；差异集中在 MP3D 中后期停滞（7 vs 4 个 episode）→ 闭环瓶颈仍是规划区域裁剪下对低增益候选的高估（T0.12-C），逐状态排序优势无法兑现。
3. **vs E2-D100**：E2-D100（预训练 V2 + D100 微调 5k 步）仍是 MP3D 最佳（0.364 / 0.553），V3 两组均低于它；Macarons++ 上 T1 与 E2D 持平。
4. **vs MAGICIAN**：V3 两组在 MP3D 上均高于 MAG-G / MAG-B（T1 Final +0.131，CI 不含 0）；Macarons++ AUC 持平、Cov@20 更高（+0.062）、Final 略低（MAG-G 0.905 vs T1 0.883）。
5. **成本**：V3 推理每步 ≈ 1 s、显存 ≤ 19 GB；训练 64–65 GPU-h / 组。

## 9. Training & Engineering Costs（Completed；`V3_RESOURCES.md`）
缓存构建 ≈ 22.6 GPU-h（与冒烟共享）；G0 ≈ 1.1；T0 训练 63.7；T1 训练 65.0；G3 3.1 + 3.3；G1 / G2 ≈ 1.9（含一次性 STOP 重跑）；**合计 ≈ 161 GPU-h**（L40S）。存储：4× 缓存 49 GB，checkpoint 360 MB + 9 × 90 MB / 组，闭环与固定状态输出 < 5 GB。墙钟：2026-10-29 Phase 1 → 2026-10-31 04:15 全部评估完成。

## 10. Limitations, Deviations and Final Decision（Completed）
- 局限：MP3D 闭环只有 10 个 episode（5 楼层），CI 宽；Dev-3 / Test-3 均非盲；单种子、单预算（40k 步，非上限 60k；两组选择分数到 40k 仍在下降，更长训练的效果未知）；Cross-Attention 的 token 预算（512）与视锥规则未扫描；激活层面的归因只做了权重诊断；B2 / E2-D100 的闭环为 Task 3 历史记录（同协议，未重跑）。
- 偏差（`V3_DEVIATIONS.md` #1–#10）：均为工程 / 调度项，无协议数值改动；#9 G3 先于 G1 / G2（G3 无论 G2 都要跑，标签事后补，两组 G2 均 PASS → formal）；#10 G1 / G2 第一轮 7 / 12 part 被冻结 STOP 规则终止（B2 回放一次性非逐位），原样重跑逐位通过，按原规则补跑，未引入容差。
- **最终决定**：(a) 问题 1 **No-Go**：从零训练的 V3-VAE 表示不改善闭环（相对 B2 退化、远低于 E2-D100）；(b) 问题 2 **Conditional Go（证据级）**：Cross-Attention 在同预算下给出小而一致的闭环增量（MP3D "有意义"，Macarons++ 不退化），使从零路线追平冻结 V2；(c) 主线保持 E2-D100 / B2 规划器不变；**建议的后续（未执行、需另行批准）**：把 Camera-Ray Cross-Attention 作为场景记忆支路叠加到预训练 V2 / E2-D100 线上做同样的 T1 − T0 配对实验；规划侧解决 MP3D 中期停滞（规划区域裁剪，T0.12-C）。
