# EXP6 V3 协议 v1.0（2026-10-30 06:30 冻结；正式训练前的最终版本，之后的改动只能记入 `reports/V3_DEVIATIONS.md`）

原 v0.9 草案中标注"G0 后定"的数值已由 G0（`task0_vae/report/T0_G0_TESTS.md`）校准并在此冻结；两组相同。配置哈希：`configs/FROZEN_SHA256.json`。

## 1. 数据
| 项 | 定义 |
|---|---|
| 训练集 | MP3D：E2 轨迹作业中 `purpose = train` 的 228 条（114 个楼层）；Macarons++：Train-6 去掉 S-c 的 138 条（`heldout = False`） |
| 选择 / 验证集 | MP3D VALSEL 9 条（6 个 val 房屋）；Macarons++ S-c 42 条（`heldout = True`） |
| 样本 | (scene, trajectory, t, candidate)；一步的全部候选来自同一状态；状态 t 的输入只含记录 0..t |
| 射线通道 / 标签 | E2 D100 分片（冻结构建器）：`cls, tk, dlo, dhi, lu, nseg` 与 `b, y, cat`；监督掩码 `lmask = UNKNOWN ∧ b ≠ 0` |
| VAE 输入 | `shared/build_tsdf_cache.py`：D100 融合的 4× 聚合 `[sdf, band, free] + valid`（`seen2scene.v3.partial_tsdf`）；每状态与分片粗网格逐位核对 |
| 陈旧度 | 同 V2 / E2：编码器输入可比射线通道早 Δ ∈ [1, 9] 步（p = 0.5），MP3D 不跨刷新块 |
| 采样 | 同 V2 / E2：每步每卡每域 1 个状态、8 个视角（全局 32 + 32 视角）；状态权重 t ≤ 20 ×2；RNG `[seed, rank]`，两组相同 |
| 评分地图 | DistP1 始终用原始 10% 稀疏 P1（Task 2 规划器不变） |

## 2. 模型（`configs/model_v3.json`）
- 输入 4× 网格（细体素 VS = 0.011 模型单位，填充到 8 的倍数）；通道 `[sdf/TRUNC, band 比例, free 比例]`，valid 作第 4 通道并用于掩码。
- VAE：stem 32 @4× → Res3D → stride-2 64 @8× → 2 × Res3D → 1×1 → μ / logvar（16 通道，每格 8 个细体素）。Decoder（仅训练）：64 → 2 × Res3D → ×2 上采样 32 → Res3D → `[sdf_hat, 3 类 logits]`。
- Ray Encoder：V2 的 Conv1d 结构，32 个采样点，每点 `[μ (16), valid8·inside]`。
- Flow U-Net：V2 结构（64/128/192/256，瓶颈自注意力）；T1 在瓶颈自注意力之后加一处 Cross-Attention（Q = 瓶颈 token + 射线方向 / 相机位置 Fourier PE；K/V = 有观测支持的潜变量格 `[μ, band8, free8] + 世界坐标 PE`；每候选视锥内 ≤ 512 token；输出投影零初始化）。
- 推理：K = 4、10 步 Euler、确定性解码、DistP1、Greedy-NV，全部沿用 Task 2。
- 整图卷积，不分块：G0 实测最大网格（dunnottar 278 × 398 × 64）单进程峰值 27.4 GB < 46 GB，不做 tiling（无拼缝）。

## 3. 损失与训练（`configs/train_v3.json`，sha256 见 `configs/FROZEN_SHA256.json`）
- `L = L_FM + λ_rec · L_recon + β(step) · KL`；L_FM 为全局有效射线均值（DDP 下 count 全局求和）；L_recon、KL 为状态均值（DDP 下按 world 平均）。
- **λ_rec = 1.0**：初始 FM 1.8–1.9 与 Recon 1.7–1.9 同量级（G0 #6 / #7），不额外加权。
- **β_max = 0.02，从第 0 步线性预热到第 5,000 步后恒定**：KL/unit 初始 0.08，无压力时 200 步内升到 0.31（G0 #7）；β = 0.02 使 KL 项约为重建项的 1–4%，保持真实 VAE 后验而不压制 Flow 的条件信息；训练中记录 KL/unit、μ 标准差、active units，若 active units 降到 < 8 则在报告中标注后验坍缩（不改 β）。
- 优化：AdamW（β 0.9 / 0.99，wd 0.01），峰值 lr 1e-4，预热 1k 步，余弦到 0；grad clip 1.0；bf16；EMA 0.999；seed 0；每步全局 64 视角（32 MP3D + 32 Macarons++，4 卡每卡 8 + 8，来自 1 + 1 个状态），无梯度累积。
- **预算：40,000 步**（两组相同）。依据：干净 4 卡 profile 0.94–0.96 s/步（小场景子集），含 dunnottar 等大场景（单进程 1.83 s、27 GB）后按 rank 最大值估计约 1.3–1.4 s/步 → 每组约 15 h；两组顺序 + G1–G3 可在当前 4 卡分配（2026-11-01 14:50 到期）内完成并留余量。60k 的上限候选未采用（两组顺序需 > 46 h，无余量）。
- 评估 / 保存：每 5,000 步（含第 0 步）在 VALSEL（2,048 视角）、S-c（2,048）、训练子集上评估并保存 EMA；`ckpt_last.pt` 每 1,000 步（可恢复：optimizer / EMA / RNG）。
- **初始化**：两组同 seed 从零构造；公共模块参数 sha256 相同（G0 实测 `6a9d6071a225…`，写入 run manifest）；T1 的 Cross-Attention 独立初始化，输出投影为零；不加载任何已训练权重。
- **checkpoint 选择规则（冻结）**：在 VALSEL 与 S-c 上，score = ES_in / ES_in(step 5k) + ES_out / ES_out(step 5k)，取最小；并列取后者；不看任何评估场景；选中的 EMA 复制为 `checkpoints/final.pt`。
- **停止规则（冻结）**：损失非有限 → STOP；过 2/3 预算后连续两次评估 score 无改善 → 提前停止（两组同规则）。
- **运行方式**：T0 在 gpu117（4 × L40S）；T1 若第二个 4 卡分配在 T0 结束前启动则并行，否则在 gpu117 顺序运行；两种情况下配置、数据、种子完全相同。

## 4. 关口
- **G0**：数据三态 / 填充 / 泄漏 / 对齐测试；模型形状、梯度（FM → μ；Recon、KL → μ, logvar）、T0/T1 公共初始化一致；tiny overfit（单卡 200 步：FM 与 Recon 都明显下降）；4 卡 DDP 冒烟（loss 与单卡口径一致、resume 逐位）；profile（sec/step、显存、GPU-h 估计）。
- **G1**（开发集固定状态，与 E2 相同的 VALSEL 90 + Dev-3 275 个状态，稀疏评分）：Future depth MAE（known / unknown / Coverage-new@10 / Physically-unseen@100）、HIT / Brier / ES、K = 4 离散度与离散度–误差相关；对照 B2、E2-D100。
- **G2**：同状态的 Early / Middle / Late regret、Top-1、Spearman、Top-2 分数差、逐场景配对差与 CI。
- **G3**：Task 2 / Task 3 闭环协议；AUC、Final、Cov@k、逐场景、配对 bootstrap、移动 / 重访、时延、显存。两组都跑；若 G2 未通过仍跑，标注 exploratory。**参照组**：B0、B2、E2-D100（Task 3 / E2 已有闭环）以及 **MAGICIAN Greedy / Beam（MAG-G / MAG-B，Task 2 共享动作协议下的闭环，不重跑）**，统一报告 AUC 与 Final / Cov@k；MAGICIAN 官方复现（`work_dirs/exp5_future_view/task0/magician_comparison/official_reproduction`）保留以备重跑。
- **G2 → G3 放行**（两组各自判定，冻结）：整体 regret ≤ 1.05 × B2 且 Top-1 ≥ B2 − 0.03（两个数据集都满足），或至少一个数据集 Early regret ≤ B2 且另一数据集未退化（regret ≤ 1.10 × B2）。未通过仍按用户要求跑 G3，标注 exploratory。
- **G3 判定**（T1 − T0 为主比较；T0 − B2、T1 − B2 同时报告）：沿用 Task 3 §4 的"有意义 / 退化"定义（ΔAUC ≥ +0.010、ΔFinal ≥ +0.010、胜 ≥ 4 场景、无场景 < −0.030；退化 = ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010）。

## 5. 公平性
- 同数据、同样本索引流、同噪声、同步数、同 effective batch、同优化器 / 调度 / EMA / 损失权重、同选择规则；T1 不读取 T0 权重。
- 正式训练开始后不改架构、种子、损失、域比例、掩码或评估协议；重大 bug 需同时停两组、记录并从同一状态重启。
- GPU：T0 用 gpu117 的 4 张 L40S，T1 用第二个 4 卡分配；总并发 ≤ 8 卡。
