# UNIFIED_PROTOCOL — EXP Unified Outdoor Architecture Benchmark（冻结）

- **版本**：v1.0，2026-11-03 约 05:00 冻结，写于任何新训练开始之前。
- **来源**：用户夜间指令 Stage 2。
- **配方细节**：`TRAINING_PROTOCOL_DRAFT.md`。由子 agent 起草，我已审阅；本文件记录最终决定。

## 1. 训练密度：S10（按预注册规则）
- 规则写在 `experiments/exp_overnight_queue/OVERNIGHT_PLAN.md` §4，时间早于看到 Density A/B 第二批 G3 结果。判定过程见 `experiments/exp_density_ab/NEXT_TRAINING_RECOMMENDATION.md` §1。
- 两个条件都满足：
  1. 室外固定状态 G2：E2S 优于 E2D，4 种组合方向一致，多个 CI 不含 0；
  2. 室外 G3：E2S 不差于 E2D（3/4，CI 都含 0）。
- **证据强度：中等**，来自混合微调而不是从零训练。D100 分支（`configs/*_d100.json`）保留为待验证分支，本夜不跑。
- 所有架构使用同一批室外轨迹：Train-6 去 S-c，138 条；S-c 的 42 条只用于监控。数据是同一套 S10 分片（task1 T1.1）。V3-T1 的 4× TSDF 缓存由同一 S10 P1 构建（`build_tsdf_cache_s10.py`，构建时逐状态核对粗网格）。

## 2. 统一配方（4 × L40S DDP）
**所有臂都相同**：
- 60k 步（V4C / V5 为 48k + 12k）；
- 全局 64 视角 / 8 状态，每 rank 16 视角；
- seed 0；bf16；EMA 0.999；
- Flow-Matching 定义一致；采样分布一致：V3 用 v2_pool 采样器，dry check 中与 V2 / V4 的批次 1200/1200 相同；
- 每 2k 步保存一次 EMA（含 10k / 20k / 40k / 48k / 60k）；
- 完整状态 checkpoint 不超过每 2k 步一次，同时保存逐 rank 的 RNG 快照，用于精确续训。

| 臂 | 初始化 | 步数 | 峰值 LR / warmup / 调度 | 损失 |
|---|---|---|---|---|
| V2-S10 | 随机 seed 0 | 60k | 2e-4 / 1k / 余弦到 0 | FM |
| V4-S10 | 随机 seed 0 | 60k | 2e-4 / 1k / 余弦 | FM |
| V3-T1-S10 | 随机 seed 0 | 60k | **2e-4**（统一值）/ 1k / 余弦 | FM + 1.0·recon + β·KL（β 在 5k 步内从 0 升到 0.02） |
| V4C-12k-S10 | V4-S10 EMA@48k | +12k | 1e-5 / 500 / 余弦（沿用历史续训） | FM |
| V5-warm-12k-S10 | V4-S10 EMA@48k 的编码器 + U-Net；新解码器随机初始化 | +12k | 新模块 1e-4 / 200；继承模块先冻结 2k，之后 1e-5 / 500 | FM + 0.37·L_surf + 0.096→0（4k）·L_bridge |

**决定 D1（V3 的 LR）**：采用统一值 2e-4，保证公平。预先登记的回退规则：若训练器因损失非有限而 STOP，失败的运行改名保存，再用 1e-4 从零重跑，并记为偏差。

**决定 D3（V3 采样器）**：采用统一的 v2_pool。V3 正式训练前先做 4 卡 profile 冒烟；若 OOM，V3 标为 BLOCKED，不自动更换采样器，等用户决定。

**剩余的架构例外**（D2 / D4–D7，见草案 §3）：
- V3：accum 2、4× TSDF 输入、recon / KL；
- V5：L_surf / L_bridge 与参数组 LR；
- V4C / V5：沿用历史续训配方。

**V5 的结论是复合干预**（新解码器 + 多表面监督 + 蒸馏），提升不能单独归因于架构。

## 3. V2-S10 复用历史 V2 Original
`verify_v2_reuse.py` 逐项确认 task1 main 与统一 V2-S10 的配方一致：
- 配置 sha 相同，180 个分片的 sha 与当时记录一致；
- 从未续训；
- 训练代码在 V2 路径上逐位等价。

可复用的 checkpoint：EMA@10k / 20k / 40k / 60k（另有 45k / 50k）。没有 48k，但 V2 不需要 48k 父节点。60k EMA 的 sha 为 `a379c0fb…`。

## 4. 执行顺序（gpu115 的 4 张卡，交互作业到 11-04 15:11）
1. **V4-S10 60k**：约 7.7 h。在 step 1000 的第一个完整 checkpoint 后**主动中断一次再续训**，作为 4 卡精确续训测试，结果记录在 `runs/v4_s10/resume_log_r*.jsonl`。
2. **V4C-12k-S10**，然后 **V5-warm-12k-S10**：约 1.8 h + 2.6 h，父节点都是 V4-S10 EMA@48k，sha 首次启动时写入 `PARENT.json`。
3. **S10 TSDF 缓存**：4 个 GPU worker，约 1.2 h。
4. **V3-T1-S10**：profile 冒烟后正式训练，约 22 h；作业到期前跑不完，会保留可恢复的 checkpoint。
5. **评估**：V2-S10（复用）的固定状态 G1 / G2 与 V4-S10 训练并行（每张卡 1 个评估进程，显存足够）。其余模型在训练完成且有空闲时评估。

**不启动 Mixed 20k**。不给先完成的模型额外预算。

## 5. 评估设置（每个完成的模型）
- **固定状态 G1 / G2**（VALSEL 90 + Dev-3 275，`scripts/uo_fixed.py`，复用 Density A/B 的评估器）：
  - **D100 推理 + S10 评分**（主）；
  - D100 推理 + D100 评分；
  - S10 推理 + S10 评分：输入密度对照，因为 Density A/B 显示 S10 训练的模型在两种推理下各有长短；
  - S10 推理 + D100 评分：同一次运行中免费得到。
- **G3**：资源允许时按历史协议运行（MP3D 10 + Macarons++ 30 个 episode）。
- 历史 B2 / E2-D100 始终作为参照基线。
- **中间 checkpoint 不宣称最终性能。**

## 6. 偏差记录
| 日期 | 偏差 | 原因 | 影响 |
|---|---|---|---|
| 2026-11-03 05:13 | S10 TSDF 缓存提前在 CPU 上构建：24 个单线程 worker，`build_cache_cpu.sh`。原计划是 V5 之后用 4 个 GPU worker 构建 | 当时 GPU 被 Stage 1 占用，CPU 空闲（负载约 11 / 64） | 同一个构建脚本、同一套认领机制，每个状态都与分片粗网格逐位核对。训练链到缓存步骤时只补缺少的轨迹。CPU 与 GPU 之间的浮点差异只可能影响 sdf 通道的末位，占用类别由逐位核对保证一致 |

## 7. 运行记录
- **2026-11-03 08:41 续训测试**：V4-S10 在第 1000 步的第一个完整 checkpoint 后被主动中断（SIGTERM），08:43 续训。`runs/v4_s10/resume_log_r{0..3}.jsonl` 显示 4/4 个 rank 的 RNG（numpy 采样器、torch CPU / CUDA）**精确恢复（EXACT）**，日志为“resumed at step 1000”，之后 LR 与 loss 连续。损失约 30 分钟的计算。
- V4-S10 与 V2-S10 评估、CPU 缓存构建并行时为 0.55 s / 步（历史 0.43 s / 步）。
