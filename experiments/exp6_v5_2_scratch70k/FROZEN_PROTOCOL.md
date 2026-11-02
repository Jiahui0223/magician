# EXP6 V5.2-M Scratch70k —— 冻结协议 v1.0

**冻结时间：2026-11-02，在正式训练第 1 步之前。** 之后的任何改动都记入文末 §9 偏差日志，并写明时间和原因。
本文件不是结果。所有结果都由 `scripts/overnight_driver.py` 按本协议自动生成，写入 `RUN_STATUS.json`、`AUTO_DECISION.json` 和 `reports/`。
评估场景（Macarons++ Dev-3 / Test-3、MP3D VALSEL）在本项目中都**不是盲测**。

## 0. 不变量
- **完全从零训练**：所有参数随机初始化（seed 0）。不读取任何 V4 / V5 / V5.2 权重，只复用架构代码。由 G0 测试 `test_parameters_and_scratch` 检查。
- **Idea 2 数据流**：D100 输入 → 3D U-Net encoder → Ray-to-Memory Query → 2D Spatial U-Net（整幅 64×114 射线网格上下文）→ Depth-Aware 1D Decoder → 表面 belief b + 条件 hazard → **在计算透射率之前**施加 D100 FREE / BAND 约束 → p_first / p_nohit → S_visible + λ·S_potential → NBV。
- 不使用 Flow 或 K-world 采样，不生成 Depth 图，不调用 DistP1。规划器中的守卫会对这些调用抛错，计数必须为 0。
- **损失只有** `L = L_surface + β·L_firsthit`，不加 coverage / ranking / flow loss。

## 1. 模型（`configs/model_v52m.json`，实测参数见 `PARAMETER_AUDIT.md`）
- 总参数 **26,999,143**，在目标 25–27M 内；超过 35M 的缩放规则未触发：
  - 3D encoder：2,412,000
  - Ray query：201,060
  - Ray map：8,256
  - Spatial 2D U-Net [96,192,384,384]×2 ResBlocks：22,547,488
  - Depth decoder（宽 192，6 个 DWConv+FFN×4 block，k=5）：1,830,339
- 64 个深度区间：近处细、远处粗，16 vox 起；与旧 V5.2 相同。
- **FREE-before-hazard**：每个区间的 D100 FREE 占比 `free_frac` 按以下方式得到：
  - 取射线 `[t_near, min(d_lo, t_known, t_max))` 段与该区间的重叠比例；
  - 再乘以区间在 domain 内的比例（4 个子点处的 coarse dom 分数）。
  - 然后 `a = ρ·dt·(1 − free_frac)/S0`，之后才进行透射率累积。完全 FREE 的区间首中质量严格为 0。
- **BAND / 已知交叉（软约束）**：只在可信 TSDF 零交叉 t_known 所在区间加 `softplus(known_bias)`。known_bias 可学习、非负，初值 3.0。不强迫每个 BAND 体素都命中。
- **UNKNOWN**：自由预测。
- **截断**：t_max = min(出网格, tcap)；超过 t_max 的质量归入 no-hit。

## 2. 数据与标签
- **训练集**：Train-6 去掉 S-c，共 138 条轨迹（E2 D100 室外 shards，`work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100`）。MP3D 训练样本 0；Dev-3 / Test-3 / MP3D 均不参与训练或选择。
- **标签**（复用已验证的标签，不重新生成）：
  - 多表面标签：`work_dirs/exp6_v5/labels/v1`（Train-6 depth peeling，每条射线最多 8 个表面）；
  - 首中标签：shard 中的 `tstar`（T1.1 网格首中）。
  - 无效标签按旧 valid mask 屏蔽。
  - 首中落在完全 FREE 区间的射线（TSDF / 网格偏差造成）记为 FREE 冲突，从 NLL 中屏蔽并单独计数，不当作伪标签。
- **每个 view 的射线采样**：
  - 256 条表面标签射线；
  - 首中射线 256 条 = 128 条 UNKNOWN 类（按类别多项采样）+ 128 条均匀射线。
  - stale_p 0.5，stale_max 9。
  - 分层份额每 500 步记录：known-FREE、known-BAND、UNKNOWN+命中、UNKNOWN+真 no-hit、近 / 远命中、第二+表面。
- **校准与选择**：只用 **Train-6 S-c holdout**（42 条轨迹，256 views）。
  - 校准类指标（no-hit ECE、预测 / 实际 no-hit）只取**均匀射线**，即自然比例。
  - 其余指标用全部首中射线。

## 3. 优化（训练前冻结）
- **步数**：70,000 次同步 optimizer step。
- **全局 batch 固定**：64 views / 8 states，与 world size 无关：
  - 8 rank × 8 views；或
  - 4 rank × 16 views；或
  - OOM 降级后 micro-batch 减半、梯度累积加倍。
- **seen samples** = step × 64 views，单独报告，不只比较 step 数。
- **优化器与精度**：
  - AdamW，lr 2e-4，betas (0.9, 0.99)，weight decay 0.01；
  - warmup 1k，cosine 衰减到 0；
  - grad clip 1.0，EMA 0.999；
  - bf16 autocast，首中累积用 fp32。
  - 来源：repo 内的 scratch 训练经验，即 V4 / V3.1 的 40k scratch 配方（`experiments/exp6_v4/configs/outdoor_train.json`）。只是把总步数改为 70k。
- **β = 1.0**。规则：Train-6 上最后 20 步的两项 loss 梯度范数中位数之比在 [0.1, 10] 内时，β 保持 1。G0 实测 surf 2.17 vs firsthit 6.06，比值 2.79，因此 β = 1。
- **world size 改变时**（4 ↔ 8 rank 弹性扩容 / 降级）：模型、优化器、EMA、step 严格恢复；数据 RNG 流按 (seed, step, rank) 重新播种并写入日志。同一 world size 内恢复逐位一致（G0 已验证）。

## 4. 容错
- **checkpoint**：
  - 每 1000 步原子写 `checkpoints/last.pt`（模型、EMA、优化器、step、accum、best）、每 rank 的 RNG 文件和 `last_manifest.json`；
  - EMA 标签存于 2k / 10k / 20k / 40k / 55k / 70k，以及每 5k。
- **NaN / Inf**：写 dump 后退出码 87，状态为 blocked，不再启动评估。
- **CUDA OOM**：所有 rank 同步降级——每卡 micro-batch 减半、累积加倍（全局 batch 不变），射线 / 解码 chunk 减半。最多 2 次，仍失败则退出码 86，状态为 blocked。
- **作业时限**：每轮的 deadline = 所用 Slurm 作业最早结束时间 − 10 min。到点同步保存后退出码 85。未完成时状态写 checkpointed / partial，**不得写作已完成 70k**。

## 5. 选择、里程碑与门槛（只决定是否运行昂贵评估，不用于调参）
- **G1 holdout**：每 5k 步（EMA）在 Train-6 S-c 上计算：
  - 首中 NLL / 最高区间准确率，分自然 / UNKNOWN / known；
  - no-hit ECE；
  - `p_nohit > 0.95` 的真实命中率（high-confidence false no-hit）；
  - KNOWN 射线穿透率；
  - FREE 冲突率；
  - 首表面 / 第二+表面召回、深遮挡（≥128 VS）召回、分距离众数误差。
- **best_train_holdout**：取 `select = NLL_UNKNOWN + 2·HCFalseNoHit_UNKNOWN + 2·KnownPassThrough` 最小的 holdout 点。
  - KnownPassThrough = KNOWN 射线在 t_known + 8 VS 之后的质量 + no-hit 质量，即违反已观测表面的物理违规。D100 FREE 内的质量按构造为 0。
  - 一律预测 no-hit 会同时抬高 UNKNOWN NLL、误判率和穿透率，因此无法靠这种方式取巧。
- **保留的权重**：`last70k = step_070000.pt` 和 `best_train_holdout.pt` 都保留。**主权重** = best_train_holdout；若 best 在 70k，则与 last70k 相同。评估上的好坏不参与选权重。
- **G2 只读里程碑**（20k / 40k / 70k，以及最终的 best）：
  - 状态集：Outdoor Dev-3 275 态（B0 150 + B2 125）和 MP3D VALSEL 90 态。状态、候选、区域、R10 刷新、B2 地图与 V5 / V5.2 固定状态评估完全相同。
  - 每个 checkpoint 的归一化 floor 在 Train-6 S-c 上校准，沿用 V5.2 规则：60 态，floor = 0.05 × 每态最大值的中位数。
  - λ ∈ {0, 0.1}，η = 0.5 固定。
  - 保存每个候选的 S_vis / P_surf / P_space 和组合分解。
  - 不根据 G2 调 λ / η / 架构 / LR，也不挑权重。
- **G3 门槛**（在主权重上判定）：
  - (a) Outdoor Dev-3 G2 Top-1（λ=0，275 态）≥ 0.46；
  - (b) MP3D VALSEL G2 Top-1（λ=0，90 态）≥ 0.40；
  - (c) 两个域的 UNKNOWN 射线中，`p_nohit ∈ [0.9, 1]` 的射线实际命中比例（false no-hit）都低于旧 V5.2 的同一指标。若新模型在该区间没有射线，视为改善并如实报告。
  - **G3 smoke 总是运行**：4 个 Outdoor episode（alhambra s0/s1、bridge s0/s1，100 步）和 2 个 MP3D episode（8194nk5LbLH L0 S10、EU6Fwq7SyZv L0 S67，150 步），各跑 λ 0 / 0.1，与旧 V5.2 smoke 是同一组 episode。
  - **smoke 有效**的条件：12 个 episode 都完成（exit 0、步数完整），守卫计数为 0。
  - **G3 Full** 当且仅当 (a)(b)(c) 成立且 smoke 有效。Full 为 Outdoor Test-3 30 episode × 100 步和 MP3D 10 episode × 150 步，各跑 λ 0 / 0.1，分场景 / 楼层统计，给出 bootstrap CI。否则写明 skip。
- **规划器设置不变**：B2 规划器（D100 输入、S10 新颖性、K=4 候选、n_win=5、R10、P1 更新），只替换 camera utility。

## 6. 资源与无人值守链
QOS 限制每个用户 8 张 GPU，其中 4 张由交互作业 3443137（gpu115，到 11-04 15:11）占用。2026-11-02 时，`-N2` 的 8 卡 sbatch 必须等交互作业结束才能开始。用户已同意利用交互作业的 4 卡、再申请 4 卡。因此：
- **节点 B**：sbatch 作业 3449325（`scripts/slurm/nodeB.sbatch`，gpu112，4 × L40S，30 CPU，到 11-04 03:00），运行 `nodeB_loop.sh`。
- **头节点**：gpu115 上的 `overnight_driver.py`（setsid）。
- **训练轮次**：节点 B 在线时用跨分配双节点 torchrun（8 rank，ib0 地址）；否则用 4 rank。在 checkpoint 处弹性扩容 / 降级。
- **评估分工**：
  - 里程碑 G2 与训练并行，在头节点上跑；
  - 最终 G2 和 G3 用两个节点的 8 张卡，G3 每卡 2 个 worker；
  - 全部结束后写 `ctl/SHUTDOWN` 释放节点 B。
- **续跑**：`scripts/submit.sh`（sbatch 续跑脚本 `scripts/slurm/continue_1node.sbatch` / `continue_2node.sbatch`，从 last.pt 恢复后执行同一条链）。

## 7. 对比与报告
- **对比对象**：V5.2-M λ0 / λ0.1，对比旧 V5.2 pilot λ0 / λ0.1、V5-Warm12k、V4-Continue12k、V4@40k、B2（frozen V2-Mix D100）、E2-D100，以及 G3 中的 MAGICIAN。
- **比较口径**：固定状态和闭环都使用同一状态 / episode，配对比较。
- **G3 用户基准**（AUC / Final）：
  - V5：Outdoor 0.711 / 0.891，MP3D 0.333 / 0.501；
  - 旧 V5.2：Outdoor 0.608 / 0.781，MP3D 0.238 / 0.310；
  - B2：Outdoor 0.730 / 0.906；
  - E2：Indoor 0.364 / 0.553。
- **FINAL_REPORT 必答三个问题**：
  1. MP3D UNKNOWN 过度自信 no-hit 是否减少；
  2. Outdoor G2 Top-1 是否提升；
  3. G3 是否改善。
- **归因限制**：容量、scratch、2D 上下文、训练分布、几何约束同时改变，只能归因于联合干预。

## 8. 交付物
- **实验文件夹**：
  - ARCHITECTURE / PARAMETER_AUDIT / FROZEN_PROTOCOL / G0_VALIDATION / TRAINING_LOG_SUMMARY；
  - G1_FIRSTHIT / G2_FIXED_STATE / G3_SMOKE / G3_FULL / FINAL_REPORT；
  - AUTO_DECISION.json、RUN_STATUS.json、STATUS.md；
  - `scripts/`（含 `submit.sh`、`overnight_driver.py`、`slurm/*.sbatch`）。
- **`work_dirs/exp6_v5_2_scratch70k/`**：checkpoints、logs、eval、ctl。

## 9. 偏差日志
- 任务书指定的 `SLURM_BSU_8L40_TEMPLATE.sbatch` 在仓库和文件系统中都不存在，因此按以往 sbatch 自写（用户同意）。8 卡由两个分配组成，原因见 §6，而不是单个 `-N2` 作业。
