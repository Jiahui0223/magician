# EXP6 — V3 从零训练设计：Simple 3D VAE 与 VAE + Camera-Ray Cross-Attention

> 日期：2026-10-29 · 状态：架构与执行规范（**未实际修改训练仓库、未启动 GPU 作业**）  
> 目标：直接交给代码 AI 审计、实现、自动验证并依次训练；无需逐步等待人工批准。  
> 现阶段**仅运行两个新增完整训练**：`V3-VAE`、`V3-VAE-XAttn`。Indoor-only / Outdoor-only 数据域消融延后。  
> 与历史文档 `EXP6_V3_EXPERIMENT_DESIGN.md` 不同：**本版明确废弃预训练 Sparse VAE、双分支 Adapter/Gate 和从 V2 checkpoint 微调的旧路线**。

## 0. 决策摘要（冻结的架构意图）

1. 数据集：只用 **MP3D + Macarons++** 的现有合法训练拆分，Mixed Training。训练/验证/测试按 **scene 或 house** 分离，绝不以同一场景的不同轨迹冒充独立场景。
2. 输入：**到时刻 t 为止真实深度融合的 Partial TSDF/P1**；保留 signed distance、BAND、EMPTY、UNKNOWN 及空间坐标。使用 100% 历史有效 depth 写 BAND，保留既有 EMPTY carving。Future GT 深度仅作监督，不更新 P1。
3. 编码器：仅一个从零训练的 **Simple 3D Convolutional VAE**，替换原 3D U-Net；输出单一**空间潜变量** `Z`。不能引入外部 VAE 权重、第二条 3D CNN 场景分支、特征融合 adapter 或 learned gate。
4. 生成器：保持 V2 的 Candidate Ray Geometry → Ray Encoder → 2D Rectified Flow U-Net（K=4、既有 10-step Euler 与 deterministic depth/HIT decode）。Scene Encoder、Ray 接口以及新 Attention 层需要重写并从零初始化。
5. NBV：完整输出经已有 **DistP1，使用历史 10% Sparse P1 做 Scoring**；Greedy Planner、候选位姿、Coverage、步数不变。**不得**把预测深度融合回 P1。
6. 新模型：`V3-VAE`（T0，不加 Cross-Attn）；`V3-VAE-XAttn`（T1，唯一增加 Camera-Ray 对 3D Latent 的 Cross-Attention）。两次**独立 scratch full training**；T1 不能由 T0 的已训练 checkpoint 继续训练后再与 T0 进行公平因果比较。
7. 评估：G0 实现与几何验证；G1 Prediction；G2 Oracle Ranking；G3 Closed-Loop。两次主训练代码均可在技术 Smoke 通过后自行顺序启动；G2 指标不好仍应留下完整结果，G3 若额外运行须如实保留协议。

## 1. 数据：什么是一条“训练轨迹”？

### 1.1 定义

**Scene** 是独立三维环境（MP3D house/floor、Macarons++ 建筑/场景）。**Trajectory** 是一个探索策略在某个 Scene 中产生的有顺序的真实观测序列，并不是一次 optimizer step，也不是一张训练图片。

给定场景 `s`、起点 `c0`、某个轨迹生成策略 `π`（例如 random / one-step GT-greedy），记录：

`τ=(c0,D0,c1,D1,...,cT,DT)`，其中 `ci` 是实际执行相机位姿，`Di` 是在该位置采集/渲染的**真实**深度。

在时间 `t`：

- 历史：`H_t = {(ci,Di)}_{i=0}^t`；`P1_t = Fuse(H_t)`，只含过去/当前真实观测。
- 候选集合：`A_t = {a_t1,...,a_tM}`，每个候选附 `(R,T,K)` 和合法动作检查。
- 一条训练样本 = `(scene_id, trajectory_id, t, candidate_id)`，**不是整条轨迹同时送进神经网络**。
- 输入 = 同一个历史 `P1_t` + 该候选相机 Ray Geometry + VAE scene latent。
- 监督 = 候选视角的 `GT first-hit depth t*` / HIT / 原有 V2 `b,y,cat,lmask` 标签，必要时从 GT mesh 或保存的候选视图计算，**只出现在训练损失/离线 Oracle 中，绝不进入 encoder 输入或在线规划**。
- 相同历史状态 `P1_t` 可对应多个不同候选相机样本，scene 编码应复用或缓存。

**示例**：一条 150-step 轨迹会产生约 150 个不同的历史状态；每个状态可能有约 7–8 个候选视角（实际数量必须读取存储记录，不能硬编码）。因此一条轨迹可产生约千个训练候选记录。**轨迹数量 ≠ 候选视角记录数量 ≠ 独立训练场景数量。**

### 1.2 已有 E2 审计中记录的真实训练规模（复用，不随意改造）

| 数据 | 训练轨迹 | 候选视角记录 | 说明 |
|---|---:|---:|---|
| MP3D | **228** | **263,358** | 114 个 train floors × random 与 GT greedy 两条（对应 61 个官方 train houses 的楼层筛选） |
| Macarons++ | **138** | **98,887** | 原 Train-6、去除用于验证的 S-c |
| 合计 | **366** | **362,245** | 并非 366 个独立场景，也非 36 万个独立场景 |

现有验证：MP3D VALSEL 6 个 val houses/9 floors；Macarons++ S-c 验证 loss，Dev-3 供 G1/G2（**不是全新 blind set**）。现有 MP3D 5 个闭环评估楼层和 Macarons++ 6 个闭环场景按原协议保留。必须审计每个 Scene 的 train/dev/test ID；S-c 为同训练场景内留出的起点，不能拿其视作 unseen-scene test。MAGICIAN 和其他外部论文协议如有不同，不宣称数值可直接公平对比。

### 1.3 时间点数据如何形成 V3 输入

```
trajectory τ (scene s)
  t=0: 真实 (C0,D0)                     → P1_0  → candidate A_0,1 → Flow label GT(A_0,1)
  t=1: 真实 (C0,D0),(C1,D1)             → P1_1  → candidate A_1,2 → Flow label GT(A_1,2)
  t=2: 真实 (C0,D0),(C1,D1),(C2,D2)     → P1_2  → candidate A_2,5 → Flow label GT(A_2,5)
  ...
```

- `P1_100(t)`：给 VAE/模型使用；历史有效深度 **100% BAND**。按当前 MP3D 区域局部重建/refresh 与 Outdoor 全局增量规则重建，严格复用现有 E2 构建器的语义。
- `P1_10(t)`：给 DistP1 用，复用原始采样种子/10% mask，**不让新的编码器重建结果代替评分地图**。
- `t*`、`b,y,cat`、`lmask`：对候选视角的 GT 监督；若依赖 `P1_100` 必须复用 E2 `encode_labels` 精确重算。`lmask = UNKNOWN ∧ b≠0`（以实际代码为准）。
- 防泄漏：`P1(t)` 不得使用 `D_{t+1}`、候选未来 GT、Scene mesh 或 Oracle 面积采样点；场景坐标与标定仍使用历史已知地图合法参数。

### 1.4 数据多样性解释

同一场景的很多轨迹能够带来不同可见比例、遮挡和 UNKNOWN Frontier，但不能变成很多独立的建筑。当前数据足够做 **限定于 MP3D + Macarons++ 的架构研究**；不能由此保证强的跨所有室内外世界泛化。保留已延后 `ablationstudy.md`，**本轮不跑 Indoor-only / Outdoor-only**。

## 2. VAE 输入：真实 Partial TSDF，不用旧的四个粗统计量

输入原始语义来自 P1：BAND 中有效截断 signed distance、EMPTY 的可靠自由空间、UNKNOWN 的未观测空间。必须将 UNKNOWN 的哨兵负值与真实负 signed distance 分开处理。

**建议单一 VAE 输入（最小且明确）**：

`X_t = [ s_t, m_band, m_free ]` （3 channels）

- `s_t = clip(TSDF/τ,-1,1)` **仅在具有有效 TSDF 的 BAND 中有物理值**；其余位置填 0，由 mask 区分。
- `m_band=1` 表示真实观测支持的表面截断带；`m_free=1` 表示真实 carving 支持 EMPTY；二者都为 0 时即 UNKNOWN（有效地图外由 spatial-valid mask 排除，不能当 UNKNOWN 监督）。
- 数据侧保留 `voxel_size`、`grid_origin`、`world_to_grid`、`spatial_valid` 以及时间 `t`，用于世界坐标对齐和 Padding Mask；外部几何条件（ray direction、pose、t_known 等）仍由原确定性模块输入，不需要增加额外的 learned head。
- 原始有符号 TSDF 的符号、截断距离 `τ`、Carving precedence 需用现有 P1 代码实测校验，不能猜测。`m_band / m_free` 互斥且不能把未知看成自由空间。

**可训练的大场景输入策略（不得把超过 1 亿体素直接送单张 dense 3D CNN）**：

1. 从真实高分辨率 P1 通过**状态感知的 2×/4×聚合**得到 `X_t`：BAND 是否出现、EMPTY 有效性与 BAND 有效 signed distance 必须独立聚合；不是旧的仅 4 通道 fraction summaries。优先尝试 2×；若实际 OOM，可固定采用 4×，两组模型完全一致，并记录细结构损失与物理 voxel-size。
2. 用 shared Conv3D 权重处理固定尺寸 **64³ patch**（初始候选，具体应根据 GPU 实测选定）；对部分地图做 deterministic tiling，记录 `origin / stride / overlap / valid mask`。必须保证跨 patch 的 Ray Feature 在世界坐标中连续，不允许 tile boundary 漏特征或错位。
3. 3D latent patch 输出拼接为可缓存的空间 latent `Z_t`；同一历史状态下不同候选相机共享。MP3D 按原 refresh cadence 重算；Outdoor 按现有增量地图更新规则维护依赖缓存；不允许因 VAE 每候选重新编码而把 planner 性能人为压低。
4. Patch 训练对 BAND/frontier/EMPTY/UNKNOWN 不同区域合理取样；重建归一化按 observed 类别统计，不允许大片 UNKNOWN 主导 loss。
5. 记录完整地图编码单次耗时、GPU peak、拼接边界差异、有效细表面召回。若 patch 方案无法支持历史地图全局一致且在资源预算内，STOP 并提供最小备选，而不是悄悄退化为 V2 统计特征。

**重要**：状态感知降采样会丢掉一些原始几何，是折中而非无损；需要报告其影响。两个数据集物理长度单位不同，绝不可在未校验的情况下将 Macarons++ scene units 直接视作米。

## 3. Simple 3D VAE 架构（两个模型完全共用）

```
Real Partial TSDF patch X=[sdf,band,free] + valid mask
           │
     Conv3D Stem / Res3D
           │
     Stride-2 Conv3D / Res3D
           │
    μ(X), logσ²(X)      spatial latent with 16 channels (draft)
           ├─────────────────────────────────┐
           │                                 │
     z_rec=μ+σ⊙ε                        z_cond=μ
           │                                 │
     Simple 3D Decoder                 Camera Ray gather(Z, C)
           │                                 │
  reconstruct observed geometry           Ray Encoder
           │                                 │
     L_rec + β L_KL             conditional 2D Rectified Flow U-Net
                                             │
                                             L_FM
```

- 初版为**浅层 Conv3D Encoder-Decoder + 单尺度 spatial latent**。例如 Stem 24/32 ch，1 或 2 个 stride2 3D blocks，latent 16ch；真正层数/shape 根据物理尺度与显存 Smoke 校准，并在 **两组训练前固定**。不加第二条场景 Encoder、语义预测器、可学习 fusion gate、小型附加 head；`μ/logvar` 与 Decoder 的最终输出属于 VAE 标准组成。
- Decoder 可以用一个输出卷积产生 `[sdf_hat, class_logits(UNKNOWN/EMPTY/BAND)]`（相同空间）；不是用于在线预测未知完整场景，只用于**重建当前历史观测状态**。
- **Train**：`z_rec=μ+exp(0.5logσ²)*ε` 进入 Decoder；`z_cond=μ` 进入 Camera-Ray/Flow。这样 KL 与重建训练真实的变分后验，而条件生成的随机性仍由 Flow 的 ε 控制。
- **Inference**：只加载 VAE Encoder 的 μ 路径；VAE Decoder 可以不实例化。`μ` 对同一 `P1_t` 和候选集合保持固定，便于共享缓存及候选公平比较。
- 标准重参数化使 `L_rec` 对 Encoder 的 μ、logvar 可导；`L_FM` 对 μ 可导，而**不直接对 logvar 求导**（这是预期的设计，不是 bug）。

### 3.1 核心联合损失

`L_total = L_FM + λ_rec · L_recon + β(step) · KL`

- `L_FM`：完全复用原 V2 对 UNKNOWN 且有效的候选射线的 rectified-flow velocity MSE，保留 `cat,b,y,lmask` 编解码，不引入 Ranking Loss。
- `L_recon`：当前 `X_t` 的**观测重建**。建议 `CE(class_hat, class)`（按 UNKNOWN/EMPTY/BAND 分组或重权归一，防 UNKNOWN 过大） + `masked_L1(sdf_hat,sdf)`（只在 BAND 与有效 TSDF 位置）；训练监督来源仅历史 P1，不需要完整场景 GT。
- `KL`：每个实际编码有效 latent 单元的 diagonal-Gaussian KL **按有效单元与通道数归一**，避免体素数不同改变权重。记录 rate、μ 标准差、logvar、active units 与 posterior collapse 指标。
- `β(step)`：先 warm-up，再达到固定非零 β；初始值在短程 Smoke 中依据各损失/梯度尺度确定，**在正式 T0/T1 两次训练前一次冻结**，禁止根据测试 AUC 回调 β。
- `λ_rec`：按统一 loss 规范固定，不给两个模型不同设置。预检必须显示 FM→Encoder、Recon→Encoder、KL→Encoder 梯度有效，并检查 Flow 不是只使用原有 Ray 几何忽略 Z。

### 3.2 正确的 VAE 概率语义

`q(z|historical P1)` 是**当前已观测几何的变分编码后验**，并不天然是未来隐藏场景的 epistemic uncertainty。未来候选视角的多模态不确定性仍由现有 stochastic Flow/K=4 建模。不能因为 VAE 有 `σ` 就将其直接解释为未知表面信息增益。

## 4. 两个主模型、初始化与 X-Attention

### 4.1 模型 T0：`V3-VAE`（从零训练）

`P1_100 → Simple 3D VAE Encoder μ → Candidate Ray Samples + deterministic 11ch Ray Geometry → Ray Encoder → original-architecture 2D Flow U-Net → Future Depth → DistP1(P1_10)`。

- 不使用旧 `F0/F1` / 3D U-Net；Ray Encoder 的唯一新增 3D scene 特征来源是 VAE latent `Z`。
- 原有 Ray Traversal 和原本 Flow 输出物理语义保留；因为接口改变，**3D VAE、Ray Encoder、Flow U-Net 全部随机初始化并联合训练**。VAE Decoder 训练时随机初始化。

### 4.2 模型 T1：`V3-VAE-XAttn`（**再次独立从零训练**）

完全复用 T0 结构与超参数，只在 Flow U-Net 一处低分辨率 bottleneck 增加**标准 Cross-Attention**：

- `Q`：候选相机的低分辨率 ray/view tokens，包含 2D Flow hidden features 与已知几何射线条件；严格体现该候选相机（不能每个 candidate 共用同一个 Q）。
- `K/V`：同一历史 `Z=μ` 的空间 tokens，加 world-coordinate positional encoding 与 observation/valid mask，只有真实历史观测支持的 token 才能作为已观测证据；不把 UNKNOWN token 误称为已知表面。
- 允许有限预算的局部/frustum tokens（例如初始不超过 256/512，须由 GPU/时延 Smoke 检查），禁止对完整大地图做无约束全局注意力。
- `H' = H + CrossAttention(LN(H), M_scene)`，标准 Q/K/V 输出投影即为 Attention 必需部分，**不另加 fusion head、learned adapter 或 gate**；与 T0 共用其余结构和损失。
- 同一 scene memory 在 K=4 Flow samples、Euler steps 和多个 candidates 之间缓存 K/V；Query 依候选相机/当前生成状态变化。

### 4.3 什么才叫“从头训练”？

- **T0**：VAE/Decoder、Ray Encoder、Flow 均随机初始化，不加载 V2 Original / V2-Mix / E2 或外部 VAE 权重。
- **T1**：在同一随机种子下，其**共有模块使用与 T0 一致的初始参数**（可以通过相同 seed + 显式校验实现），新 Cross-Attn 独立随机初始化；**不加载已训练 T0 checkpoint**。
- 两组所有可训练参数从第 0 步联合更新，VAE Decoder 是训练期辅助模块；独立 optimizer/EMA/scheduler。
- 若先训练 T0，再基于 T0 checkpoint 添加 XAttn 并微调，只能叫 `T1-warmstart`，它是另一种有价值的优化策略，但**不能混淆为本次 T1 scratch 主对照**。
- 旧 V2 系统可保留作为**历史性能基线**，但因为它经历不同的预训练/微调预算，与新的 Scratch V3 不能宣称完全隔离架构的公平因果比较；T0 vs T1 的 scratch 同预算对照才最干净。

## 5. 全量训练预算与自动执行

**不要把 E2 的 5000 步当作 scratch 训练默认值**。E2 的 5000 步从已训练的 V2-Mix EMA checkpoint 开始微调；Task 2 的 V2-Mix 也曾从 V2 Original@60k 开始做 20k 训练。Scratch VAE + Flow 可能需要更长的学习过程。

建议作为第一次运行配置（**需要先按实际仓库/吞吐量审计、输出配置快照**）：

- 优化器 `AdamW`，BF16、grad clip 1、EMA 0.999；从 scratch 的初始 LR 不直接继承 fine-tune `2e-5`，可尝试峰值 `1e-4` 作为可执行初稿，预热 1k steps + cosine；数值是否稳定必须以 Smoke 确认。因 VAE 的 KL 训练敏感，可以设置同一 optimizer 不同固定参数组 LR，但两模型保持一致，禁止动态因测试表现修改。
- Batch target：每 global step 32 MP3D + 32 Macarons++ candidate views；如果新增 VAE 导致 GPU OOM，可 gradient accumulation 达到同样 global effective batch，不随意减少训练监督。
- Loss：保留 E2 原来的 **按全局有效监督射线求平均**，同时输出实际 indoor/outdoor 的有效射线比例、domain losses、梯度；不在当前架构研究里顺便改变 Domain Loss Weight，否则引入新的混淆变量。
- Full-run 初始预算：**每组最多先规划约 60k optimizer steps**（不是 5k 微调；属于待实测确认的执行上限，不保证这就足够收敛），定期在固定开发集评估 5k/10k/20k/40k/60k；用**预先指定的 Dev 选择规则**选最终 checkpoint，而非使用测试 Scene 选。若连续无改进或明显数值失败，按 frozen 停止规则处理并保留全部日志；确需超预算或改变重要超参数时先报告资源缺口。
- 必须先 profile 100–300 steps 的 `sec/step`、GPU memory、scene encoding cache memory、candidate inference latency，再计算全量 GPU-hours。旧 20k/4 L40S 约 10.5 h（已有 Task 2 实测），不能把它直接当成新的 VAE 模型时长。
- 必须支持 checkpoint resume、Slurm preemption 后重入、seed 固定、配置 hash、训练数据 snapshot hash、step/logging 原子写入。

**无需人工逐步审查的执行顺序**：

1. Audit existing `seen2scene/futureview/`、`t22_train_mix.py`、E2 P1 builder/labels、scene split、current Slurm GPU；产出审计日志。
2. 实现 raw-partial-TSDF VAE 数据路径、tile/physical coords、最小 VAE/Decoder、shared candidate flow training、T0 和 T1 Cross-Attn 模块；完成单元测试。
3. G0：逐位/公差校验 P1 history、无 GT 泄漏、拼块世界坐标、Decoder mask、可训练梯度；单卡 Tiny Overfit 与 DDP Smoke；预估完整训练成本。
4. 锁定 `protocol.md` 与两组配置、训练集索引、开发集 checkpoint 选择规则、硬性资源上限。无需用户批准可继续；**出现硬故障（OOM 无法解决、无合法 split、坐标/标签错误、GPU 不可用）则 STOP 报告，不带病训练。**
5. 完整训练 `V3-VAE` (T0)；保存 EMA checkpoints + 定期 Dev G1/G2。完成后自动进行 T0 固定状态预测/排序评估。
6. **不从 T0 权重继续**，按同样预算、相同样本索引和共同模块初始化完整训练 `V3-VAE-XAttn` (T1)；自动进行 T1 G1/G2。
7. 两组都完成后，按冻结协议对二者进行 G3 同样规模闭环，复用 B0/B2/E2-D100 已有结果；如果 G2 失败而仍执行 G3，明确标注 exploratory，不事后更改 gate。
8. 自动生成一份**同时包含 T0 与 T1** 的最终统一报告；保存单独可追溯训练曲线与闭环数据。

## 6. 评价、报告与停止规则

### 6.1 必须报告

- **G0 Geometry**：观测 BAND 重建 MAE/F1、EMPTY 分类、UNKNOWN→假表面错误、thin-structure、patch seam、KL/active units、GPU memory、FPS。Indoor/outdoor 独立报告。
- **G1 Future Prediction**：同一 100% P1 条件下 known/unknown、`Coverage-new@10` 与 `Physically-unseen@100` 两种新增定义分别报告；MAE、HIT/Brier/ES、K=4 Diversity、diversity-error calibration。
- **G2 Candidate Ranking**：Early/Middle/Late Oracle Regret、Top-1、Spearman、Top-2 score margin、按 Scene 的配对差、Scene bootstrap CI。
- **G3 Closed-Loop**：MP3D 与 Macarons++ 分别报告 AUC、Final、Cov@20、曲线、Episode/Scene wins、movement/revisit、runtime/peak memory。Macarons++ 新增三个室内场景与原始室外场景类型不应混为一类。
- **训练动力学**：FM、Recon、KL 三项随步数变化，实际 KL 权重，梯度范数、后验均值/方差、有效监督 ray 比例与每域 loss；T0/T1 完全独立从零初始化证据。

### 6.2 核心比较（避免 Ablation 过多）

| 比较 | 核心研究问题 | 是否完整训练 |
|---|---|---|
| T0 `V3-VAE` vs B2/E2-D100 | 新 Geometry Latent VAE 方案能否胜过既有 V2 系列 | 新增 T0 1 组 |
| T1 `V3-VAE-XAttn` vs T0 | 在同样从零训练及数据下，Cross-Attn 是否带来增益 | 新增 T1 1 组 |
| T1 vs B2/E2-D100 | 统一 V3 是否改善室内外 NBV | 不增加新训练 |

不增加 `AE-only`、`XAttn-only`、`Indoor-only`、`Outdoor-only` 等全量训练到当前主线；可以列入 Future Work。若需要正式声称“KL 是改善的原因”，未来至少需一个 β=0 的受控实验；本轮**不能只凭两个主模型证明 KL 的独立贡献**。

### 6.3 硬停与结果性失败区分

- **硬停（不能带病运行）**：历史 GT 泄漏、错误时间索引、世界坐标错位、TSDF UNKNOWN 当 EMPTY、NaN、空 valid-ray / empty masks、无法恢复的 OOM、无法取得授权 GPU、训练/测试 scene 重叠。
- **结果性失败（记录，不自动篡改超参）**：Dev Prediction/Regret 较差、Outdoor 退化、KL 偏低、性能不如 V2。按预设 early-stop/预算留下结果；用户既已要求串行运行两个主模型，不应因为 T0 指标没有改善就默默取消 T1，除非技术上不可用。
- 完成 G3 后**不自动新增架构改造、改变 scorer、增加第三组完整训练**；先给出证据，再讨论下一轮。

## 7. 交付物（建议路径）

在实际项目根目录下创建 `experiments/exp6_v3/`（先检查与已有目录冲突）：

```
experiments/exp6_v3/
  protocol.md                   # 完整冻结数据/训练/评估规则
  configs/v3_vae.yaml
  configs/v3_vae_xattn.yaml
  scripts/                      # dataset / TSDF tile / model / train / eval / run scripts
  tests/                        # 时间泄漏、mask、coords、KL、反向传播、DDP 测试
  outputs/v3_vae/               # 每步日志、EMA、G1/G2/G3
  outputs/v3_vae_xattn/
  reports/V3_COMBINED_REPORT.md # **一份同时包含 VAE 和 VAE+XAttn 的报告**
  reports/V3_T0_VAE.md
  reports/V3_T1_XATTN.md
  reports/V3_DEVIATIONS.md
```

`V3_COMBINED_REPORT.md` 应采用另外提供的 `EXP6_V3_REPORT_TEMPLATE.md` 结构，包括：Research Questions、Dataset Scene Split、Trajectory 数量与样本构建、架构/损失、Scratch 初始化证据、完整 G0/G1/G2/G3 对照、按 Scene 分解、时延显存、负面结果、局限、最终 Go/No-Go。任何未运行项目标 `not run`，禁止造数。

## 8. 当前可确定与仍须代码审计的事项

**已经确定**：两个模型都从零训练；使用真实历史 100% P1 作为模型条件、10% P1 评分；同两个数据集、同样本/协议；VAE 三通道语义输入、单 Latent、联合 FM+Recon+KL；T1 增加单层 Cross-Attention；两组训练自动衔接；单一报告统一分析。

**要由代码 AI 自行核对并记录，不需要用户先确认**：当前 P1 TSDF 的真实 signed-distance/权重/状态枚举，Macarons++ 的 physical world units，最小显存允许的 patch 降采样倍率、tile 坐标/缓存规则，Flow 条件 tensor 的真实 shape，是否可以使用现有 E2 的无泄漏 builder，以及 Slurm 实际可用 GPU / 可用 walltime。

**此方案没有声称已经修改实际代码或启动训练。** 若仅拿到这份文档而没有项目源代码与可执行计算环境，应把它作为 coding-agent handoff 而不是训练已完成的记录。
