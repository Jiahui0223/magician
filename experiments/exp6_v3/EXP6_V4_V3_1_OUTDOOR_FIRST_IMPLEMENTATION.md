# Seen2Scene · EXP6 V4 + V3.1：Outdoor-first 架构与跨域训练执行规范

**状态：用户批准的下一阶段实施任务书（设计与执行要求；不是已运行结果）**  
**范围：V4 3D U-Net + Ray-Aligned X-Attn + 2D Rectified Flow；V3.1 Simple 3D VAE + 原 Ray Encoder + 2D Rectified Flow。**  
**执行原则：先仅用 Macarons++ Outdoor 训练两个模型；冻结权重后，对 Outdoor 和 MP3D Indoor 同时做 G1/G2/G3；Indoor Fine-tuning 设计与准备好但在 Outdoor 阶段完成前不启动。**

> **必须正确理解用户意图**：用户前文反复要求“先 Outdoor 训练，然后 Indoor Fine-tune”；偶有“先都在 Indoor”口误。本任务以 Outdoor-first 为准。若历史训练配置与口述不一致，请记录并以版本化事实回答，而不是猜。用户要的是**候选相机条件下的未来 Depth 生成**，不是现阶段新增网络直接回归下一 Camera Pose。Planner 暂时冻结，不引入 Beam Search、Frontier Bonus 或其他新评分器。

## 0. 一分钟任务摘要

1. **审计**仓库 `plan.md`、V2 Original、V2-Mix/B2、E2-D100、EXP6 V3 Task0/Task1 训练配置、checkpoint 继承和完整评估协议；核对 `Macarons++ Train-6` 及 MP3D 训练/评估划分。
2. 建立 `experiments/exp6_v4/`：V2 的 **3D U-Net + 原 Camera-Ray 采样特征**为主干，在 Ray-to-3D 几何条件模块中添加**轻量 Ray-Aligned Cross-Attention**；**不修改 2D Flow、输出、Loss、Sampler 或 Planner**。从随机初始化 Outdoor-only 训练。
3. 建立 **独立** `experiments/exp6_v3.1/`：复用 V3-T0 的 **Simple 3D VAE + 原 Camera-Ray 条件 + 2D Flow**，从随机初始化 Outdoor-only 重新训练；保留原 V3 Task0 作为 Mixed-from-scratch 历史参照，不覆盖其任何文件。
4. 两组**并行 4 GPU + 4 GPU**（若合法可用，未确认资源则按实际 Slurm 分配），使用同一 Outdoor 训练集及预先冻结的有效样本协议，40k 作为**建议第一阶段固定预算**；不可从评估 Test 场景选择模型。正式预算须依据审计和吞吐测试冻结，并对相同预算的两组一致。
5. Outdoor 权重完成后，在**不训练 MP3D**的情况下分别运行 Macarons++ 的 30 条、MP3D 的 10 条 G3，输出 **AUC / Final Coverage / Cov@20**，与 **MAG-G / MAG-B / V2 Original / B0 / B2 / E2-D100 / V3-T0 / V3-T1** 在统一表中对照。每个模型也做 G1/G2 和逐场景统计，重点观察 G3。
6. 分别记录 Outdoor-only 最终 checkpoint、基于训练前冻结选择规则的 EMA；准备 MP3D Indoor Fine-tuning 配置，但**第一阶段结束先提交结果和事实审计，不自动发起 MP3D 微调训练**。用户确认后才启动第二阶段。
7. 将 `n_win=5` 留作主实验公平配置；`n_win=7/9` 是**冻结模型、Planner-only 独立消融**，须先检查评分窗口与模型输入/输出范围的具体关联，不可混入架构训练。2D Flow 架构与新 Loss 的改造暂缓。

## 1. 科学问题与对照边界

### 1.1 本次明确回答的四个问题

- **RQ1（V4）**：在 V2 3D U-Net + Rectified Flow 的框架中，加入 ray-aligned Cross-Attention 是否改善 Future Depth 质量、Candidate Ranking、最终 Closed-Loop Coverage？
- **RQ2（V3.1）**：V3 Simple 3D VAE 采用 Outdoor-only 从头训练后，相比既有 Mixed-from-scratch 版本是否出现不同的 Outdoor 能力与 MP3D Zero-Shot Transfer？（**是训练策略整体比较，不是只改变一个变量的严格训练顺序因果检验**。）
- **RQ3（Transfer）**：两个仅用 Outdoor 数据训练的模型是否可以跨域泛化到 MP3D？Indoor Fine-tuning 后又改善多少，Outdoor 是否遗忘？
- **RQ4（科学叙事）**：基于候选 Camera Pose 直接生成其未来观测的 NBV 框架，能否在性能、推理开销、相机几何条件利用效率上展现差异，而非仅靠增加 Attention 参数量？

### 1.2 绝不可混淆的三种对照

1. **性能/部署对照**：V4 与历史 V2/B2/E2/MAGICIAN 比较；要完整列出但不可把所有差异归因于 X-Attn。
2. **X-Attn 因果对照**：相同 U-Net、数据、训练预算、初始化和 Flow，**只有 X-Attn 有/无**。历史 V2 Original 若训练来源及预算不匹配，只是历史参照；严格对照需另训 **U0 (matched U-Net-only)**。本轮先不强制启动第三个大模型；先查 V2 Original 是否适合，若不适合则在结论标注“X-Attn 因果结论尚待 U0”。
3. **Mixed vs Outdoor-only 对照**：V3-T0（Mixed）和 V3.1（Outdoor-only）在总样本构成、有效射线、训练数据曝光与数据域上不同；不能仅根据两个结果断言“训练顺序”本身导致优劣。训练顺序的严格消融需在同一架构、同一总数据域曝光量与同一预算下进行，未来独立实验。

### 1.3 不在本次实施范围

- 不新增神经网络 Pose Predictor；当前 CandidateGenerator 输出合法候选 Pose，模型预测每个候选 View 的 Future Depth。
- 不做 Full 3D World Generation、Beam Search、新的 Gain Head。
- 不继续调 S1/S2/S3、DepthU、Frontier Bonus；前述实验表明它们未可靠改善 G3。
- 不改变原始 DistP1、Candidate Generator、动作约束、P1 输入/评分密度、G3 Coverage 定义。
- 不同时重写 2D Rectified Flow（速度网络、Sampler、Objective）。Flow 作为第三模块的独立后续研究。

## 2. 依赖审计（任何正式训练前必须完成）

请逐项读取真实源码、配置及报告，记录 **路径 / commit SHA / checkpoint SHA256 / 实际字段 / 结论**。不能凭此任务书中的推测替换仓库事实。

1. 根 `plan.md`、实验 `README`、V2 与 V3 当前模块源码和冻结协议。
2. **V2 Original EMA@60k**（历史路径例：`work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt`），确定它的 60k 步是否 **真的是 Outdoor-only**、数据来源、全局 batch、loss 和输出协议。
3. **V2-Mix EMA@17.5k**（历史路径例：`work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt`）；已知它从 V2 Original@60k 继续 20k 的 50/50 Mixed 微调中选取第 17.5k。**B2 和 B0 使用同一 V2-Mix 权重，但 P1 输入/评分设置不同**；务必由代码复核。
4. **E2-D100** 的权重与输入：从 V2-Mix 出发、额外混合微调的方案；复查实际 checkpoint，不将其错认作 Outdoor-only。
5. **V3 T0/T1**：40k from scratch Mixed；实际 VAE Latent 空间大小、F0/F1、KL 与 Recon、Cross-Attention 的接法、Flow Head 形状、训练采样率和 EMA 选择；V3.1 从 T0 复制配置结构，但必须重新初始化并改为仅 Outdoor 数据。
6. `Macarons++ Train-6` scene ID 与 Test-3/Dev-3：确保训练数据无场景泄漏；`MP3D` 用于后续 Fine-tune 的 Train 房屋与 Zero-Shot 的 Eval 房屋严格隔离。
7. V2 原 Camera-Ray Encoder、3D U-Net F0/F1 具体 shape 和 sampler：历史参考是 F0 `32 × X/8 × Y/8 × Z/8`、F1 `64 × X/16 × Y/16 × Z/16`；每条 ray 约 32 个三维采样点；2D View Grid 约 `64 × 114`。**以上必须由代码确认，不能硬编码为普遍事实。**
8. DistP1：历史有效 BAND、新颖表面筛选、距离权重、K=4 sample scoring、输入 P1 与评分 P1 的关系，所有设置保留。MP3D 主对照 `n_win=5`；室外按已有冻结参数。
9. 训练数据：每条视角的 `Historical P1`、Camera Pose / K、未来 Depth/HIT、有效 UNKNOWN Ray Mask，摄像机坐标系与深度单位；确认没有使用未来视角作为 P1 的信息泄漏。
10. 评估：MP3D 5 楼层 × 2 starts × 150 steps = 10 EP；Macarons++ 6 scenes × 5 starts × 100 steps = 30 EP；3 Test + 3 Dev；场景级 Bootstrap 及 Base Seed；哪些场景已用于多次开发（Test-3 不再可视为全新盲测）。

**必交**：`AUDIT_BASELINES.md`（真实性与未知处）、`FROZEN_PROTOCOL.md`、`CHECKPOINT_LINEAGE.md`。审计不阻塞合理代码准备，但审计未完成不得提交正式训练 Gate。

## 3. 源码结构与接口设计

遵循现有包结构，把通用可复用模块加到 `seen2scene/futureview/` 对应合理目录，**以审计确认的真实包路径为准**。不要向旧模块引入破坏默认运行的行为；新版本走明确的 config flag / model registry。

建议实验目录：

```text
experiments/
├── exp6_v4/
│   ├── README.md
│   ├── AUDIT_BASELINES.md
│   ├── CHECKPOINT_LINEAGE.md
│   ├── FROZEN_PROTOCOL.md
│   ├── STATUS.md
│   ├── configs/
│   │   ├── outdoor_train.yaml
│   │   ├── outdoor_eval.yaml
│   │   ├── mp3d_zeroshot_eval.yaml
│   │   ├── indoor_finetune_DRAFT.yaml
│   │   └── nwin_ablation_DRAFT.yaml
│   ├── scripts/
│   ├── tests/
│   ├── report/
│   │   ├── G0_VALIDATION.md
│   │   ├── OUTDOOR_TRAINING.md
│   │   ├── G1_G2_EVAL.md
│   │   ├── G3_OUTDOOR.md
│   │   ├── G3_INDOOR_ZEROSHOT.md
│   │   └── FINAL_REPORT.md
│   └── reports/
│       ├── V4_COMPARISON_TABLES.md
│       ├── V4_DEVIATIONS.md
│       └── V4_RESOURCES.md
│
└── exp6_v3.1/
    ├── README.md
    ├── FROZEN_PROTOCOL.md
    ├── STATUS.md
    ├── configs/
    │   ├── outdoor_train.yaml
    │   ├── outdoor_eval.yaml
    │   ├── mp3d_zeroshot_eval.yaml
    │   └── indoor_finetune_DRAFT.yaml
    ├── scripts/
    ├── tests/
    ├── report/
    │   ├── G0_VALIDATION.md
    │   ├── OUTDOOR_TRAINING.md
    │   ├── G1_G2_EVAL.md
    │   ├── G3_OUTDOOR.md
    │   ├── G3_INDOOR_ZEROSHOT.md
    │   └── FINAL_REPORT.md
    └── reports/
        ├── V3_1_COMPARISON_TABLES.md
        ├── V3_1_DEVIATIONS.md
        └── V3_1_RESOURCES.md
```

由于 `exp6_v3.1` 含点号，**不要**把它直接用作 Python module import name；可将可导入的模块放在真实 `seen2scene/...` 包内，或使用 `exp6_v3_1` 作为 Python 模块名。不要更改用户明确指定的磁盘实验目录名。

所有大 Checkpoint、Render、数据和日志放 `work_dirs/` 或集群既有 scratch，不提交到 Git。

## 4. V4 模型：3D U-Net + Ray-Aligned X-Attn + 2D Rectified Flow

### 4.1 结构原则

**基准 V2 路径必须保持可运行：**

```text
Historical P1 (per-frame partial TSDF)
 → 3D U-Net → multi-scale F0 / F1
 → Original Camera-Ray Encoding / Trilinear Sampling
 → 2D camera-conditioned feature map
 → Original 2D Rectified Flow → Future Depth / HIT
 → Original DistP1, K=4 → selected legal camera
```

**新 V4 路径仅增加一处 Ray-Aligned Cross-Attention：**

```text
Historical P1 → SAME 3D U-Net → F0 / F1
                              ↓
Candidate Camera Pose & Intrinsics → Ray Geometry
     ├─ original ray sampling/base feature ───────────────┐
     └─ per-ray sampled 3D tokens (key/value), ray query ──┼→ Ray-Aligned Cross-Attn → residual fusion
                                                            ↓
                                       SAME 2D Flow + SAME Future Depth/HIT
                                                            ↓
                                           SAME DistP1/K=4 Planner
```

第一轮**保留 V2 原 Ray Sampling**，X-Attn 是附加的信息聚合，不是删掉原模块重写所有特征。这样更容易消融其增量价值，并降低早期训练崩溃风险。

### 4.2 建议的张量接口（以代码审计后匹配的实际 shape 为准）

- 输入 `scene_features = (F0, F1)`，每层为 `[B, C_l, X_l, Y_l, Z_l]`。
- 候选相机的 Ray origin、方向、near/far、相机内参，构成 `[B, R, ...]`，`R=H_view×W_view`。
- 对每条 ray 沿原 V2 约 `S=32` 个采样深度点，使用 `grid_sample`/trilinear 取多尺度特征：`sample_tokens [B, R, S, C_token]`。
- `ray_query [B,R,D]` 包含已有原 Ray Encoder 特征及相机射线编码，不把 GT Depth 作为 Query。
- `valid_mask [B,R,S]` 表示采样点在合法 3D 空间/窗口内；UNKNOWN 是合法未知，并非 invalid。所有被 mask 的 ray 需要安全回退。
- Output `attn_context [B,R,D]`，恢复 `[B,D,H_view,W_view]`，与原二维条件特征 residual 融合，保持原 Flow 输入通道数和接口兼容。

建议核心计算：

```python
# ray_feat:    [B, R, D]   V2 original camera ray feature
# samples:     [B, R, S, C] multi-scale 3D samples along each ray
# geom:        [B, R, S, G] relative depth, ray projection, scale info
# valid_mask:  [B, R, S]
q = to_q(ray_feat)                                    # [B,R,D]
k = to_k(concat(samples, geom))                      # [B,R,S,D]
v = to_v(concat(samples, geom))                      # [B,R,S,D]
logits = dot(q[..., None, :], k) / sqrt(D)            # [B,R,S]
logits = logits + geometric_bias(geom)               # optional: fixed shape, no GT
weights = masked_softmax(logits, valid_mask, dim=-1) # all-masked safe
context = sum(weights[..., None] * v, dim=-2)        # [B,R,D]
condition = ray_feat + projection(context)           # [B,R,D]
```

真实代码可以采用多头 Attention 与分块/Chunked Ray 处理；**不允许**构造不必要的 `[R,R]` 全 Ray Attention 矩阵。`geometric_bias` 初版可只使用沿射线采样点的相对 depth/尺度嵌入，不必在本轮引入第二个全局 Attention 分支。

建议将输出投影以**受控 residual / 小尺度初始化**，并证明原支路仍然工作；在随机初始化训练时检查 X-Attn 梯度实测为非零、不会持续无效。

### 4.3 Cross-Attn 需要回答的消融问题

- Ray-Attention 是否会对不同相机位置/方向产生合理的不同注意力分布？
- 它是否真实改善了未观测表面的 Future Depth/HIT，而不只是提升已知区域重建？
- 有没有增加过多延迟和显存？各设置按相同候选数、K=4、Flow Euler steps 比较。
- 与历史 V2 的比较可做性能对照；**如要宣称 X-Attn 造成改善，必须满足 U0 matched baseline（见 1.2）**。
- 如第一版普通 ray-aligned X-Attn 有价值，未来可以进一步考虑几何相对位置 Bias 的方法创新，但当前**不要**同时堆叠多种 Attention 形式。

### 4.4 第一阶段禁止改变

- V2 3D U-Net 主体、Partial TSDF 通道语义和 Input P1。
- 现有 2D Rectified Flow 的 Network、Loss、时间噪声采样、Target Depth/HIT 编码与 Sampler。
- 正式推理 `K=4`、Euler 迭代次数（历史为 10，代码确认）、评分 DistP1。
- MP3D 主报告 `n_win=5`；不混合之前 S2/S3、Frontier 的探索性改动。

## 5. V3.1 模型：V3-T0 Simple VAE 的 Outdoor-only 独立重训

1. 从 `exp6_v3/task0` 的**源代码定义与有效参数配置**创建 V3.1 训练入口，**不是**加载 V3-T0 已在 Indoor+Outdoor 上训练的 checkpoint。
2. 3D VAE 的 `encoder → μ/logvar → reparameterized z → decoder` 路径、F0/F1 对接、Flow Conditioning 路径与 T0 一致；KL、Reconstruction、FM Loss 及权重保持一致，除非 G0 证明原设置在 Outdoor-only 数值上存在严重问题。
3. 保持 V3-T0 对 P1 的表示、深度有效性、HIT/NO-HIT 和评估协议，不加 X-Attn。若过去 T0 使用 VAE `mu` 作为 Flow 条件，V3.1 亦遵循；具体以审计后的真实源码为准。
4. 随机初始化全部可训练参数；分开记录模型参数数目、GPU 显存、Forward/Backward 延迟、Flow/Recon/KL 的**分别按域/射线有效数归一化**的损失。
5. 检查 Posterior Collapse（如 KL 过低、latent 对结果不敏感）、Recon 失衡、`mu/logvar` 数值与复现噪声；不要没有证据就更改 KL 权重。
6. 因为本轮变动为数据域，不要拿 V3.1 对 V3-T0 的结果宣称“VAE 架构优化”；应描述为**相同 VAE 架构的 Outdoor-only 训练对照**。

## 6. 第一阶段训练协议：Macarons++ Outdoor-only

### 6.1 数据、泄漏与采样

- 只使用原先核查过的 **Macarons++ Train-6**，**零 MP3D 训练样本**；相同的原 P1 构建、历史状态采样、相机姿态来源、GT 未来 Depth/HIT 标签、UNKNOWN/FREE/BAND 语义。
- V4、V3.1 尽量共享**数据 Manifest、Ray 有效性判定和确定性采样序列**，但保持各自单独 Seed、DDP RNG 与 DataLoader 状态日志。
- 严禁将 Dev-3/Test-3 的场景或闭环结果用于训练或 checkpoint 选择；训练前核对这 6 个评估场景与 Train-6 交集为空。
- 记录场景样本分布、每个场景被曝光的次数、Ray Depth/HIT 直方图、near/far、场景单位，不让某单一 Outdoor 场景主导训练。
- 如果正式的 Outdoor-only validation 没有与训练明确分离的场景，应采用**训练前冻结的固定 checkpoint（如 step 40k EMA）**为主结果，而不是在 Test/Dev 的 G3 中挑最优步数。

### 6.2 训练与公平对照

**建议起始协议**（以代码审计和 1k-step 吞吐测试后的正式冻结版为准）：

```yaml
stage: outdoor_pretrain
initialization: scratch
outdoor_dataset: macarons_pp_train6
mp3d_training_samples: 0
steps: 40000                 # suggested common first-stage budget
seeds: [20261031]           # example; use frozen common seed(s)
global_batch: audit_and_freeze
num_gpus_preferred: 4       # per model, if available
checkpoint_steps: [0, 1000, 5000, 10000, 20000, 30000, 40000]
checkpoint_policy: fixed_ema_step_40000
planner: frozen_original_distp1
K_inference: 4
nwin_mp3d_main: 5
```

- V3.1 应尽可能继承 EXP6 V3-T0 的训练设置，仅移除 MP3D 数据。V4 则应与 V2 的 Ray Encoder/Flow 训练实现尽可能一致；由于 VAE Loss 不同，跨架构比较标为**匹配训练预算的性能对照**，而非严格同损失对照。
- 若 V4/V3.1 在 40k 仍未收敛，请保存固定 40k 主结果。后续延长到 60k 需独立追加阶段、提前声明规则并对两组预算公平处理；不要看到 Test-3 的 G3 结果后只单方面延长某一模型。
- 相同数量 `optimizer.step()` 不等于相同 FLOPs，也不等于相同有效监督 ray 数；必须报告 Samples、Effective Rays、GPU-hours。
- 每次正式训练都记录 EMA 与 raw model、Optimizer、Scheduler、GradScaler、完整 RNG 状态用于 resume，尤其检查 DDP 和重新排队时不会悄悄重复/跳过训练数据。

### 6.3 优化器/损失（真实设置以审计冻结为准）

- V4：原 V2 的 Flow Matching Velocity MSE（有效未来射线上），如 `E[mask * ||vθ(z_t,t,P1,C) − (z1−z0)||²]`，按照原协议归一化。
- V3.1：原 T0 的 Flow Matching + VAE Reconstruction + β·KL；分别记录原始与加权值。
- 对齐原 Flow 的常规采样时间 t、噪声分布、输出尺度、Depth/HIT 编码、K=4 / 10-step Euler；**不可**为了提升新模型实验结果私改其输出语义。
- 训练期间 CPU/IO/GPU 优化允许，但不得引入不同的新监督目标、GT Oracle Gain Loss 或当前冻结范围以外的 Planner。

## 7. G0：正式训练前必须通过的测试

| Gate | 具体检查 | 硬性标准 |
|---|---|---|
| G0-1 Base audit | Old V2/Mix/B2/E2 checkpoint 和训练来源 | SHA/config 真实可追溯 |
| G0-2 Data split | Train-6 与 Test-3/Dev-3；MP3D zero-shot | 严格无场景/房屋泄漏 |
| G0-3 DDP | 4 卡同步、global batch、grad accumulation | 世界大小改变后 loss 归一化一致 |
| G0-4 V4 shape | F0/F1、Ray Samples、X-Attn、2D Flow | 输入输出一致、无 NaN/Inf |
| G0-5 X-Attn gradient | Q/K/V/输出投影、U-Net、Flow | 非零且有限；可小样本过拟合 |
| G0-6 Camera conditioning | 同 P1 不同合法 Camera，射线/输出不同 | 必须真的对 Pose 有条件响应 |
| G0-7 Visibility mask | UNKNOWN≠FREE，invalid/all-masked rays | 不产生虚假合法样本或 NaN |
| G0-8 V3.1 VAE | μ/logvar、重参数化、KL、Recon | 有限数值，可反向传播 |
| G0-9 Zero leakage | Future Depth 仅作 label、未选 Camera 仅作评分 | P1 不包含未来观测；GT 不泄漏 |
| G0-10 Checkpoint | Resume、EMA、DDP RNG、global step | 重启后轨迹/训练可复现至容差 |
| G0-11 Baseline replay | 原 V2/B2 Planner | 固定种子下重放原 G3 一致 |
| G0-12 Throughput | 1k-step smoke、VRAM、NFS | 记录吞吐；不 OOM |

**不要求相机条件改变任何像素必然有益**；G0-6 只验证 Geometry/Camera 真实参与计算，而不是网络忽略 Camera。对 Pose Perturbation 采用同状态的合法相机候选；测试 3D 采样坐标应按真实相机外参变化。

## 8. G1/G2/G3：冻结权重后的双域评估

### 8.1 两域必须在 Outdoor 阶段结束后同时评估

- **Outdoor (Macarons++ Test-3 + Dev-3)**：G1 Depth/HIT（UNKNOWN 切片、HIT/Brier、MAE/ES）、G2 同状态真实选择 Gain 和 Top-1、G3 30 EP×100 steps；AUC@100、Final@100、Cov@20。
- **Indoor (MP3D Zero-Shot)**：不用 MP3D 更新任何权重，直接加载 Outdoor-only EMA；G1/G2、G3 10 EP×150 steps；AUC@150、Final@150、Cov@20、停滞 EP。
- **固定 G3 协议**：合法 Candidate Generation、P1、DistP1、K=4、随机采样、坐标系、Camera movements、Coverage metric、停止规则、初始种子和评估分组均与现有 EXP6 保持一致。
- **不允许**用整个 Test-3 或 MP3D G3 选最优 checkpoint。主结果一律使用训练前冻结的 checkpoint-selection 规则。
- 记录评估每一步 walltime、调用次数、VRAM、平均新表面 Gain（若可获得）、Episode 级 Coverage 曲线和轨迹。

### 8.2 历史固定对照数据（必须复核现有 `g3.json`）

**MP3D：5 楼层×2 starts，150 步（历史值）**

| Model | AUC@150 | Final@150 | Cov@20 |
|---|---:|---:|---:|
| MAG-G | 0.278 | 0.377 | 0.169 |
| MAG-B | 0.292 | 0.377 | 0.165 |
| B0 | 0.322 | 0.469 | 0.195 |
| B2 | 0.334 | 0.530 | 0.197 |
| E2-D100 | 0.364 | 0.553 | 0.195 |
| V3-T0 Mixed | 0.311 | 0.463 | 0.183 |
| V3-T1 Mixed | 0.331 | 0.508 | 0.184 |
| V2 Original | **待读取已有报告并核对协议** | **待核对** | **待核对** |
| **V3.1 Outdoor-only** | **本轮实测** | **本轮实测** | **本轮实测** |
| **V4 Outdoor-only** | **本轮实测** | **本轮实测** | **本轮实测** |

**Macarons++：6 场景×5 starts，100 步（历史值）**

| Model | AUC@100 | Final@100 | Cov@20 |
|---|---:|---:|---:|
| MAG-G | 0.704 | 0.905 | 0.507 |
| MAG-B | 0.685 | 0.891 | 0.504 |
| B0 | 0.729 | 0.884 | 0.622 |
| B2 | 0.730 | 0.906 | 0.572 |
| E2-D100 | 0.711 | 0.880 | 0.570 |
| V3-T0 Mixed | 0.702 | 0.872 | 0.568 |
| V3-T1 Mixed | 0.712 | 0.883 | 0.566 |
| V2 Original | **待核对** | **待核对** | **待核对** |
| **V3.1 Outdoor-only** | **本轮实测** | **本轮实测** | **本轮实测** |
| **V4 Outdoor-only** | **本轮实测** | **本轮实测** | **本轮实测** |

**请不要把 MAG-G/MAG-B 省略，也不要将 B2/E2 Mixed + Finetune 的数值误标为 Outdoor-only。**

### 8.3 强制输出的对比表

**Table A — Outdoor-only 后的 Macarons++**：Model、Training Data/Stage、AUC@100、Final@100、Cov@20、Δ vs MAG-G、Δ vs MAG-B、Δ vs V2 Original/B2、推理时延、GPU-hours。  
**Table B — Outdoor-only 后的 MP3D Zero-Shot**：同上，MP3D 150 steps；标明哪些模型真正 zero-shot，哪些为历史 Mixed/Fine-tuned，仅用于横向性能对照。  
**Table C — G1/G2 的两域质量与 Camera Ranking**：Future Depth Unknown MAE、HIT Brier/ES、选中 Camera 的**真实单步新增 Coverage Gain pp**、Top-1；Regret 只放附录，不作为主表。  
**Table D — 每场景/楼层及每个起点**：MAG-G/MAG-B/V2 Original/B0/B2/E2/V3-T0/T1/V3.1/V4，逐场景 AUC、Final、Cov@20、成败与轨迹图。  
**Table E — Stage 2 Indoor Fine-tune（待用户批准后填写）**：Outdoor checkpoint、MP3D Zero-shot、MP3D Fine-tuned 的 AUC/Final/Cov@20 配对变化。  
**Table F — Stage 2 Outdoor Retention（待用户批准后填写）**：同一个模型 Indoor Fine-tune 前/后 Outdoor 的 AUC/Final/Cov@20，展示遗忘或迁移。  
**Table G — Planner-only n_win（可选独立后续）**：冻结模型下 `n_win=5/7/9` 的 MP3D AUC、Final、Cov@20、计算代价。

所有新结果：**scene/floor-mean then scene-level bootstrap 95% CI，逐场景胜/负数与配对差**；MP3D 5 个楼层独立簇，Macarons++ 6 个场景独立簇；不能把 10/30 EP 中 150/100 步视作独立样本。历史数据如果不是完全相同评估协议，要单独标记而不能硬并。

## 9. `n_win` 扩大视野：独立消融，不混入新模型主实验

既有 GT 反事实参考（MP3D 368 个固定状态）：`n_win=5/7/9` 对应的 GT 下一步被选真实 Gain `0.521/0.549/0.557 pp`，Top-1 `0.757/0.845/0.894`；整层评分反而降到 `0.508 pp / 0.724`。这些**不是预测 Depth 的 G3**，不能推断 V4 会同样改善。

- **先审计 `n_win` 实际控制的模块**：输入的局部块、Ray Sampling 有效空间、DistP1 计数窗口、还是它们的组合？是否改变 3D U-Net 的缓存/尺寸和边界？
- 主实验无论 V4/V3.1，MP3D 统一使用旧 `n_win=5`。
- Outdoor 第一阶段结果冻结后，若用户决定继续，**保持模型 checkpoint 和候选集不变**，只在分离的 MP3D Scoring Window 实验中先运行 K=4 固定状态对照；无法支持窗口外预测时标记“不可作为纯评分消融”，不得硬扩大 mask。
- 一旦通过预注册固定状态测试，可运行 10 EP×150 steps 的 `n_win=7/9` G3，单独列为 Planner Ablation，不能把其收益归因于 V4 的新 Encoder/X-Attn。
- 不在此次主阶段同时扩大模型输入范围、近远 Depth、候选动作数、传感器 FOV。

## 10. Stage 2：MP3D Indoor Fine-tuning（**预先设计，暂不启动**）

触发条件：Outdoor-only 两组权重已冻结且 G1/G2/G3 双域结果完整；用户明确同意开始 Indoor Fine-tuning。

- V4-FT 从 **V4 Outdoor-only EMA** 初始化；V3.1-FT 从 **V3.1 Outdoor-only EMA** 初始化；不得从 B2/E2 或 Mixed V3 加载替换。
- 第一组主协议仅使用 MP3D Train（如历史 MP3D train 114 个楼层，具体以审计后的独立划分为准），不混 Outdoor 数据，评估 Indoor 专业化与 Outdoor Forgetting。
- **建议**先做 5k-step 小预算共同 FT 对照（以此前 E2 5k 工程经验为参考而非固定最优值），同一学习率/EMA/步数选择规则或分别记录事前冻结差异；主报告必须同预算且不挑测试场景最优。
- FT 前与 FT 后**两个域都进行 G3**：MP3D AUC@150/Final/Cov@20、Macarons++ AUC@100/Final/Cov@20。
- 若 Outdoor Retention 明显下降，再单独设计少量 Outdoor Replay 或 Regularization 的 FT 消融；不可悄悄将第一组 Indoor-only FT 改成 50/50 Mixed。
- FT 的参数可训练范围（all / freeze Encoder）是独立变量，应先固定一致主协议；如果 VAE KL/Recon 与 U-Net 不同，记录结构差异，保持可解释性。

## 11. Slurm / 8 GPUs / NFS 资源调度

用户过往曾分配 gpu117、gpu115 的 4 GPU 资源；**这些只是历史信息**，必须实时检查 `squeue -u "$USER"`、`scontrol show job <id>`、`nvidia-smi`、已跑进程及剩余 walltime。绝不能凭旧截图认定仍在运行或空闲；不能随意 scancel 用户其他工作。

**资源足够时的首选布局**：

| Worker | GPUs | 任务 | 要求 |
|---|---|---|---|
| A | 4 | V4 Outdoor-only Scratch | DDP，独立 Logs/ckpts |
| B | 4 | V3.1 Outdoor-only Scratch | DDP，独立 Logs/ckpts |
| CPU | 共用 CPU/内存 | 数据 Manifest、结果解析、可视化 | 缓存/IO 分流，不读爆 NFS |

- 按实际集群 `salloc` / `srun --jobid --pty` 或 `sbatch` 规则申请/附着；禁止绕过禁止直连 compute node 的规定。
- 先单 GPU smoke、再 4 GPU 1k steps 测试吞吐、OOM 和 NFS 的读压力；据实估算完成预算。由于两组模型的显存不同，可以各自启用 Gradient Accumulation 来匹配 Global Batch。
- Work dirs 中保存 Resume Checkpoint（model、EMA、optimizer、scheduler、RNG、sample cursor），保证 Slurm 周期重启可以恢复。
- 同数据 Manifest 可采用只读共享缓存，但训练时分离随机采样；大型随机读取尽量 local scratch/cache，避免重复遍历 NFS。
- 不为占满 GPU 而提交无效的第三训练；若两组 4 卡不可同时运行，按实际资源运行，报告 GPU-hours 和 Walltime。
- 不给用户承诺未来完成时间；每个阶段达到里程碑时发布**已真实完成**的日志、指标和 Job IDs。

## 12. 项目管理、日志、可复现性

- 版本目录必须分别为用户指定的 `exp6_v4` 和 `exp6_v3.1`。
- 独立 `STATUS.md`、训练配置、事件/异常 `DEVIATIONS.md`、训练报告、测试结果和 G3 表格。严格保留 EXP6 V2/V3/E2/Task2 既有结果，不覆盖。
- 训练配置冻结后记录 Git SHA、配置文件 SHA256、训练 Manifest hash、Model EMA hash、随机种子、Slurm Job ID、GPU 型号、总 GPU-hours。
- 按现有 Git 规范提交并 push 新代码与轻量报告；建议前缀 `[V4][setup] ...`、`[V4][train] ...`、`[V3.1][train] ...`、`[EXP6][eval] ...`；不添加 Claude Co-Author，不 Force Push，不提交巨型缓存、raw dataset 或权重。
- 若启动失败（GT 泄漏、Checkpoint 不一致、DDP OOM、Loss NaN），停止该实验并明确记录，保留另一实验正常运行；只接受满足同等配对原则的恢复。
- 正式**不根据同一评估集反复调参并称为 blind test**；对反复用于方法选择的 EXP6 评估场景诚实标为 diagnostic/previously used。

## 13. 论文创新叙事与本次训练的关系

**准确的已有能力**：给定历史 Partial TSDF + Candidate Camera Pose，对每个候选生成随机 Future Depth/HIT 并估计 Coverage，再选择下一 Pose。**不是**网络先直接预测任意下一 Pose；不要在论文里写“learned camera pose generation”直到实际实现与实验支持。

候选论文问题：*Can camera-conditioned generative future observations guide next-best-view selection efficiently without explicitly reconstructing the whole unseen world?*

初步技术线索：3D Multi-scale Scene Memory、Camera-Ray-Aligned X-Attn、2D Flow 的 Pose-Conditioned Future Observation Generation。它们只有在新增模块带来**准确、可重复的贡献**时才能作为论文 Claim。普通的 ray-conditioned novel view/depth generation、Frontier、NBV Coverage 不是未经探索的新概念。

建议未来必须准备的关键消融：

1. Pose Condition On/Off / Wrong Pose Perturbation：验证输出依赖目标相机；真正评估新视角的几何条件。
2. 3D U-Net + 原 Ray Sampling U0 vs U-Net + X-Attn V4（数据/初始化/预算严格匹配），证明 Attention 因果效应。
3. V3.1 Outdoor-only vs V3-T0 Mixed（注明域曝光差异）；Outdoor→Indoor Fine-tune 前后双域测试。
4. Model latency/peak VRAM / per-camera generation speed vs MAGICIAN，以及同动作空间 G3 结果。
5. 2D Rectified Flow 的 NBV-aware weighting 或候选 gain-aligned learning **在另外的实验中再设计**；不能加入当前 V4/V3.1 主线。
6. 若未来升级到 Geometry-Aware Bias/全局上下文或连续 Pose 优化，必须作为单独新方案，不要先行宣称本轮已实现。

## 14. 执行顺序、阶段回复与 Go/No-Go

1. **Phase 0 — Audit**：读所有已有代码/报告，产出真实性清单；检查 GPU，锁定 Data Split、总预算、EMA 选取和统一 G3，提交审计文件。
2. **Phase 1 — Implement**：创建两实验目录、V4 XAttn、V3.1 Outdoor data route、配置与测试；保留旧 V2/G3 可重放。测试通过后固定 G0/Protocol。
3. **Phase 2 — Launch**：若两组 4GPU 资源可用同时启动，完成 1k smoke 与 40k（或前述冻结预算）；只在**全部关键 G0 PASS** 后提交正式作业。
4. **Phase 3 — Outdoor Checkpoints**：在固定训练步冻结 V4、V3.1 EMA，输出训练曲线、参数量、吞吐、GPU-hours、数值稳定性。
5. **Phase 4 — G1/G2/G3 Both Domains**：同一冻结权重上先验证数据语义再运行完整 10+30 EP；产生 Table A–D、与 MAG-G/MAG-B/B2/E2/T0/T1 的逐场景比较及配对 CI。
6. **Phase 5 — Review**：提交对 V4/V3.1 和 CVPR 创新性的有证据结论、未来 U0 与 n_win 的单独 Go/No-Go；**此时停止，不自动启动 MP3D FT、V4-U2 或 Flow 重设计**。
7. **Phase 6 — Future after approval**：按第 10 节进行 Indoor Fine-tune 和 Outdoor Retention；若用户希望，可并行添加 n_win Planner-only 消融。

**第一条阶段回复（Phase 2 正式跑起来之后）必须包含**：

- 真实 code SHA、Checkpoint 初始化说明（scratch）、参数量。
- 数据：Train-6 以及训练中 MP3D 样本严格为 0 的审计结果。
- 两个作业的 Job ID、GPU 数量/型号、正式总步数/global batch、ETA **不用猜测或承诺**、日志/ckpt 路径。
- G0 测试结果及已发现偏差。

**第二条阶段回复（Phase 4 双域 G3 完成后）必须包含**：

- MP3D / Macarons++ 的 AUC、Final、Cov@20 统一表，含两组 MAGICIAN、B2/E2 和 V2 Original（可验证时）。
- 每个新模型相对 MAG-G/MAG-B/B2 的场景配对 Δ、CI、赢/输数。
- 是否发现 Attention 真实有帮助、VAE Outdoor-only 的差异（明确性能 vs 因果限制）。
- Outdoor Fine-tuning Stage 2 的准备状态与需要用户决定的问题；不擅自启动。

## 15. 可直接复制给代码 AI 的执行话术

> 请完整阅读并执行附件 `EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md`。本轮创建 `experiments/exp6_v4/`（3D U-Net + Ray-Aligned Cross-Attention + **原** 2D Rectified Flow）与 `experiments/exp6_v3.1/`（V3-T0 Simple 3D VAE + **原** Ray Encoder + Flow）。两组都从零初始化、只用 Macarons++ Train-6 做 Outdoor-only 第一阶段训练；如有两组合法四卡 GPU 则 4+4 并行。先审计原始 V2/V3 训练 lineage、Checkpoint、场景划分、Input/Scoring P1、模型和评分窗口等真实配置，再做代码和 G0。不要直接加载曾看过 MP3D 的 B2 来冒充 Outdoor-only。**阶段1不修改 2D Flow、DistP1 或 Planner；MP3D 主结果 n_win=5。**
>
> 训练结束按冻结 EMA 规则选择 checkpoint，不允许看 Test-3 G3 调 checkpoint，然后在 Macarons++ 和 **完全未参与训练的 MP3D** 分别跑全套 G1/G2/G3。必须输出 AUC、Final、Cov@20 与逐场景 CI，对照表直接列 MAG-G/MAG-B/V2 Original/B0/B2/E2/V3-T0/V3-T1/V3.1/V4，并标注历史组的训练数据差别。必须先完成 Outdoor-only 双域正式评估；Indoor Fine-tuning 的代码/配置可以准备，但不要自动启动微调。`n_win=7/9` 仅作为冻结模型 Planner-only 独立消融，暂不影响主结果。所有代码和报告分版本独立保存，使用真实 Slurm 资源，记录资源成本/异常并提交推送轻量代码报告。只要 G0 通过，可自主完成两组第一阶段训练和完整 G3；向我汇报启动状态和最终双域结果即可。

---

**验收原则**：只有真实运行、无 GT 泄漏、对照语义一致、含 MAG-G/MAG-B 的双域 G3 AUC/Final/Cov@20 表、完整资源与错误报告，才算完成本任务。不能用 G2 Regret 或人工推测替代 G3；不能把独立历史 checkpoint 的数值差当作 X-Attn 的严格因果收益。
