# Seen2Scene · EXP6 V5：完整 AI 工程实施任务书

**状态**：可以交给连接 `seen2scene` 远程仓库与 Boise State 集群的代码 AI 执行。  
**架构依据**：同目录的 `EXP6_V5_ARCHITECTURE_DESIGN.md`（**必须先完整阅读**），其四项核心决策已经由用户批准。  
**执行范围**：V5 3D U-Net + Geometry-Guided Local/Coarse Ray Query + Multi-Surface Belief + 原 2D Flow；Outdoor-only Warm-start 训练、双域固定状态与完整闭环评价、文档归档。  
**本任务不修改 Planner，不启动其他任务**：不做 U0、V4.1、V3.1 延长、MP3D Fine-tuning、Exploration Memory、Frontier/Beam、n_win 参数搜索。

## 0. 首先明确：这不是让你自行发挥的新模型提案

已确认：

1. Full-Ray **Multi-Surface** Belief，必须表达被当前第一个 Hit 遮挡的潜在表面；不是单纯 First-Hit Depth 回归。
2. 每条 Camera Ray 动态检索**局部几何与粗尺度上下文**，不是固定 32 个 ray 3D feature sampling，也不是对所有 3D token 做无约束全局 Attention。
3. 从 **V4 Outdoor-only EMA@40k** Warm-start，**保留 3D U-Net 和原 2D Rectified Flow** 的相容权重；设计特征分布迁移，不能把随机新 Decoder 直接硬接旧 Flow。
4. **Ray Latent 与 Belief Summary 共同进入旧 Flow**；旧 Flow 继续输出 K=4 Depth/HIT。
5. 默认 P1：**网络 D100、DistP1 评分 S10**。Experiment A 的 D100 评分在 Outdoor 明显退化，不得将 A1 改为 V5 默认。
6. **训练起始域 Outdoor-only Macarons++ Train-6**，MP3D 数据绝不进入此次训练；MP3D 仅 Zero-Shot 评估。

必须先根据真实仓库核对路径、shapes、checkpoint、数据来源和标签能力。以下示例函数和 config 是**预期接口，不是宣称源码已经存在**。

## 1. 首轮要审计的真实代码、文件和科学协议

检查并写入 `AUDIT_V5.md`：

- `seen2scene/futureview/model.py`：V4 U-Net F0/F1、32-point sampling、Conv1d Ray Encoder、Ray X-Attn 的精确 shape、数据流、loss、EMA、Flow 条件通道数、Depth/HIT 语义。核实旧模块在 G0 中独立可运行。
- `work_dirs/exp6_v4/checkpoints/final.pt`：确认为 Outdoor-only @40k EMA；真实 sha256（历史片段 `1b31afff...`）；完整权重、Optimizer/EMA/Step 状态和 GPU dtype。
- `experiments/exp6_v4/` 的冻结协议 `FROZEN_PROTOCOL.md`、`G0_VALIDATION.md`、`G1_G2_EVAL.md`、`G3_OUTDOOR.md`、`G3_INDOOR_ZEROSHOT.md`、`V4_COMPARISON_TABLES.md`，以及 Experiment A 的真实报告/commits。不得凭对话概括替代仓库数字。
- `Macarons++ Train-6` 清单、训练历史 P1 与候选真实目标 Depth/HIT、Pose/K、有效射线的 mask、用于生成 **多重表面交点** 的训练 Mesh/API。明确场景级 Train-6 与 Dev-3/Test-3/S-c 的隔离情况。
- 3D 坐标：本地 P1 网格、世界坐标、候选相机 Pose、每级 feature voxel 物理尺度；Outdoor 全局与 MP3D 每10步局部窗口处理的差异。
- Planner `futureview.scoring.new_set_from_depth`/`band_dil`、S10 与 D100 两个历史 P1 来源、距离权重、K4、`n_win=5`、合法候选和 G3 动作环境，冻结。
- 资源：L40S 的有效 GPU 数量/配额、GPU 余量、已有作业、NFS 数据读取、DDP/梯度累积和 checkpoint resume；**不得擅自取消现有工作**。

**标签阻塞 Gate**：本版本已选显式 Multi-Surface；若 Train-6 训练 Mesh 或可靠多表面射线监督无法获得，必须说明具体不足并停止声称“已实现 Multi-Surface 训练”。可完成基础代码和单元测试，但不得在没有可靠标签的情况下将首次 Hit 后全部标为 Free 或随意造 GT。可提出经真实多视角可观测支持的替代标签方案并报告其监督覆盖率；不允许暗中换成仅 First-Hit Label。

## 2. 建议文件结构和命名

```text
experiments/exp6_v5/
├── README.md
├── ARCHITECTURE.md                # 从架构任务书同步实际参数与偏差
├── AUDIT_V5.md                     # 代码、数据、checkpoint、坐标、标签审计
├── FROZEN_PROTOCOL.md              # 本轮训练与评估前冻结
├── CHECKPOINT_LINEAGE.md
├── STATUS.md
├── configs/
│   ├── train_mesh_labels.yaml
│   ├── warm_bridge_outdoor.yaml
│   ├── warm_joint_outdoor.yaml
│   ├── v4_continue_control.yaml
│   ├── eval_macarons.yaml
│   └── eval_mp3d_zeroshot.yaml
├── scripts/
│   ├── audit_v4_shapes.py
│   ├── build_multisurface_labels.py
│   ├── probe_ray_memory.py
│   ├── train_v5.py
│   ├── eval_g1_g2.py
│   └── eval_g3.py
├── tests/
│   ├── test_ray_world_geometry.py
│   ├── test_multisurface_targets.py
│   ├── test_retrieval_masking.py
│   ├── test_v5_forward_backward.py
│   ├── test_teacher_migration.py
│   └── test_g0_invariants.py
├── reports/
│   ├── G0_VALIDATION.md
│   ├── LABEL_QUALITY.md
│   ├── TRAINING.md
│   ├── G1_G2.md
│   ├── G3_OUTDOOR.md
│   ├── G3_INDOOR_ZEROSHOT.md
│   ├── V5_COMPARISON_TABLES.md
│   ├── V5_RESOURCES.md
│   ├── V5_DEVIATIONS.md
│   └── FINAL_REPORT.md
└── figures/
```

若实际仓库已有基础组件，最大化重用，**不要复制一套僵化代码或破坏 V2/V3/V4 默认行为**。可放公共模块于 `seen2scene/futureview/` 下清晰命名的 `geometry_query.py` / `surface_belief.py` / `v5_model.py`，以源码检查结果为准。实验 checkpoint、Cached GT、G3 trajectories、临时输出放 `work_dirs/exp6_v5/`，不加入 Git。推荐模型标识：`V5-Warm12k-S10Score`；单独对照 `V4-Continue12k-S10Score`，不称作 U0。

## 3. 新模块必须实现的接口与不变量

```python
# interface pseudocode; use actual repository tensor/layout conventions
scene_memory = v5.scene_encoder(p1_d100)  # inherits V4 U-Net
ray_query = v5.ray_embed(camera_pose, K, pixels, physical_depth_range)
memory_tokens, geo_meta, mask = v5.retrieve(scene_memory, ray_query, p1_d100)
ray_latent = v5.geometry_cross_attention(ray_query, memory_tokens, geo_meta, mask)
belief = v5.surface_belief(ray_latent, probe_depths)  # no 3D U-Net feature grid_sample
belief_summary = v5.summarize_belief(ray_latent, belief, physical_depth_range)
flow_condition = v5.condition_adapter(ray_latent, belief_summary)  # same shape as V4
future_depth_hit = v5.original_flow.sample(flow_condition, K=4)
scores = original_distp1(future_depth_hit, scoring_p1=p1_s10)
```

### 3.1 3D Memory 和 Ray Geometry

- 从实际 F0/F1 Feature Grid 取得每个 Token 的可定位中心 `x_j`、`f_j`、feature voxel size `s_l`、P1 对应的 FREE/BAND/UNKNOWN 比例；这些从**当前历史**派生。
- 从 `Pose + K + pixel` 建 `o_C, d_r, near/far`，单位和坐标系统一。不得把已知 `GT candidate depth` 当作 ray length 或掩码。
- 对每个 ray 与 token 算 `tau=(x-o)·d`, `delta=||(x-o)-tau*d||`，`tau>0` 与合法视锥、区域有效性检查。`rho(tau, s_l, camera intrinsics)` 的距离门限随物理尺度合理变化。
- F0 Local ≤24、F1 Local ≤12、Coarse ≤8 作为初始显存预算，实际有效 Token 数可变，**没有固定的 32 个 ray depth sample**。用 grid-specific 近邻检索、空间哈希或分块，不做全部 `7296×N_3D` 全对矩阵。
- 允许 UNKNOWN Token（区域内的未知仍是合法场景先验），区域外 token mask。空 Local 的 ray 使用 Coarse/learned null 安全回退，不许 NaN。
- CrossAttention 用相对几何编码/bias，首版建议 dim≈192、4 heads，允许从性能分析合理调整并记录。输出 `[B,R,d]`；形状按真实模型实现。

### 3.2 Surface Belief（显式多表面）

- `b_r(t)` 是有限深度邻域内存在表面的条件概率，非 first-hit PDF，不要求 `sum_t b=1`。支持 **3m 墙 A 和 9m 墙 B 同时高概率**。
- 可用共享 `SurfaceMLP(z_r, Fourier(log1p(t/voxel_scale)), t/t_far)`，表面 sigmoid。按实际检测到的多表面标签训练。在不同 depth scale 探针上查询 `b` **不等于**从 F0/F1 采样旧的 32 个 3D 点。
- `BeliefSummary` 从例如 16 个物理尺度自适应/对数深度探针汇总较少通道，保留远近多峰位置信息。最后 Adapter 投影至 V4 Flow 条件的同一维度与空间布局。
- 需要验证 `Flow loss -> Adapter -> Summary/Belief -> z -> Attention` 的梯度不是零；如果摘要来自 `b_r(t)`，不得无意 detach。没有可靠 Mesh 的深度区间 mask，不得当 0。
- **不允许将 `b_r(t)` 中的遮挡后表面直接累加到当前 DistP1 score**，也不允许把它们写回历史真实 P1。

### 3.3 2D Flow 必须原样保留

- 从 V4@40k 载入所有匹配的 Flow 参数；原 Depth/HIT 编码、FM Time Sampling、噪声、Sampler/Euler steps、K=4、mask 与 Loss 不变。
- 只有 Flow 的**条件特征来源**从旧的 32-point + Conv1d + Ray X-Attn 变为 `Ray Query + Multi-Surface Summary`。
- `V5` 正式推理 call graph 中不得使用旧 `RayEncoder`、旧 X-Attn 或 `grid_sample` 固定 32 点来取 3D U-Net Feature；只有临时训练 Teacher 可使用旧路径。

## 4. 训练前生成 Multi-Surface 标签

### 4.1 首先做几何求交正确性审计

- 训练 Mesh 必须来自**Macarons++ Train-6**。原始 simulator GT 当前 `first_hit` 与求交列表 `t_1<t_2<...` 第一个合法值大体一致；不一致的 Ray 分析是否由双面/薄面/透明/无效 depth/网格界限导致。
- 若一条 Ray 穿过两面墙，训练标签必须包含 `t1` 和 `t2`（及更多），不能只用 GT Depth 的 `t1`。
- Double-sided Mesh、非封闭网格、多三角形共面/近邻交点、ray/tangent、长度截断、网格坐标变换和实际深度的定义必须统一。求交失败或场景模型不完整时 mask。
- Mesh 只生成离线训练 Label。不得被网络在推理/G3 时访问；不能在训练缓存中偷偷加入未来 GT scene feature 作为模型输入。

### 4.2 高效 Label 缓存

建议对 Train-6 的历史状态与目标 Camera 选取分层 Ray 子集生成交点列表和 sampled `(ray_id,t,label,valid)`：
- 先覆盖全场景和不同 `near/far`、FREE/BAND/UNKNOWN、多 UNKNOWN segments；包含首个和遮挡后的表面正例。
- 抽样正例邻域 + 可信负例邻域，避免全负例让模型学会 `b=0`；后方无完整几何证据时 `valid=0`。
- 每条 Ray 的 `t` 使用真实米数，`Δ` 由物理 feature / P1 voxel 尺度定义；对 Indoor/Outdoor 的绝对尺度和 `log depth` 同时编码。
- 缓存以 Train-6 scene / pose / K / near-far / Mesh SHA / GT update timestamp 为索引；保存 label schema 版本与 checksum。缓存数量按 GPU/NFS 能力调整，不在在线训练热路径对所有像素与完整 Mesh 做昂贵求交。

`LABEL_QUALITY.md` 输出：有≥2表面 Rays 占比、遮挡后正例数量、求交 vs Simulator 首 Hit 的差异统计、标注不可靠 / masked 比例、三组可视化与具体错误来源。若训练数据中多表面 Rays 过少、失真严重，务必报告并停止假设“Multi-Surface 已学会”。

## 5. Warm-start 训练：建议固定初始预算 12k

**注意**：这次不是从零开始的 V5 40k；12k 是**相对 V4@40k 的新增优化步数**，例如总谱系可标 `V4 Outdoor40k → V5 Bridge2k + Joint10k`，不能伪装为“V5 scratch 12k”。真实训练预算在资源审计后预先冻结并写配置。

### 5.1 迁移矩阵和安全加载

- `V4 → V5 3D U-Net`: strict name+shape matching，日志列出成功加载参数及 SHA。
- `V4 → V5 2D Flow`: strict matching，严禁随机重置而不发现；日志列出匹配参数总量。
- `new Geometry Query + Belief + Adapter`: 随机初始化，参数量、梯度分布独立记录。
- `old Ray Encoder/X-Attn`: 仅冻结 Teacher 用，删除/不调用于正式 V5 模型 forward；保存 teacher feature 迁移日志。

### 5.2 Phase A：Bridge2k

- 冻结旧 U-Net 和 Flow；新 Query、Belief、Adapter 训练。
- 对同一历史 Train-6 P1、同一 Camera 从固定 V4 Teacher 读取旧条件特征 `H_old`；新 V5 得 `H_new`。
- Loss 包括 `L_bridge`（特征尺度迁移）和 `lambda_surf L_surf`；`L_bridge` 在阶段内递减，以免模型只模仿旧 Ray Encoder。记录旧/新 feature 均值、方差、余弦和 Flow 预测误差。
- G0 small overfit / 2nd-hit positive 检验成功后才能继续。缓存 Teacher Feature 的策略依据 NFS/GPU 实测，可即时冻结计算或离线缓存。

### 5.3 Phase B：Joint10k

- 逐步/部分解冻 2D Flow 与 U-Net，新 Query+Belief 始终可训练；`lambda_bridge=0` 或按预注册规则快速衰减至 0。
- 优化：`L_V4_original + lambda_surf L_surf`。原 Flow 目标监督只来自 Train-6 当前合法 Candidate Future Depth/HIT。Belief 目标来自 Train-6 多表面 Mesh。
- 新参数学习率建议量级 `1e-4`，继承网络较小 `1e-5`，具体配方先依据 V4 Optimizer/Scheduler 与吞吐审计。原 EMA 不要被重置为无法解释的全新权重；清晰记录 EMA 新旧权重与训练步数。
- 同一训练集抽样次序与随机种子记录；DDP 下有效 Rays 与 Belief 查询点个数分别全局归一化。记录 FM、Surface、Bridge 各自 loss 及权重、模型/Flow 梯度、Surface Positive Recall、GPUs/VRAM/step。
- Checkpoint 保存模型、EMA、Optimizer、Scheduler、Step、RNG、Data sampler state 和输入 schema；断点恢复后结果与不停训练应在可实现容差内等价。

### 5.4 必须有延长训练对照

从相同 `V4 Outdoor@40k` 再训练相同的 Outdoor 数据、新增约12k步的 `V4-Continue12k`；训练预算、有效样本和优化规则尽量匹配，记录 V5 Bridge 期 Flow 未更新造成的差别。可根据合法 GPU 资源并行，但**不得取消现有作业强行凑并行**。该对照主要回答“只是多训了一段，还是新系统有价值”，并不能把 `V5−V4` 纯归因于 Attention，因为 V5 同时使用额外 Mesh 监督和不同优化阶段。

**若资源不足**：优先完成 V5 + 历史 V4 G1/G2 和一个小规模闭环，再给出资源预算，不能假装已跑 `V4-Continue12k`。但正式架构优劣结论必须明确预算混杂。

## 6. G0：所有正式训练前必须通过

| Gate | 测试 | 硬要求 |
|---|---|---|
| G0-1 | 旧 V4 复现 | 输入、输出、EMA、原 Planner 均可重放；原代码默认路径不变 |
| G0-2 | 无信息泄漏 | V5 Forward 只读历史 P1/候选 Camera；训练 Mesh 只到 target head；MP3D eval GT 绝不训练 |
| G0-3 | 空间几何 | Camera Pose/K、local-to-world、物理尺度、F0/F1 3D coords 正确；不同区域窗口无坐标复用 |
| G0-4 | Memory retrieval | cone、front/back、relative bias、mask、空 token fallback 正确；UNKNOWN 不当 invalid |
| G0-5 | 无 32-point 推理 | V5 推理/Profiler 完全不调用旧 3D Ray Sampler / Conv1d / V4 X-Attn |
| G0-6 | Multi-Surface GT | 3m/9m 双表面合成测试；被前方墙遮挡的第二表面正例保留；不可信区间 masked |
| G0-7 | Belief & Flow 连接 | Belief Summary 改变能影响 Flow 条件输出；二者梯度可达，数值有限 |
| G0-8 | Warm-start 权重 | U-Net/Flow 与 V4 checkpoint 一致，旧 Teacher 不进入 V5 inference |
| G0-9 | 训练 & DDP | 小批过拟合、Mask 归一化、GradAccum/DDP、EMA Resume、K4 生成通过 |
| G0-10 | VRAM/运行时 | Full Ray Grid 与 K4 多候选无 OOM；测分段耗时和峰值显存 |
| G0-11 | Planner G3 不变 | S10 Scoring、n_win=5、距离加权、动作约束、真实 P1 更新与原 V4 一致 |

`G0_VALIDATION.md` 必须逐项提供运行命令、输入大小、输出 shape、diff/误差、失败修复和 Git SHA。不能只是勾选“已通过”。

## 7. 快速原型 Gate 与正式训练

- 第一个短程可执行 smoke：10–100 batch 确认 loss、GPU、bbox、surface positives；然后训练 Phase A Bridge。
- 记录 Train-6 / 合法 S-c（如果独立）验证的 FM、ES/Brier、Surface Loss、first/second-hit recall；**不要以 Test-3/MP3D 闭环挑选训练步数或 Token 半径**。
- 本次用户希望加快进度，故建议首次采用**预注册固定末步 EMA** 为正式结果；中间 Checkpoints 可画学习曲线，但不能事后选 Test-3 最佳点。
- 若短程 Bridge 无法在训练多表面正例上拟合简单例子，先修复 label/encoding/gradient，而不是直接训练10k并称成功。
- 如果训练收敛但性能不佳，正式报告应保留负结果，不能偷偷修改 Planner、评分密度或 Flow 架构补分。

## 8. G1 / G2 / G3：输出格式

### G1：未来观测与多表面预测

1. 旧协议：Known、Unknown、Truly-New Depth MAE、HIT Brier、Fair ES、FM validation、K=4 diversity/误差校准。
2. V5 额外：`Surface Belief` 对 GT 多次相交的 Ray（**来自该域评价端 GT，不进入模型**）分别报告 first/second+ surfaces 的 Recall、Precision、Brier/ECE、Ray-scale buckets、门/走廊/遮挡可视化。无可信 eval GT 时标注无法评价，不能填假数。
3. 显示原 V4 `H_old`、新 `H_new` 特征统计和 Flow 生成速度；对比时同样候选状态/P1/随机种子/Mask。

### G2：固定状态 Camera Selection

- 使用已有 365 个 B2 replay states（MP3D 90、Mac++ Dev-3 275），或实际冻结协议规定的等效固定 pool；候选与 GT Gain 一致。
- **主要指标**：真正被选 Camera 的一步 Coverage Gain（pp）、Top-1、早/中/晚状态分组、选择与 V4 不同的比例。Regret 仅附录。
- 不能因为 `b_r(t)` 预测的墙后有更多表面就改变 DistP1 的可见表面收益定义。本轮主比较 V5 与 V4 都是 `D100 model / S10 score`。

### G3：闭环

| 数据集 | 评估集合 | 主指标 | 用户批准状态 |
|---|---|---|---|
| Macarons++ | 6 场景 × 5 起点 × 100 步 | AUC@100 / Final / Cov@20，逐场景 paired CI | 已在旧实验反复使用，**非盲** |
| MP3D | 5 eval 楼层 × 2 起点 × 150 步 | AUC@150 / Final / Cov@20，停滞，逐楼层 paired CI | 只 Zero-Shot，不用于训练/调参 |

统一对照数字须从仓库实际归档读取；已知历史核对表：

| Model | MP3D AUC / Final / Cov20 | Mac++ AUC / Final / Cov20 |
|---|---|---|
| MAG-G | 0.278 / 0.377 / 0.169 | 0.704 / 0.905 / 0.507 |
| MAG-B | 0.292 / 0.377 / 0.165 | 0.685 / 0.891 / 0.504 |
| V2 Original | 0.315 / 0.441 / 0.179 | 0.724 / 0.898 / 0.578 |
| B0 | 0.322 / 0.469 / 0.195 | 0.729 / 0.884 / 0.622 |
| B2 | 0.334 / 0.530 / 0.197 | 0.730 / 0.906 / 0.572 |
| E2-D100 | 0.364 / 0.553 / 0.195 | 0.711 / 0.880 / 0.570 |
| V3-T0 / T1 | 0.311/.463/.183 ; 0.331/.508/.184 | 0.702/.872/.568 ; 0.712/.883/.566 |
| V3.1 | 0.335 / 0.499 / 0.185 | 0.697 / 0.865 / 0.551 |
| **V4-A0 S10** | **0.327 / 0.495 / 0.187** | **0.710 / 0.878 / 0.556** |
| V4-A1 D100Score (仅参考) | 0.345 / 0.503 / 0.200 | 0.666 / 0.841 / 0.528 |
| **V4-Continue12k** | 待运行 | 待运行 |
| **V5-Warm12k** | 待运行 | 待运行 |

**证据要求**：不能仅依靠 G1 Surface 质量推断 G3 提升；不能把 Mixed、60k+FT、V4 40k 和 V5 40k+12k 的历史差异解释成单模块因果。输出每场景/楼层结果、配对 Bootstrap CI、胜负数、Coverage Curve、推理时延与 GPU 成本。

## 9. 运行管理与可靠性

- 在真实 Slurm 可用资源内申请、记录作业 ID、GPU 类型、CPU/Memory、GPU 数、Walltime。训练 2k/10k 的开销**未知**，新 Query 可能比旧 Ray Encoder 慢；先 profiler，不预言总 GPU-h。
- 所有大型读写进 `work_dirs/exp6_v5/` 或可合法使用的 scratch/local cache；避免高并发 NFS 小文件读取。Mesh Labels 应避免每个 batch 重复 raycast。
- 保持所有训练状态/Checkpoint 可 Resume。Login 节点维护及 `salloc` 断开会导致作业结束，尽量采用能恢复的 sbatch 形式；不擅自破坏既有 GPU 分配。
- 如果训练遇到数值错误、求交标签不一致、Mask 全无效、Flow 条件维度不兼容，记录并修复相应原因；不要偷偷改模型目标/Planner 后直接续跑却不写 `V5_DEVIATIONS.md`。
- 所有本轮改动仅在可追溯分支/commits；保留原 V4 Code、Config、Checkpoint 与实验报告不可覆盖。

## 10. 最终应交付

1. **架构实现**：V5 模型可独立推理，`3D U-Net + adaptive Ray-to-Memory + Multi-Surface Belief + original 2D Flow`，不再在 inference 中调用旧 32 3D Feature Samples。
2. **标签系统**：训练期多表面 GT 缓存、场景 Mesh 来源、遮挡后多个交点正确性和 mask、无 GT 泄漏证明。
3. **训练证据**：Train-6 Outdoor-only Warm-start 2k+10k（或审计后冻结的等价配方）训练日志、EMA、Loss/Belief/Flow 曲线、OOM/VRAM/成本；V4 continued-budget 对照若资源允许同步完成。
4. **完整 G0/G1/G2/G3**：MP3D Zero-Shot 与 Outdoor 双域表、Coverage Curves、场景配对 CI、运行命令/种子、脚本和模型权重哈希。
5. **科学结论**：V5 和原 V4/B2/E2 在不同指标上的改善/退化；是否改善 Unknown-Ray、Second-Hit Geometry、真实下一 Camera、长期 G3；明确哪个结果仅为性能对照、哪个可归因。
6. **交接摘要**：`STATUS.md`、`FINAL_REPORT.md`、可复现命令、提交 SHA、checkpoint 路径、未解决风险和下一步可选实验，但**不要擅自开始其他架构训练或 Planner 改造**。

### 完成判定和停止条件

- 若标签 Gate 不满足 → 停止正式 Multi-Surface 声称，提交阻塞审计和可用代码。
- 若 G0 有泄漏/形状/梯度/旧 Sampler 调用/无法复现 → 修复后再正式训练。
- 若训练或 G3 资源不足 → 提交已完成真实结果 + 精确未完成项，不得填虚假结果。
- 若完整执行 → 提交真实报告，**本轮停止在 V5 双域结果**；Indoor Fine-tuning、Exploration Memory 与新 Planner 均由用户另行决定。

---

**向用户汇报格式**（先给最重要表，再解释）：

1. 审计/标签与 Warm-start 是否成功；是否真的移除旧 32-point sampling。
2. V4@40k / V4-Continue12k / V5-Warm12k 的双域 G3 **AUC、Final、Cov@20**，加 B2、E2、MAG-G/MAG-B。
3. Multi-Surface GT 和 Second-Hit 指标，First-Hit Depth/HIT 指标、G2 真实选中 Gain，给出“几何变好但 G3 不变”的证据边界。
4. 训练与推理成本、可复现路径、Git commits、偏差、结论。不要仅汇报 Loss、Regret 或一张可视化。
