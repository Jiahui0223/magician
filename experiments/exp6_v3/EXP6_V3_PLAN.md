# EXP6 V3 — 工程实施、从零训练与统一验收计划

> 版本：v1.0（交给代码 AI 的执行规范；本文件不是已执行结果）  
> 关联：仓库 `plan.md`、`idea.md`、`experiments/README.md`、`EXP6_V3_FROM_SCRATCH_PLAN.md`、`EXP6_V3_REPORT_TEMPLATE.md`、历史 EXP5/E2 代码与报告。  
> 执行模式：**先完成 Phase 1（V2 归档提交与 GitHub Push），提供简短阶段反馈，然后自动推进 Phase 2–5（V3 建设、检查与训练）**。必要的参数和工程方案允许 AI 根据实测优化，但不得破坏实验公平性、安全边界或擅自修改已冻结测试协议。

## 0. 优先级与计划兼容性

1. **最新明确指令优先**：用户本次明确要求提交、Push V2，并新建 `experiments/exp6_v3/`，覆盖旧 `plan.md` 中“暂不提交、不上传”的过时限制，但**仅限本次授权的 V2 归档及后续按仓库约定的 V3 工作**。不允许借此推送无关或敏感文件。
2. `plan.md` 中实验目录使用 `expN/taskM`、`idea.md` 讲研究思路、`experiments/README.md` 讲实验规则、完整复现记录、保留原始结果等工程习惯继续遵守。旧文档描述的 HDSAM/3D-FRONT/Replica/`exp6_replica` 属于另一阶段历史方案，**不直接迁移为当前 MP3D+Macarons++ V3 训练协议**。如果编号冲突，用 `exp6_v3` 独立命名，不改写旧 `exp6_replica`。
3. 操作前实际阅读仓库内 `plan.md`、`github.md`（若存在）、README、现有 EXP5/V2 结构，记录实际文件/分支/远端。若计划有新版本或存在当前不可兼容的硬约束，记录在 `reports/DEVIATIONS.md`，采取最小变更，不默默覆盖。
4. **禁止伪造已运行结果**：本文给出的路径、超参数和资源均为拟议规范；实际环境由代码 AI 核实。所有训练结论仅来自真实日志与评估。

## 1. 全部任务路线（执行顺序）

| 阶段 | 主任务 | 可否自动继续 | 交付/放行条件 |
|---|---|---|---|
| **Phase 1 · V2 Freeze & Push** | 整理当前 V2/EXP5/E2/G3 的代码、配置、报告并规范 commit、push | 应先独立完成，**必须提供简短阶段反馈** | 远端 branch 有对应 commit，记录 commit SHA、push 成功信息；无意外变更 |
| Phase 2 · EXP6 初始化 | 新建 `experiments/exp6_v3/`、`task0_vae/`、`task1_xattn/`，锁定研究协议和目录 | 是 | 目录清晰；数据拆分与模型设计记录齐全；不修改旧 EXP5 结果 |
| Phase 3 · G0 工程审计和代码 | Raw partial TSDF 输入、Simple VAE、Ray/Flow 联训、X-Attn 两版本和测试 | 是，测试通过后推进 | 单测、集成、Tiny Overfit、DDP smoke、物理尺度检查通过 |
| Phase 4 · Scratch Training | T0：V3-VAE；T1：V3-VAE+XAttn，独立从零训练 | 是，可两组各 4 GPU 并行 | 初始化公平、训练稳定、checkpoints/配置/日志齐全 |
| Phase 5 · Evaluation & Reports | G1/G2/G3 + task0/task1 独立报告 + V3 综合报告 | 是，按预冻结门槛和预算运行 | 数字、逐场景结果、图、负面结果、资源和偏离记录完整 |

执行期间不要为了小问题不断等待用户确认；只在**关键数据泄漏、无法安全提交、测试划分冲突、非法资源占用、关键协议无法遵守或超过明确算力上限**时暂停并报告。其余问题允许根据单测、profile 与开发集分析动态修复，必须记录依据和代价。

---

## 2. Phase 1（首要任务）：V2 整理、提交与 GitHub Push

### 2.1 先核实，不直接提交

1. `git status --short --branch`、`git branch -vv`、`git remote -v`、`git log -5 --oneline`、`git diff --stat`。核对分支、远端、未跟踪文件、当前代码归属；不得假设当前就是 `main`。
2. 盘点 V2/EXP5 已完成内容：V2-Mix 架构、训练与预测、P1/B0/B1/B2、Task 3 密度消融、E2 G1/G2/G3、configs、plots、reports、测试与已有复现实验入口。归档**现有完成的真实文件**，不要替用户虚构、补写实验结果。
3. 根据仓库现有目录组织最小必要文件，修正明显失效的路径/导入/README 入口，但**不借归档改动算法、checkpoint、数据划分、指标或历史结果**。非必要的大规模重构等到 V3。
4. `git diff --check`、对应 V2 现有单测与最小 import/smoke；审查是否含绝对路径、账号、令牌、密钥、个人数据、超大缓存、dataset、private weights。
5. 仅暂存需提交文件：优先 `git add <逐项核实路径>`，不要未经检查就 `git add -A`。保留无关工作区改动，不要用 `git reset --hard`、`git clean -fd` 清理。

### 2.2 Git 约定（严格采用用户给定格式）

**提交标题格式**：`[V2][<category>] <English-or-Chinese concise description>`。第二个 `[]` 是简洁英文分类，不得留空。建议分类：`archive`、`docs`、`fix`、`test`、`exp5`、`e2`。示例：

```text
[V2][archive] Freeze V2-Mix and completed EXP5/E2 results
[V2][docs] Index V2 experiments and reproduction protocol
[V2][test] Add V2 baseline regression checks
```

V3 若创建后续提交，统一采用：

```text
[V3][setup] Initialize EXP6 V3 experiment structure
[V3][task0] Implement scratch 3D VAE with masked TSDF reconstruction
[V3][task1] Add camera-ray cross-attention
[V3][eval] Add G1 G2 G3 evaluation and reports
```

- 遵守 `github.md` 中除旧“禁止提交/推送”外仍适用的要求；**不要添加 Claude Co-Authored-By 或其他自动署名 footer**。
- 如已有合法可写 remote/tracking branch，按仓库权限使用 `git push -u <verified-remote> <verified-branch>`；**不强制 push**、不改写公共历史、不擅自推送到错误项目、不新建未经批准的 PR。
- 仅在确实需要、且仓库有现成约定时创建 release tag；不要仅因方便就新建/移动 `v2` 标签。
- Push 后再次 `git status -sb`、`git rev-parse HEAD`，对比远端 commit（如 `git ls-remote`）；确认 push 已被远端接收。
- 若 GitHub 权限/网络/冲突导致失败，明确停止 Phase 1 的“已完成”声明，报告精确 blocker；**不得声称 Push 成功**。没有远端权限时可完成本地整理但标记 `push blocked`，避免在未经归档时悄然开始 V3 大改。

### 2.3 必须向用户发送简短阶段反馈（完成 Phase 1 当下）

使用 4–7 行简洁回复：

```text
[V2] 已完成归档并推送。
Branch/Remote: <已核实信息>
Commit: <完整 SHA> ；Commit subject: [V2][archive] ...
检查：<真实通过的测试和数量>；已整理：<主要目录/报告>
未提交的大文件/注意事项：<如有>
下一步：将自动创建 experiments/exp6_v3/ 并执行 task0/task1，无需再次等待批准。
```

如果 Push 失败，用 `[V2] Push blocked` 替代成功句，并给出可执行的解法。这个阶段反馈**不能省略**，且发送反馈后继续推进不需要再次询问用户（仅当 Phase 1 真正完成时）。

---

## 3. Phase 2：EXP6 V3 文件结构（必须结构化）

### 3.1 目录建议

尽量遵守现有 V2 的模块化风格。具体包名和 import path 先审计 repo 再最终确定；下面是目标层级，其中 `report/` 为**每个子任务独立报告目录**。

```text
repo/
├── plan.md                                 # 旧项目总计划，保护原文
├── idea.md                                 # 追加/更新 V3 研究概述（勿覆盖其他研究）
├── experiments/
│   ├── README.md                           # EXP6 新增索引，不删除历史章节
│   ├── exp5_future_view/                   # V2 冻结，不改历史产物
│   └── exp6_v3/
│       ├── README.md                       # 实验目标、运行导航、状态
│       ├── protocol.md                     # 开跑前冻结，含数据/损失/评估/选择规则
│       ├── STATUS.md                       # 进度与 checkpoint / job 表
│       ├── configs/
│       │   └── common.yaml                 # 两个模型共享的可复现设置
│       ├── shared/                         # 公共数据/P1/Ray/Flow/metrics 接口，避免复制
│       ├── tests/                          # 跨任务公平性、数据一致性与集成测试
│       ├── task0_vae/
│       │   ├── README.md                   # 模型结构与任务接口
│       │   ├── configs/                    # VAE + Flow scratch 训练配置
│       │   ├── scripts/                    # 构建/训练/评估/提交入口
│       │   ├── tests/                      # VAE/TSDF/KL/梯度与端到端测试
│       │   └── report/                     # **独立** G0/G1/G2/G3/训练报告
│       │       ├── T0_AUDIT.md
│       │       ├── T0_G0_TESTS.md
│       │       ├── T0_TRAINING.md
│       │       ├── T0_G12.md
│       │       └── T0_G3.md
│       ├── task1_xattn/
│       │   ├── README.md                   # Query/Key/Value 设计与形状
│       │   ├── configs/                    # XAttn + 同款 VAE/Flow scratch 配置
│       │   ├── scripts/                    # 构建/训练/评估/提交入口
│       │   ├── tests/                      # attention mask / ray-query / 一致性测试
│       │   └── report/                     # **独立** G0/G1/G2/G3/训练报告
│       │       ├── T1_AUDIT.md
│       │       ├── T1_G0_TESTS.md
│       │       ├── T1_TRAINING.md
│       │       ├── T1_G12.md
│       │       └── T1_G3.md
│       └── reports/
│           ├── V3_COMBINED_REPORT.md       # 两模型统一主报告，参照已给模板
│           ├── V3_COMPARISON_TABLES.md
│           ├── V3_DEVIATIONS.md
│           └── V3_RESOURCES.md
└── work_dirs/
    └── exp6_v3/
        ├── shared_cache/                   # 中间 P1 / patch cache，gitignore
        ├── task0_vae/{runs,checkpoints,eval,figures,logs}/
        └── task1_xattn/{runs,checkpoints,eval,figures,logs}/
```

上述目录中的 `{runs,...}` 表示五个真实子目录，不是要求 Shell 原样保存花括号文本。若现有项目采用 `src/` 的正式 Python 包进行模块管理，应将可复用的 VAE/Attention/Trainer 核心实现放在适当 Python 包下，并在上述 `task*` 路径放清晰入口、配置、测试、报告；**不得为满足目录而拷贝两份不同的训练器**。

### 3.2 文件与职责原则

- **架构层**：清晰区分 `partial_tsdf`、`vae`、`ray_geometry`、`ray_encoder`、`flow_unet`、`cross_attention`、`planner_scoring`；单模块独立、函数职责单一，函数名可解释；避免超大单体脚本、类中嵌套一堆局部 helper、冗余接口适配层。
- **Experiment 层**：T0/T1 配置分开但公共参数同源；`train.py` 和 `eval.py` 调用公共训练/评估代码，不复制既有 Task 2 训练循环并改名冒充 V3。
- **输出层**：只将可复现的配置、脚本、报告、较小图表、日志摘要纳入 Git；重 checkpoint、数据、缓存、原始大规模日志放 `work_dirs` 等既有忽略目录。保存 run manifest 指向实际文件。
- **文档索引**：更新 `experiments/README.md` 和 V3 局部 README。必要时给 `idea.md` 添加一节 V3 架构动机，但不要将旧 HDSAM 正文全部替换。
- **每个任务必须独立可运行**：Task0 不依赖 Task1 的 imports；Task1 从共享组件与 Task0 的模型**代码**复用架构，不依赖 Task0 训练好的 weights。

---

## 4. 冻结的架构与数据要求（基于既有 V3 from-scratch 方案）

### 4.1 数据集与训练记录

- 仍使用 **MP3D + Macarons++**；本轮只用合法训练划分做 Mixed Training，不做 Indoor-only/Outdoor-only 完整消融。
- 既有 E2 审计中的 MP3D 228 trajectories/263,358 candidate views、Macarons++ 138 trajectories/98,887 candidate views，作为审计基准而不是硬编码训练长度。将 scene/house、trajectory、time t、candidate a 记录和 split 固化为 manifest。
- `trajectory` 是某一场景中按时间顺序执行真实相机观测得到的序列 `[(C0,D0),...,(Ct,Dt)]`；一个训练样本来自 `(scene_id,traj_id,t,candidate_id)`；候选未来真实 depth 只做监督，不得写入当前输入地图。
- Scene-level/house-level 划分，确认 MP3D val/test 与 train disjoint、Macarons++ train/dev/test 场景关系；保留原 benchmark 的 *seen-scene trajectory vs unseen-scene* 区别。未知 split 状态 STOP；不得凭文件名猜测。
- 相同样本索引流与 Ray/Flow 噪声控制供 T0/T1 公平对照。注意数据域 batch 32/32 并不等于梯度严格 50/50；报告实际有效监督射线与 domain losses。

### 4.2 真实 Partial TSDF 与空间一致性

- Scene Encoder 使用历史真实观测融合的 **100% BAND P1**（EMPTY carving 维持已有定义），不使用候选未来 GT、整场景 mesh、VAE 推测深度更新 P1。
- 保留真实 signed distance、观测状态 `BAND/EMPTY/UNKNOWN`、空间有效掩码、voxel size、truncation、origin/world coordinates；UNKNOWN sentinel 不得当作物理负 TSDF。推荐输入 `[valid_sdf, band_mask, free_mask]`，另提供 spatial-valid mask；以实际 P1 实现语义为准。
- 全局 P1 无法直接作为巨大 dense 3D Tensor，应使用**显式记录尺度与世界坐标的状态感知降采样和确定性分块**；Patch/stride/overlap 由 profile 后锁定，所有模型完全相同。
- 必须测试 MP3D 与 Macarons++ 的物理尺度换算：室外 scene units 是否是 meters、TSDF truncation 真实值、相机外参 convention、grid-origin、三维索引顺序。检查 tile stitching 与同一 world point 两侧的 ray feature 数值一致性；必要时保存 3D 可视化。

### 4.3 Task0：Single 3D VAE Encoder + Flow

- 只有一个 3D Scene Encoder：Simple 3D Convolutional VAE（从零训练）；`mu/logvar` 为 standard VAE posterior；Decoder 只在训练时重建当前 Partial TSDF/观测状态，不在线补全完整场景。
- Train: `z_rec = mu + exp(0.5*logvar)*epsilon`，Decoder reconstruct；Flow condition: `z_flow = mu`，供已有 Ray Encoder + Rectified Flow；test 也用 `mu`，保证推理稳定。
- 总损失：`L = L_FM + lambda_rec * L_observed_reconstruction + beta(step) * KL`，`L_recon` 至少包含 observation class CE + band-valid SDF masked L1（可以同一 decoder 输出）；UNKNOWN 不作为真实空闲距离监督。KL 按有效 latent 数和通道数归一；记录 active units/ posterior collapse。FM 保留原 V2/E2 的有效射线与标签定义。
- 不添加旧 3D U-Net 并联分支，不加 Adapter、Gate、额外语义预测头、新 Scoring Model。
- 允许基于 G0 中真实的内存/梯度/重建检验对 VAE 通道、patch 规模和 lambda/beta 范围做**有限初始校准**，但在两组正式训练前冻结公共协议；不能按 test AUC 事后调。

### 4.4 Task1：同款 VAE + 单处 Camera-Ray Cross-Attention

- T1 使用与 T0 **相同的 VAE、Decoder、Ray Encoder、Flow 主干和训练损失**，只在 Flow U-Net 低分辨率层插入一处 Cross-Attention。
- `Q` 必须随当前候选相机的 pose/rays 与 Flow feature 改变；`K/V` 是真实历史 VAE `mu` 空间 tokens，含合法 3D world coordinates 和 observed/valid masks。Attention 不把 UNKNOWN token 作为已确认表面。
- 使用有依据的局部/frustum token selection 或紧凑 scene memory，不允许全图无限制 attention；注意 KV cache 跨 candidate 和 Euler steps 的正确性、grad preservation（训练时不可误 detach）。
- 不新增学习式融合 head / gate，不修改 DistP1。已有 Self-Attention 保持。

### 4.5 Scratch 训练公平性

- 两组**独立从头**：不加载 V2、E2、Seen2Scene 预训练权重，也**不从已经训练的 T0 权重开始训练 T1**。
- 尽量使 T0/T1 共有层的初始参数完全一致（显式记录并 hash 比较），新 XAttn 层独立初始化；初始化器、采样顺序、优化步数、effective batch、优化器/scheduler、EMA、损失权重、训练/验证 split 保持一致。新参数数目变化需记录。
- 训练时间不直接沿用 E2 5000-step fine-tune。现有文档以约 60k optimizer steps 为**待实测确认的预算上限候选**；具体步数/LR/gradient accumulation、保存点/early stop 由 G0 profile + dev 预定义规则确定，锁定后两组相同，不能单独给 T1 更多训练时长。
- T0/T1 必须能 resume；记录 `git SHA / dirty diff / configs hash / data manifest hash / seed / CUDA/PyTorch / GPU / SLURM job ID`。checkpoint 原子保存，训练终止/中断后恢复不丢 optimizer/EMA/RNG/scheduler。
- 同一 P1 history 的候选视角应共享几何编码；避免每一个 candidate 重跑 3D VAE。训练时即使缓存 encoder activation，也要正确保留可微梯度，不能以 inference-style detached cache 破坏端到端训练。

---

## 5. G0 小规模测试 → 可以自主优化的范围

### 5.1 强制自动执行的最小测试

**代码静态检查**：语法、imports、format/lint（仓库已有命令为准）、`git diff --check`、依赖版本固定；无未使用的模块和旧实验绝对路径。

**数据测试**：
1. 时间/候选泄漏：构建 `P1_t` 时只允许历史 `0..t`，随意替换未来 GT 不影响 `P1_t`；完全等价的 sparse historical builder 需要复现冻结 10% 评分地图。
2. TSDF 三态：合成已知墙面/自由区/未知区，验证 TSDF 数值、mask、truncation、有效区外 padding；UNKNOWN 不能被监督成 EMPTY。
3. 空间对齐：不同 scene unit、坐标系、外参、voxel origin、tile 拼接、ray first-hit 与三维采样点；建立至少一个 MP3D 和一个 Macarons++ 真实状态的对照检查。
4. Scene split：相同 Scene/House 不得跨 train/dev/test 未经标记；候选 target 仅在 loss/eval 使用。

**模型测试**：
5. `VAE.encode()` 返回 `mu, logvar`，形状、浮点范围、有效 mask 正确；`KL>=0`（允许数值微小误差）、mask 归一稳定；梯度分别从 `FM`、`Recon`、`KL` 到 VAE Encoder，特别确认 logvar 梯度通过 reparameterization 和 KL，Flow 只经过 mu 是设计预期。
6. 小 batch：同一输入复现推理结果；不同候选 pose/ray 查询应该导致不同条件；K=4 Flow 输出范围、HIT decode、梯度和 NaN/Inf 检查。
7. XAttn：Q/K/V shape、invalid token mask、不同 pose query、局部 token selection 的坐标正确、无 key/value 跨场景混用；加入 XAttn 不改变 target/scorer。
8. 单卡 1–200 step Tiny Overfit：FM/Recon 各至少能学习到非平凡信号；训练前后记录有效监督射线、active latent units、diversity、optimizer gradients。
9. DDP 4 GPU smoke：每卡有效 batch、全局 loss reduction 与 1 GPU 对照、BF16 数值、OOM 处理、checkpoint resume 与重现性；必要时跑 8 卡环境可用性测试，但 T0/T1 可以分别各使用 4 卡。
10. 100–300 step profile：`sec/step`、显存、CPU 内存、磁盘容量、DataLoader throughput、VAE patch encoding cache、CrossAttn candidate latency，估算全程 GPU-hours。

### 5.2 AI 自主权（有限、可追溯）

可以自主修复 import、shape、mask、dtype、训练不收敛的显然数值错误；自主选择**经 profile 支持的** patch size、downsampling、token budget、gradient accumulation、num_workers、checkpoint 间隔、BF16/activation checkpoint，或在正式训练前小范围校准 `KL warmup、lambda_rec、beta、LR`。每项应记入审计报告：问题 → 候选解 → 小测试指标 → 最终冻结值。

如果已经开始正式训练，**不允许**悄悄修改 architecture、随机种子、loss 定义、domain batch ratio、有效监督掩码、eval protocol；若必须修复重大 bug，先停止两组受影响运行，记录版本/step、废弃或从相同状态重启，保证 T0/T1 公平。不得因为某个开发场景性能差就无限调参；不得接触 test 选择模型。

**硬停止/汇报**：Push 不成功、敏感内容准备被提交、历史 P1 未来泄漏、train/test overlap、无合法 GPU 资源、错误的场景物理尺度未修复、无法通过关键单测、不可恢复的 OOM、非有限 loss、显式预算上限被超出。遇到 hard stop 写 blocker 并请求具体决策；不假装继续成功。

---

## 6. GPU / Slurm 编排（授权至多 8 张卡，按实际排队与配额执行）

1. **先审计现有 4 张 GPU**：确认是哪个 allocation/job ID、节点、型号、剩余 walltime、是否由本用户持有；使用 `squeue -u "$USER"`、`scontrol show job <id>`、`sinfo -o ...` 与 `nvidia-smi`。**不能因曾经用过 4 张卡就假设仍可用**。集群禁止直接 SSH 计算节点时用已有合法 `srun --jobid`/`salloc`/`sbatch` 接入。
2. **新申请另外 4 张**：按集群规则编写 `sbatch` 脚本（建议 `--gres=gpu:L40:4` 或集群实际可用 `L40S` 名称，先核对 `sinfo`），按需设 CPU/RAM/walltime；不猜分区、账户或路径。可以在现有 4 GPU 资源上跑 T0，新增 4 GPU 跑 T1。
3. **并行条件**：两份代码都已通过 G0、共同配置和共同初始化已锁定、GPU/节点实际可用、共享数据 IO 不构成瓶颈。满足时可让 **T0 4 GPUs + T1 4 GPUs 并行**以充分利用 8 卡；否则 T0 与 T1 顺序跑。即使并行，T1 也不能读取 T0 的训练后权重。
4. **DDP 与作业治理**：每个实验独立 job ID、端口、run directory、seeds、logs、checkpoint；避免两个作业写到同一路径。使用 Slurm `--output`/`--error`、定时 checkpoint、preemption/resume，保存 job IDs 与启动参数到 `STATUS.md`。
5. **失败恢复**：`sbatch` 返回成功只代表 job 已提交，不代表已开始/已成功训练；检查 `squeue`、`sacct`、stdout、最后 checkpoint；OOM 优先检查模型缓存、patch、batch/accumulation，一次小测试后再恢复；不能无限反复申请抢占资源或超过授权预算。
6. **资源上限**：最多使用用户当前明确授权的 8 卡，不自行申请更多卡、不可擅自长期占用额外节点；训练总步数以已经冻结且合理的预算为准，超出上限需说明并请求用户许可。保留逐组 GPU-hours、实际 wall-clock、峰值显存、存储耗用。

---

## 7. 评价协议与两级报告

### 7.1 冻结指标

- **G0**：P1 三态与分块编码、物理坐标一致性、历史几何 BAND/EMPTY/UNKNOWN 重建、TSDF SDF masked MAE、thin-structure、KL 与 active units、显存和速度。
- **G1**：Future Depth MAE，Known/Unknown、`Coverage-new@10%` 与 `Physically-unseen@100%` 分开；Brier / HIT / ES、K=4 Diversity 与 uncertainty-error correlation；MP3D/Macarons++ 独立。
- **G2**：相同 fixed-state 的 Early/Middle/Late Oracle Regret、Top-1 Accuracy、Spearman、Top2 Score Margin、逐场景统计与 paired CI；DistP1 scoring 仍使用 10% Sparse P1。
- **G3**：锁定 MP3D 和 Macarons++ 现有闭环场景、候选动作、评估步数，报告 AUC、Final、Cov@20/中途 checkpoints、逐 episode/scene、movement/revisit、latency、GPU peak，给出曲线和置信区间。两个模型都应有闭环记录；若协议门槛不通过仍跑 G3，明确 exploratory 与预注册规则不同。
- **对照**：保留原 B0、B2、E2-D100 实测，不重复旧训练；重点 `T1 vs T0` 是同预算 scratch 的干净结构对照。若没有 AE-only 消融，不能独立证明 KL 带来的收益。

### 7.2 Task 级报告（每个 `report/`）

每个 `T{0,1}_AUDIT.md` 至少记录：代码/数据路径与 SHAs、环境、输入输出张量形状、物理尺度、具体实现变更、失败测试与修复。  
每个 `T{0,1}_G0_TESTS.md`：完整测试清单、命令、通过/失败、真实 smoke 图、速度与显存，以及为何选择最终 `patch/tokens/beta/lambda/LR`。  
每个 `T{0,1}_TRAINING.md`：总步数、全程 loss（FM/Recon/KL、Val）、EMA/初始化、checkpoint、异常、吞吐/GPU-hours，**训练到哪步、是否真正完成**。  
每个 `T{0,1}_G12.md`：完整 G1/G2 和每数据集/阶段的差异、配对不确定性、负面结果。  
每个 `T{0,1}_G3.md`：真实闭环协议、覆盖曲线与逐场景、失败、成本。

### 7.3 综合报告 `reports/V3_COMBINED_REPORT.md`

严格按照已提供 `EXP6_V3_REPORT_TEMPLATE.md` 章节扩展，并至少包含：研究问题、当前 V2 基线、数据与训练轨迹解释、`T0/T1` 架构图、损失公式、两组 scratch 的同初始化验证、G0–G3 统一主表、Indoor/Outdoor 场景差异、KL 与几何重建诊断、时间/显存、负面结果、与历史 baselines 的非同预算局限、协议偏离、最终科学结论。图像/表格须链接真实生成文件，不得依据预期虚构曲线。

**结果要与状态分开**：`Completed`、`Running`、`Failed`、`Not Run`、`Exploratory` 要准确标记；所有数字带日志路径与产生配置。不能在没有训练时写出“提升 X%”。

---

## 8. 验收 Checklist

- [ ] 已阅读并记录 `plan.md` 等现有规范；未覆盖历史 HDSAM/exp6_replica。
- [ ] **V2 归档检查完成，使用 `[V2][category] description` commit，成功 Push，并发送 4–7 行阶段反馈**。
- [ ] 远端 V2 commit SHA、branch、remote 可追踪；无大数据、敏感文件被 Push。
- [ ] 已建立 `experiments/exp6_v3/task0_vae/report/` 与 `task1_xattn/report/`，以及综合 `reports/`。
- [ ] 在正式训练前冻结 protocol、data manifests、共同模块初始化/损失/训练预算；附 config hashes。
- [ ] 只从历史真实 Depth 构建 Partial TSDF；测试无时间泄漏与空间错位；10% scoring 复现。
- [ ] VAE 的 `FM+Recon+KL` 三项梯度正常，posterior 不塌缩，Tiny Overfit 与 DDP smoke 成功。
- [ ] T1 的唯一结构变化为一处明确的 Camera-Ray XAttn；无多余 adapter/head/gate。
- [ ] T0/T1 独立 scratch，匹配初始公共参数/数据/优化预算；4+4 GPU 使用合法且可追溯。
- [ ] T0/T1 各自的 `report/` 实时记录，不会等全项目结束才补写。
- [ ] G0/G1/G2/G3 真实运行；完整独立场景划分、配对评估和负面结果。
- [ ] `V3_COMBINED_REPORT.md` 与统一主表齐全，工程与科学结论区分明确。
- [ ] 最终给用户简短文字总结：Git/V2 SHA；两个实验各自状态与最佳 checkpoint；室内/室外核心 AUC/Final；异常与下步建议。

## 9. 与已有方案的关系

- `EXP6_V3_FROM_SCRATCH_PLAN.md`：保持为**模型与数据科学定义文档**；本文件细化 Git 归档、任务目录、Slurm、测试、报告和自主执行。
- `EXP6_V3_REPORT_TEMPLATE.md`：综合主报告的基础模板，本文件补充 Task 级报告规范。
- `EXP6_V3_EXPERIMENT_DESIGN.md`：历史探索草案，部分内容已被更清晰的 scratch VAE 单编码器路线替代，**不要以旧的 Frozen VAE / Adapter / Gate 方案覆盖本文件**。
- `ablationstudy.md`：Indoor-only/Outdoor-only/Mixed 域消融的未来计划，**不加入本次两组主训练**。

> 执行代理在遇到真实代码与本文件的路径/数值不一致时，应以审计证据为基础记录差异，并在不改变科学核心约束的前提下做最小修正。用户允许小范围工程自主优化和训练自动衔接，但没有授权无边界地改研究目标、烧更多 GPU、改写 Git 历史或使用测试场景调参。
