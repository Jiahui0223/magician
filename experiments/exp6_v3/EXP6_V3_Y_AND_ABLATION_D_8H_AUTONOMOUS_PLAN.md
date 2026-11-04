# EXP6-V3：Y Fine-tuning + Ablation Study / Experiment D — 8-hour Autonomous Execution Plan

> **交给代码 AI / HPC 执行代理的正式工作指令**  
> 基础：2026-11-04 已完成的 `exp6_v3/H` H0–H3 结果，`FINAL_REPORT(6).md`，现有 CVPR Results/Table 1。  
> **执行意图**：立即开展两条可以独立运行、互不污染的研究任务；利用已经取得的 **8 张 L40 GPU**，为夜间约 8 小时建立连续训练、连续评估、监控与自动恢复的流程，并在截止前自动整理真实结果。  
> **重要**：本文件本身不代表 HPC 任务已经提交或运行。接收任务的代码 AI 必须实际检查集群、代码、权限、任务状态后才能启动。未核实的路径/资源/参数不得猜测。

---

## 0. 一页概览（必须首先阅读）

### 0.1 目标与已有证据

已有官方 Macarons++ S10 Coverage（14 个实际可运行场景，每场景 5 起点 × 100 步；`sofia_church` 尚缺）：

| 方法 | Dev-5 AUC@20 / AUC@100 / Final | All-14 AUC@20 / AUC@100 / Final |
|---|---|---|
| V3-T1 原始 H0 | 0.384 / 0.748 / 0.897 | 0.384 / 0.744 / 0.903 |
| V3-H3 | **0.444 / 0.759 / 0.875** | **0.424 / 0.762 / 0.901** |
| B0 | 0.398 / 0.733 / 0.880 | 0.403 / 0.748 / 0.897 |
| B2 | 0.384 / 0.732 / **0.898** | 0.388 / 0.742 / **0.907** |
| MAGICIAN Beam | 0.317 / 0.695 / 0.883 | 0.307 / 0.711 / 0.906 |

`FINAL_REPORT(6).md` 确认：H3 与 H2 仅差两步 lookahead；All-14 相对 H2 的 AUC@20/AUC@100/Final 分别为 **+0.023 / +0.018 / +0.012**，场景配对 bootstrap 95% CI 均不含零。H3 的主要痛点是 **Dev Final 0.875**（相对 H0 -0.022）和 `colosseum`（H0 Final 0.824，H3 0.671）、`pisa`（H3 Final 相对 H0 -0.051）。原 H2 的 `mean - λ·std` 风险惩罚可能误伤真实高收益未知区；深层未来动作的可达性仍需审计。

**核心研究目标（不是结果承诺）**：提高 Dev 与 All-14 AUC@20、AUC@100，同时尽量保持或提升 Final Coverage，重点修复 `colosseum` 退化；探索模型继续微调与更深搜索的互补性。不得宣称一定达到 All-14 Final 0.92 或 AUC 0.9；这些只能作为挑战目标。

### 0.2 两条工作线与唯一允许的目录

**Y：V3-T1 Fine-tuning**

- 代码/实验控制目录：`experiments/exp6_v3/Y/`
- 大型权重/运行目录：优先 `work_dirs/exp6_v3/Y/`（或审计后符合现有仓库的实际目录）。
- 从现有 **V3-T1 Mixed Scratch 40K** 冻结 checkpoint 复制启动；继续训练 **V3-T1 的 VAE + Scene Cross-Attention + Flow**，不改架构、不把 V3.1 Outdoor-only 当父模型。
- 分阶段保存并评估 45K、50K；如果运行速度和验证结果允许，可有一个 55K 的 optional 分支。使用实际已存在的父权重文件、完整 sha256 和真实 optimizer/EMA 状态。

**D：H3 Short-Horizon Planner Extensions**

- **不新建 Z 文件夹**。接到已有：`experiments/exp6_v3/ablation_study/`；在 A/B/C 所在层级新增 **`experiment_D_h3_lookahead/`**（如果现有命名风格不同，按现有风格保留 `experiment_D_` 前缀并记录最终路径）。
- 大型轨迹、缓存、结果放 `work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead/` 或现有标准根目录。
- 唯一主模型使用历史 V3-T1 **冻结模型**（完整 sha 校验），只修改 Planner/Scoring；不得把 Y 微调权重混入 D 单变量搜索。
- 历史 `H/` 的代码、权重、协议、结果必须保持只读，不覆盖 H3 官方 Table 1 行。

**最终组合（仅当两线各自已可靠评估、资源允许）**：用 Y 中表现最好的 checkpoint 与 D 中已锁定的最佳策略组合，创建独立模型 ID `V3-YD-COMBINED-*`；绝不把组合效果冒充 Y 或 D 单独的贡献。

### 0.3 用户允许夜间自主决策，但必须遵守这些边界

用户授权代码 AI：在当前 8 GPU、约 8 小时预算内，自动提交/调度已有授权计算资源、分配 GPU、监控、重启非致命失败作业、保存检查点、动态选择值得继续评估的变体以及自动写结果报告。**不需要在普通小问题上等待确认**。

不授权：删除/覆盖历史训练结果；越过已有集群额度；访问 GT 作为 Planner 特征；改动现有 MAGICIAN 官方 Coverage 主协议；手工改历史 Table 1 数字；变更 Test 的定义；将不完整结果填满或造数；擅自更改账户、申请额外付费资源、终止其他用户的任务。

如果出现影响科学结论的协议变动（例如 D100 改 S10、评估器变动、损失/模型结构改变），不能默默实施；可以建立单独标识的额外实验，但需在报告中指出其与主比较不能混排。

---

## 1. 开工前必须完成的审计（目标：快速且不可跳过）

### 1.1 找到真实目录与现有文件

以仓库真实内容为准，不假设下面名称一定存在：

```bash
pwd
find experiments/exp6_v3 -maxdepth 3 -type d | sort
find experiments -maxdepth 5 -iname '*cvpr*' -o -iname '*table1*'
find work_dirs -maxdepth 5 \( -iname '*v3*' -o -iname '*ema*' \) | head -100
```

审阅：仓库 `plan.md`、`experiments/exp6_v3/` 下已有协议与训练入口、`ablation_study/experiment_A*/B*/C*` 目录、`experiments/exp6_v3/H/` 评分与 G3 入口、`experiments/cvpr_results/table1/`（真实路径若不同则以实际为准）及其 registry/validator/evaluator。现有历史 `FINAL_REPORT(6).md`、`TRAINING_COMPARISON.md` 可作事实参考，不能替代实际代码核验。

### 1.2 锁定父 checkpoint 与训练可恢复状态

历史 V3-T1 checkpoint 摘要 `639c9d3a…` **不是完整哈希**。必须定位真实文件并计算 SHA256。检查：

- V3-T1：VAE latent、RayEncoder、Scene XAttn、Flow U-Net 对应的模型类与张量形状。
- 完整权重、EMA、global step=40K、optimizer、scheduler、AMP GradScaler、RNG states、dataset sampler resume position 是否均存在。
- `strict=True` 或等价的完整 key/shape 校验能否加载；任何 skipped key / random init 都须报告。若 optimizer/scheduler 不能完整恢复，标注 `weights-only continuation`，不能宣称 bitwise resume。
- 训练时候的 mixed indoor/outdoor 数据实际分片、配比、数据密度和监督定义；绝对不要未经检查而把 B0 的“32+32 mixed S10”误认成 V3-T1 的训练配方。**优先复用 V3-T1 原版 D100 的 mixed recipe**，而不是不相干的 Outdoor-only/S10 续训。

### 1.3 锁定评估协议（两条线严格共用）

- Outdoor V3 历史：**D100** 用于真实 P1 / V3 几何输入，**S10** 用于新表面 band/filter 评分，官方 MAGICIAN **S10 Coverage** 才是主报告指标。
- R10 缓存刷新、K=4 stochastic Flow、10 Euler steps、64×114 Ray grid、相机/传感器/位姿、合法动作、起点、真实 P1 更新、100 步预算，全部读取并保存实际冻结配置。
- H3/H0 不读取 GT Mesh、未来真实深度、GT Oracle Gain、官方 Coverage 内部隐藏的未观测采样点。只有真实观测被执行后可以进入真实 P1；虚拟后续动作不得向 D100 P1 伪造真实观测。
- 官方 Coverage 与本地 monotonic Coverage 必须严格分列，禁止以额外单调化覆盖代替主指标。
- 主结果依旧是 Train-6/Dev-5/Test-3 → All-14（30/25/15 → 70 episodes）。`sofia_church` 若仍缺数据就标记 `All-15 PENDING`，不得当作 15 场景平均。
- 如果 Y 训练输入密度与当前主评估密度的差异不可避免，要在 provenance 解释，但不能改变正式评估口径以迎合某个 checkpoint。

### 1.4 GPU/Slurm 真实 allocation 先审计

当前用户表示 **8 张 GPU 已获批/已申请到**；执行代理必须确认究竟是**一个现存 8-GPU salloc**、**两个节点各 4 卡**还是**两个独立 allocation**。不要重复申请而造成排队/重叠。命令：

```bash
hostname
printf 'SLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${CUDA_VISIBLE_DEVICES:-}"
squeue -u "$USER" -o '%.18i %.16j %.2t %.10M %.12l %.6D %R'
scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | head -80 || true
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
```

- 有现存 allocation：用该 allocation 内经验证的 `srun` 作业步骤将 **4 GPU 给 Y，4 GPU 给 D**，注意节点边界、GPU binding、独立 CPU/内存；不另开与之竞争的 `sbatch`。**`srun` 参数必须先在该集群进行小作业测试**，不可照抄未验证的 `--exclusive/--overlap` 组合。
- 没有可复用 allocation 且有权限：用两个隔离的批处理作业或单个8卡批作业内并行进程；只申请用户已授权的资源，时限不超过剩余额度。
- GPU 可见性/隔离不正确立即停止启动，不允许用 CUDA_VISIBLE_DEVICES 猜测 GPU 编号（多节点可能都是 `0..3`）。每个子任务实际打印 `SLURM_JOB_ID`, `hostname`, `CUDA_VISIBLE_DEVICES`, `torch.cuda.device_count()` 和 GPU UUID。
- 压力测试：短时 DDP、D policy 1 episode、文件写入性能。避免把 8 卡全部用于 2 个冲突性 `torchrun --standalone` 使用相同端口；每个 process group 独立 master_port/rendezvous。

**完成审计输出**：`Y/PROVENANCE.md` 和 `experiment_D_h3_lookahead/PROVENANCE.md`，包含完整 checkpoint SHA、代码 commit、实际配置 hash、真实文件路径、allocation 记录和初始 walltime。

---

## 2. Y — V3-T1 持续 Fine-tuning（4 GPU，不中断）

### 2.1 研究问题

V3-T1 Mixed 40K 已有很好的 Dev AUC@100，但跨场景 Early/Final 不稳定。**用相同架构、优先沿用原训练数据管线继续优化，并检查更长训练是否提高生成/规划质量。** 不默认“容量大所以训久必然更好”。

### 2.2 Y 实验配置

- `Y0_PARENT_40K`：现有 V3-T1，严格只读。
- `Y1_CONT_45K`：在真实 40K 权重后继续 5K optimizer steps。
- `Y2_CONT_50K`：在 Y1 上再继续 5K。
- `Y3_CONT_55K_OPTIONAL`：只有队列与剩余 GPU 时间、训练进展及验证支持时才执行。
- **不要每个 snapshot 都单独从 40K 开始训练 5K/10K**；必须形成一条连续轨迹 `40K→45K→50K`，Y3 可续到 55K。
- 保持 VAE、Scene Cross-Attention、Ray Encoder、Flow，维度/损失权重/数据构成优先与历史 V3-T1 一致。仅在验证不稳定时可以尝试有记录的保守 LR 续训；任何学习率变化必须在 Y 协议文件中预先确定，不以 Test 结果临时调参。
- 如果历史 optimizer 状态不可用：以“新 optimizer + inherited weights”单独命名，并在结果里避免与严格连续训练混淆。

### 2.3 自动保存与不中断关键点

- **最好每 1000 optimizer steps、或每 15–20 分钟（按实际速度）** durable 保存最新 EMA/online weights/optimizer/scheduler/RNG/sampler 状态；45K/50K 的里程碑 Checkpoint 永久保存。一次写入先写同目录临时文件，fsync/rename 或等价原子提交；严禁出现半写入被自动恢复读取。
- 最多滚动保留 3 个有效 recent Checkpoint + 全部里程碑 Checkpoint；写 `checkpoint_manifest.json`（step、hash、文件大小、EMA、config、时间、完整性）。不得自动删历史权重或用户已有 checkpoint。
- 每 N 步记录 train `L_FM`、VAE reconstruction/KL、gradient norm、LR、hit/no-hit 分桶、UNKNOWN-hit MAE、GPU memory 和 `examples/s`；Validation 只读、固定样本/固定随机种子，不用 Test 来决定训练步数。
- 训练过程中遇到 NaN/Inf、OOM、CUDA transient、SLURM `PREEMPTED/TIMEOUT/NODE_FAIL`、数据读写错误时走 §5 的故障恢复路径。不能无限自动重试；禁止默默降低图像分辨率、K、输入密度或改变模型结构来“让它跑起来”。
- 训练 checkpoint 优选应考虑 **Dev G3**，不要因为 train loss 更低就直接宣布更好。若没有余裕跑全部 Dev-5，可以使用预先固定的 Dev-3 smoke 作为初筛，但最终 Dev-5/All14 必须独立完整评估，清楚标注筛选依据。

### 2.4 Y 的 G3 比较必须拆分 Planner 因素

1. 父 V3-T1 + H0：复用已经验证完全一致的旧官方记录。
2. Y1/Y2 + H0：对比训练自身对 Camera Decision 的贡献，不变 Planner。
3. 仅选中 **一个** Y 最有希望的 checkpoint + 原 H3：测“同一 Planner 下 Fine-tuning 收益”，独立模型 ID `V3-YBEST-H3`。不能把它写成 Planner D 的性能。
4. 最后才考虑 `YBEST + DBEST`，并记录交互效应。

模型评估必须登记完整 parent SHA、fine-tuned SHA、source 40K、extra steps、training sampler/seed、EMA、optimizer-state restored 与否。

---

## 3. Ablation Study / Experiment D — H3 延伸（4 GPU，优先算力用于规划）

### 3.1 命名与固定 H3

目录：`experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/`。与原 A/B/C 同级，包含 `FROZEN_PROTOCOL.md`、`configs/`、`scripts/`、`tests/`、`reports/`、`logs/`、`STATUS.md`。引用现有 H3 代码，不复制或改坏历史 H3/H2。

- `D0_H3_REF`: 历史 H3（depth 2，H2 风险项保留），只读参照。对应 All-14 AUC@20/AUC@100/Final = 0.424/0.762/0.901。
- `D1_H3_NO_RISK`: H3 depth 2，移除 H2 `mean − λ·std` 风险惩罚（λ=0），其余完全不变。它检验 H3 与 H1-like score 配合是否能修复 `colosseum` 或 Final。
- `D2_DEPTH3_BEAM3`: 从 **D1 的评分与其余配置** 出发，搜索深度变 3、Beam Width 3；每执行一步重新规划。
- `D3_DEPTH3_BEAM5`: 与 D2 完全相同，仅 Beam Width 3→5；不改变候选空间/评分/随机种子。
- `D4_DEPTH4_BEAM3_OPTIONAL`: 与 D2 同分数，深度 4、Width 3；只有预测速度、剩余计算和必要验证支持时运行。

**重要概念核验**：原 H3 已经枚举 depth-2 的第二层候选（平均约 31 个）；把“depth=2 的 width 从 3 改为 5”可能对第一步决策没意义。D2/D3 用 **depth 3** 才能真正比较 width。必须在 `SEARCH_AUDIT.md` 中写清楚：具体 layer expansion、保留 beam 的位置、公式、tie-breaking、是否完整 depth-2、实际节点数、预测次数、平均第二层复用比例。

### 3.2 不泄漏、保持物理一致的未来规划

- 每条 beam path 在**假想 novel-surface set** 上维护 3D coverage union，后续分支计算新增价值时去除重复覆盖；不能简单把每步 DistP1 之和直接当新 coverage。
- 与历史 H3 **保持 frozen-P1 open-loop multi-step proxy**：Future Depth 条件始终来自当前真实 D100 P1 的冻结 V3 编码；将每层生成结果并入**评分专用的虚拟集合**，不能修改真实 P1/伪造 D100 高密度观测后再输入 VAE。若要尝试逐步重新编码幻想的 P1，必须独立标记为更高风险的新 protocol，本轮不默认执行。
- Candidate legality：所有已知合法动作必须满足历史动作图/运动约束；未来分支不得使用 GT 碰撞、GT Mesh 或未来深度做合法性判断。如果未观测区域使动作可达性不确定，应在 score/日志中标注或依据已知状态保守处理，不得当作确定可达。
- 统一 `K4`、生成样本和候选 ID 的 deterministic seed 键（包括 scene/start/t/root/candidate/horizon/branch/sample），保证 D1/D2/D3 在相同节点上可配对复现；不要让不同 GPU 上的调度顺序决定噪声。
- 默认和历史 H3 继续使用同一几何去重尺度/coverage tolerance、S10 scoring、去访问规则、R10、真实 100-step 预算。不要在 D2/D3 随意改变 discount factor/最远可见距离、归一化或 reward 权重；如必须改变，开独立实验和单独 ID。
- Beam 重复状态与候选视角应缓存几何投影/未来生成结果，缓存必须加 P1-state hash+model hash+candidate ID+sample seed，不能跨不一致状态错误复用。
- 正常真实动作执行后，真实 Depth 更新真实 P1；重新计算下一时刻的合法候选与状态，不依赖前一时刻的幻想地图。

### 3.3 最值得诊断的两个退化场景

`colosseum`（H3 Final 0.671 vs H0 0.824）、`pisa`（H3 Final 相对 H0 -0.051）。前者早期看似更快，20–50 步后可能出现路线错过/局部饱和；H1/H2/H3 在 colosseum 上都显著退化，并非随机 fallback 造成，不能直接认定“原因就是 horizon 短”。

诊断输出：

- 按对应 5 个相同起点、相同时间 t，显示 H0、H3、D1、D2、D3 的 action/位置、Predicted Gain、实际 step Coverage Gain、总 Coverage、Novelty、回访比例、路径区域、预测 No-Hit 及候选可达/未知标志；数据只能在真实执行之后进行评价，Planner 不得读取 GT。
- 绘制 0–20 与 0–100 Coverage，以及 20–50 放大图；对所有场景逐起点输出 divergence first step 和后续 coverage plateau。
- 统计每个策略的虚拟去重比、候选节点数、early-step regret（固定状态可计算 GT oracle，**只能离线**）、回退比例、无效/重复子路径比例、每步 planning-time percentiles 和最大显存。
- 解释 Final 下降究竟与预测误差、未覆盖的远处区域、候选合法性、分支过度乐观还是 coverage 已接近饱和有关；没有证据就写“尚未归因”，不能编造结论。

### 3.4 选择优先级

1. `D1`：最便宜，直接验证 H2 风险惩罚是否妨碍 H3。
2. `D2`：Depth3 Width3，真正检验更长前瞻。
3. `D3`：Depth3 Width5，隔离宽度作用。
4. `D4`：Depth4 Width3，只要实测每 episode 可能占用过多 walltime 就放弃 **这个可选算例**；但 D1/D2/D3 中已产生的结果照实保存，不据此宣布其余结论失败。

不存在“G2 差就不许跑 G3”的规则。只要变体合法可运行，至少完成预先固定的完整 100-step Dev Pilot；最终选择主版本后优先跑满 Dev-5 和 All-14。

---

## 4. 8 小时 GPU 调度策略（按资源实时估计，不能保证固定时间）

> 所有时间为从“确认 allocation / 两条线开始运行”时的**相对预算**。如果真实剩余 SLURM walltime 不足，使用实际剩余时间缩减可选任务；至少预留退出/报告缓冲。不得因为已经午夜而假设系统 walltime 剩余 8 小时。

| 大致阶段 | Y 4 卡 | D 4 卡 | 调度决策 |
|---|---|---|---|
| 0:00–0:30 | 审计、DDP/恢复 smoke、启动持续训练 | 审计 H3 回归、搜索效率 smoke、`D1` 启动 | 两线确认真实 GPU 隔离、无污染，创建 run manifest |
| 0:30–2:30 | 持续到 45K，后台周期 checkpoint/validation | `D1` + `D2` 的少量 Dev episode（每个完整100步）、colosseum/pisa 轨迹诊断 | 用**实际每 episode 耗时**估算剩余评估预算，决定可负担变体 |
| 2:30–4:30 | 继续到 50K；里程碑后暂停评估或按剩余资源平衡 | `D2`/`D3` 同配对 Dev 起点，尽量完成 Dev-5 | 如果深度3开销超预算，降低并发/跳过 D4，不降算法核心配置 |
| 4:30–6:30 | 选 Y1/Y2，在 H0 下 Dev-5，必要时先完成冻结版本的小规模测试 | 主 D 候选完成 Dev-5，开始 Test-3/Train-6 | **优先完整主模型的官方评估**，不再分散试太多新配置 |
| 6:30–7:30 | 对选中 Y checkpoint 尝试 All-14 或 H3 固定评估 | 对选中 D checkpoint/policy 尽量 All-14，生成配对表/图 | 动态迁移空闲 GPU 给评估；任何未完成分组准确标 partial |
| 7:30–8:00 | checkpoint、manifest、结果核验 | aggregate、plot、README、final morning report | 停止新实验；确保作业安全退出、无 dangling writes、完整早晨总结 |

**允许动态再平衡**：训练结束/提前达到预算后，可以把 Y 释放的卡转为 D 并行 episode。D 扩展过慢时优先使用 D 的卡跑完整 Dev/All14，而不是牺牲数据完整性。反之如 D smoke 成本过高，把 D4 改为 `NOT_RUN_COST`、保留 D1/D2，并把剩余资源用于 Y 官方评价。绝对禁止多个进程同时以未经配置的方式占用同一 GPU 或在多个节点误绑定 GPU。

**计算预算的硬约束**：每轮 Dev sample 后自动测量 P50/P90 每个 100-step episode 运行时、当前剩余 walltime、可用并行 worker 数，计算预计完成时间。如果 `predicted_remaining > 0.8 * remaining_walltime`，自动收缩**候选数量**或可选评估范围；不得缩短正式 episode 步数、降低分辨率或更换 scoring 口径。排序优先：历史回归 → D1/D2/D3 Dev-5 → D 主版本 Test-3+Train-6 → Y 主版本 Dev-5 → Y 主版本 All-14 → 组合 → D4。

> 注意 4 GPU 的 D 并不代表一个 D episode 会变快 4 倍：通常一个 Planner episode 占一张 GPU，收益来自 4 个 episode 并行。历史 H3 ≈3.86 秒/决策步（barts_s0 受控计时），约 386 秒/100步、实际场景有波动；depth3 的真实耗时未知，必须小样本实测。Y 训练 4 卡 DDP 的吞吐也必须实测，而不是直接照搬其他模型训练时长。

---

## 5. 夜间自动监控与自恢复（必须实现，不是建议）

### 5.1 不要依赖 Login Node 的 tmux 存活

Boise State 集群曾有定期 Login/Ondemand patch 及用户之前的 `tmux→salloc` 会话中断顾虑。若当前使用的是依赖 Login shell 的交互式 allocation，**不要承诺 Login 重启后它能自恢复**；优先在允许的情况下采用 Slurm 管理的长时批处理，在实际 allocation 内以受调度的独立 step 执行。任务可能因节点故障或 walltime 被取消，须有可复现的重提交流程。不得在 Login Node 长时跑 GPU 任务或重型 monitor。

### 5.2 每条工作线都有真实、机器可读的健康状态

在工作目录写（实际输出避免和代码 git 目录混淆）：

- `STATUS.json`：`run_id`, `track`, `phase`, `job_id`, `hostname`, `gpu_uuids`, `parent_sha`, `code_sha`, `protocol_sha`, `last_heartbeat_utc`, `steps_done`, `episodes_completed/total`, `last_good_checkpoint`, `last_error`, `retries`, `estimated_remaining`, `next_action`。
- `STATUS.md`：供第二天人工阅读，每次节点或 checkpoint 变化都更新。
- `EVENTS.jsonl`：每次开始/结束、保存、失败、自动恢复、GPU/内存异常、决定不跑 D4、选择正式 checkpoint/策略都追加 timestamp 事件。不能只留一个不断被覆盖的 log。
- `logs/Y_train_rank*.log`, `logs/Y_eval_*.log`, `logs/D_worker*.log`, `logs/controller.log`, `slurm/%j.out`；stdout/stderr 非缓冲输出 (`python -u`, `PYTHONUNBUFFERED=1`)。
- `results/manifest.jsonl`：**每条 episode** 的 scene、start、config hash、model/policy hash、100-step 完整性、覆盖评估版本、输出路径、checksum 与状态。
- `metrics/live_metrics.jsonl`：至少记录 train loss、validation 与已完成 episode 的 per-split provisional metric，明确 `partial` 与 `official complete` 的区别。

心跳：建议 **每 60 秒**或每一个新训练/episode事件刷新。监控脚本本身须可以重启且不会启动重复的作业。

### 5.3 Shell 监控方式（示例，实际按集群环境调整）

```bash
# 查已授权任务及状态，不依赖 tmux
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.12l %.6D %R'

# 已完成/失败任务原因，若站点开放 accounting
sacct -u "$USER" --starttime today \
  --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES%50 \
  2>/dev/null | tail -40

# 各作业日志和 heartbeat
find work_dirs/exp6_v3 -name 'STATUS.json' -o -name 'EVENTS.jsonl' | sort
# 实际 stderr 路径需要审计
# tail -n 50 <verified_log_path>

# 独立 GPU 监控：必须在真实 GPU 分配所在的计算节点/受管理 step 中运行
nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
```

可实现 `scripts/monitor_once.sh`（一次检查打印状态/JSON）、`scripts/monitor_loop.sh`（每 60 秒只读检查，写事件）、`scripts/resume_if_needed.sh`（读取 manifest、checkpoint、job 状态，幂等地仅恢复**遗漏**任务）。**这些脚本需要由执行代理在仓库里真实实现**；本文件列的是验收接口，不是假装脚本已经存在。

### 5.4 故障分类与自治处理

| 事件 | 自动动作 | 必须记录 |
|---|---|---|
| 工作进度正常，GPU 利用率暂低 | 根据实际 I/O/step 耗时观察，不立刻中断 | heartbeat 与 utilization |
| 单个 episode transient 失败（模拟器断连、短暂 IO） | 同 scene/start/config/seed 重试最多 2 次；原残缺轨迹保持 `FAILED_ATTEMPT`，只完整100-step可计分 | 原 traceback、重试次数、hash |
| OOM | 先停止并发冲突、清理本任务已结束进程、确认 GPU binding；episode 单卡显存不足则改变**worker 调度**而非悄悄改模型/K/resolution；必要时标 `BLOCKED_OOM` | 实际 GPU UUID、峰值、解决方式 |
| DDP rank 异常 / NCCL 错误 | 停止当前本实验 process group，验证 last-good checkpoint，最多 2 次重启；不重复 global step | global_step、SHA、重启时间 |
| NaN / Inf 或 loss 爆炸 | 暂停最新异常 checkpoint 的选用，退回 last-good，检查 batch ID；保守 LR 调整须另标 `Y_LR_RECOVERY`，不可隐藏 | 首次异常 step、batch、调整 |
| Slurm TIMEOUT/PREEMPTED/NODE_FAIL | 尽力 checkpoint-on-signal，优先自动重排队/恢复，但仅在站点策略允许且仍有获批 allocation 时 | job exit reason、resume origin |
| heartbeat 超过正常步长的阈值未更新 | 先检查 Slurm state、GPU 负载、log 增长和进程；确认为挂死后才有序停止并从 checkpoint/manifest 恢复 | 挂死证据，不仅仅 `GPU=0%` |
| 计算预算不足 | 优先跳过 D4 或多 checkpoint 的 full G3；保留已完成数据，`PARTIAL_BUDGET` 标签，继续正式主模型结果 | 何时/为何预算变更 |
| 写盘失败/网络存储延迟 | 停止当前写入，确保没有坏 checkpoint 被标为有效；恢复后检查 SHA，已计算完成的 JSON 不能丢失 | filesystem、错误码、校验结果 |

所有自动恢复必须**幂等**：`(run_id, scene, start, checkpoint_sha, policy_sha, protocol_sha, seed)` 唯一标识 episode；同一键已完成且校验有效就跳过；未完成/损坏才重跑。运行中的 job 不能被另一个 controller 重复提交。控制器使用锁（如 `flock`）或 SLURM job name 及 manifest 双重校验。

### 5.5 作业时限与安全停止

- Slurm jobs 使用不超过剩余额度的 `--time`，合理 `--signal` 提前通知（例如 `USR1@600`，实际系统支持与信号处理先 smoke）及训练 checkpoint handler。
- 训练不可只在 45K/50K 保存；必须定期 durable checkpoint，以便节点故障后续跑。Stop/requeue 时不能覆盖 last-good。
- Deadline 前 **至少 30 分钟**停止新耗时实验，保留聚合、检查 SHA、绘图和 morning report 的时间；预估回收需更长时提前停止。
- 任何用于 pipeline 的 shell script 开头设置 `set -euo pipefail`，但监控器需要显式捕获子进程返回码并写入 `EVENTS.jsonl`，不能因为 `set -e` 导致错误无记录直接退出。
- 不得清空共享 `work_dirs`、不得批量 kill 用户其他任务、不得 `git reset --hard` 或 `rm -rf` 未授权目录。

### 5.6 结束时必须自动打印状态摘要

每当 controller 主循环退出（正常或异常）时，输出：

- 当前 Y 最新 checkpoint 与是否完整、Y1/Y2 实际完成的 global step。
- D1/D2/D3/D4 的完成 ep/总 ep、总 runtime、最大 VRAM、失败或跳过原因。
- CVPR Results 已注册 / 已 aggregate / 未完成的每一行及其缺失 scene/start。
- 真实 Train/Dev/Test/All-14 结果，不完整的只报告已完成个数和 `partial`。
- 下一步可恢复的指令与 `resume_manifest`，让次日人类/AI 能接着跑。

---

## 6. 最终评估、模型/策略选择和 CVPR Results 接入

### 6.1 选择的原则与优先交付

- **D** 用冻结 V3-T1 评估；**Y** 用历史 H0 评估；**Y+D** 在两线单独结果结束后评估。始终保持训练因素和 Planner 因素可区分。
- Dev-5 主看 AUC@20、AUC@100、Final，Final 大幅下降的变体需作为风险记录。选择策略不能只看 colosseum 单场景，也不能完全忽略它。
- Dev checkpoint/策略选择发生在 Test 汇总之前。Test-3 虽已经历史解封，也不再作为超参优化来源；无论 Test 结果如何不得回头修改配置。
- 若资源有限，**不必对所有 Y 里程碑和所有 D 变体均跑 70**；优先每个主方向至少一个锁定版本完整 All-14；其它版本 Dev-5/partial 需如实显示。历史 H0/H3 已有 70 episode 且经 hash regression 可以复用。
- **没有成绩必须过阈值才允许跑 G3 的禁令**：本轮以真实 Coverage 为最终结果，G2/单步排名只能作解释指标。

### 6.2 完整官方 Table 1 对齐

实际 `cvpr_results/table1` 真实路径按审计确认。注册新行建议（精确 ID 以现有 schema 定）:

```text
V3-Y45K-H0
V3-Y50K-H0
V3-YBEST-H3
V3-D1-H3-NORISK
V3-D2-DEPTH3-B3
V3-D3-DEPTH3-B5
V3-D4-DEPTH4-B3-OPTIONAL
V3-YD-COMBINED-OPTIONAL
```

每条注册需包含 `checkpoint_sha`、parent sha、训练 steps+dataset/seed、`policy_sha`、H3/D config/protocol sha、真实 D100/S10、R10、K4、Euler10、official evaluation version、scene IDs/start IDs、100-step、observational data split。不要把不同 D/Y 或 Test 后选择的策略伪装成同一行。

**正式报告（每个完成的 row）**：Train-6 30 ep、Dev-5 25 ep、Test-3 15 ep、All-14 70 ep；先每 scene 5 starts 平均，再 scene 等权；官方 AUC@10/20/40/50/60/80/100、Cov@相同步点、Final、Coverage curves（0–20、0–100、必要时20–50）、scene clustered paired bootstrap CI、win/loss 与 planner runtime/memory；辅助 monotonic 表单独列。对 H3/H0、B0、B2、MAG Beam 给出同场景同起点对照。

**All-15：** `sofia_church` 如果仍缺失，必须继续 `PENDING`，禁止把 All-14 标为 All-15；只有真实完成第 15 个场景并经官方 evaluator 验证后才生成 All-15。

- 调用现有 Table1 统一 evaluator + validator + aggregate + plot；**只通过 registry 注册**新增配置与 Episode，避免手工改 `TABLE1.md`、旧 CSV 或旧 LaTeX 行。
- 在注册后校验 `row count`、`scene IDs`、`start IDs`、`protocol hash`、episode steps、无 `NaN`/静默缺失。任何 `partialXofY` 不能参与“完整 All-14”平均。
- 主 Table1 若被设计为 auto-generated：更新注册和输入后运行生成脚本，不是通过手工文本编辑旧行。
- 原历史主表与历史图留有备份或 git diff 作为审计；旧行数值的 hash 比较必须不变（允许文件重新生成次序不同，但统计数值不能变化）。

### 6.3 数字出来后如何给出结论

为 Y、D、Y+D 分别回答：

1. **Y 是否改变了 Future Depth 质量？** UNKNOWN-ray MAE、Hit/No-Hit、Flow/Recon/KL、G2 selected gain、H0 G3 AUC20/AUC100/Final。
2. **D 的 depth/width 是否值得？** 相对 D1 和原 H3 的真实 AUC@20/AUC@100/Final，Colosseum/Pisa 后期覆盖，额外毫秒/步，搜索有效节点比例。
3. **Y+D 是否有交互收益？** 比较 YBEST-H3、D-BEST 旧V3 和组合，不要把模型收益与 planner 收益混淆。
4. **是否改善泛化？** Dev 与 Test 是否同向，Train 是否已饱和；负结果不删除。
5. **科学局限**：Test-3 非盲、14 场景、单随机种子、Beam 未来状态是近似、官方覆盖不等于 DistP1。

---

## 7. 早晨必须交付（详细且不可省略）

在 `experiments/exp6_v3/Y/reports/` 和 `.../experiment_D_h3_lookahead/reports/` 生成：

```text
Y/reports/
  00_STATUS_AND_RESUME.md
  TRAINING_TIMELINE.md
  VALIDATION_G1_G2.md
  G3_DEV_AND_ALL14.md
  Y_FINAL_REPORT.md

ablation_study/experiment_D_h3_lookahead/reports/
  00_STATUS_AND_RESUME.md
  SEARCH_AUDIT.md
  COLOSSEUM_PISA_DIAGNOSIS.md
  G3_DEV_AND_ALL14.md
  D_FINAL_REPORT.md

# 额外统一交付（位置以仓库既有报告风格决定）
  OVERNIGHT_8H_SUMMARY.md
  CVPR_TABLE1_ALIGNMENT.md
  RESOURCE_AND_FAILURE_LOG.md
  configs_frozen/...
  figures/coverage_0_20.png
  figures/coverage_0_100.png
  figures/colosseum_20_50.png
  figures/compute_vs_auc.png
```

### 7.1 `OVERNIGHT_8H_SUMMARY.md` 必须有的表

| Run | Frozen model/config | Train steps or D/B | Train 6 ep | Dev 5 ep | Test 3 ep | All14 ep | Dev AUC20 / AUC100 / Final | All14 AUC20 / AUC100 / Final | sec/step | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| H3 Historical | V3 40K H3 | D2 baseline | 30/30 | 25/25 | 15/15 | 70/70 | 0.444/0.759/0.875 | 0.424/0.762/0.901 | 3.86 | REUSED |
| Y1 | ... | 45K H0 | ... | ... | ... | ... | **真实数值** | ... | ... | ... |
| Y2 | ... | 50K H0 | ... | ... | ... | ... | ... | ... | ... | ... |
| D1 | V3-T1 40K | D2, risk0 | ... | ... | ... | ... | ... | ... | ... | ... |
| D2 | V3-T1 40K | D3, B3 | ... | ... | ... | ... | ... | ... | ... | ... |
| D3 | V3-T1 40K | D3, B5 | ... | ... | ... | ... | ... | ... | ... | ... |

未跑完：`INCOMPLETE/PARTIAL_BUDGET` 并给出 episode ID；不得填写 `0.000` 或沿用旧结果伪装为新结果。`H3 Historical` 值为现有文档参考，实际正式 report 应复用并校验真实 Table1 内容。

### 7.2 执行代理回报给用户的标准

- 先一句话说明：**Y 到第几 K、哪个 Checkpoint 可用；D 运行了哪些 Depth/Width；CVPR 完整了几行。**
- 报真实 Dev/All14 Early AUC、AUC100、Final；图和 per-scene 详细报告路径。
- 给出最明显的改进和最明显的失败场景（尤其 Colosseum），也要给置信区间或诚实承认 sample 太少。
- 列出异常与恢复：什么时候出现 OOM、哪条 episode 重跑、哪些步骤被预算放弃、实验是否仍可恢复。
- 如果仍有 job 运行，明确 job IDs、最后 heartbeat、下一次恢复入口、预期资源占用；不要声称全部结束。

---

## 8. 可交给执行代理的先后顺序（按步骤执行，无须等待人工中途回复）

1. **DISCOVER**：定位实际代码、V3-T1 完整 checkpoint hash、原 H3 参数、CVPR registries、Slurm 当前 allocation。
2. **FREEZE**：写 Y/D `FROZEN_PROTOCOL.md`、`PROVENANCE.md`、`RUN_MANIFEST.json`，启动日志与状态控制器；历史内容只读。
3. **SMOKE**：训练100步恢复测试；H3一条真实100-step regression；深度3一个 candidate-set 推理成本 smoke，检查合法路径与显存。
4. **DISPATCH**：把 4 卡给 Y 继续到45K→50K；另4卡并行 D1/D2/D3 的合法场景 episode，不等所有 D 代码完成才开始首个 D1 评估。
5. **MONITOR**：持续 shell/Slurm status 与 metrics heartbeat；异常自动分类，安全重试/续跑、完整记录；每隔约30分钟写状态摘要。
6. **EVALUATE**：优先 Dev-5 主三指标，接着对锁定的主 Y/D 版本运行 Train-6/Test-3/All-14，输出真实完整100-step closed loops；按 budget 可跳过可选 D4/Y3/组合。
7. **REGISTER**：通过 CVPR 原生流水线注册、校验、聚合，不手动改历史主表；形成图/表/CI。
8. **FINALIZE**：保留最后一份有效 checkpoint、所有完整 episode、失败日志、自动恢复 manifest；生成晨间总报告及准确的完成/未完成清单。

### 特别强调

- 用户明确希望充分利用夜间约 8 小时：**不要在达到轻微负结果时停掉整个 8 小时任务**；有显式的真实错误与预算限制时合理转移资源，继续完成更有价值的结果和报告。
- 用户仍然非常重视 **Final Coverage**；任何 AUC 改善都必须同时报告 Final，不能只有 G2 或 20-step 的孤立成绩。
- 优先做 **真实、可复现、可写进 CVPR 的完整结论**，而不是做很多只有 few-step pilot 的变体。
- 本任务文件是执行计划，**不是当前助手已经替你登录集群或启动了后台任务的声明**。
