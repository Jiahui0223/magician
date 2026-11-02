# EXP6 — V5.2-M / Idea 2 Scratch70k：无人值守训练与自动评估任务书

**状态：实施任务书，不代表模型已修改或 HPC 作业已启动。** 目标使用现有 `seen2scene` 代码仓库和 BSU Borah HPC，要求代码 AI 完成审计、实现、提交 Slurm、保存全部中间结果并自动出报告；无需用户睡觉期间交互确认。

## 0. 已授权、不可改变的科学目标

- **Idea 2**：D100 Partial TSDF → 3D U-Net → Camera-Conditioned Ray-to-Memory Query → 2D Spatial Geometry Context → Depth-Aware Multi-Surface / Conditional Hazard Decoder → First-Hit PMF → `S_visible + λ·S_potential` → NBV。
- **没有** 2D Rectified Flow、没有 K=4 Flow worlds、没有生成 Depth 图后再调用旧 DistP1；First-Hit PMF 仅用于概率可见性评分。
- 训练目标仅 `L_surface + β L_firsthit`，**不加** Final Coverage / Ranking / Flow Matching Loss。真实场景完整几何仅可用于训练标签、离线评估，推理时严禁 GT 泄漏。
- **完整 Scratch**：全部网络可训练参数随机初始化（包括 3D U-Net、Geometry Query、新 Spatial Encoder 与 Depth-Aware Decoder）；**不读取** V4/V5/V5.2 作为权重初始化。可以复用已验证数据、标签缓存、模型实现和评估代码。
- `target_optimizer_steps=70_000`：指 DDP 同步优化器更新总次数，不是每张卡 70k。保持 Outdoor-only Train-6，以便 MP3D 仍为严格的 Indoor Zero-shot；保留历史 D100 网络输入、实际执行真实 Depth 更新、候选合法性、`n_win=5`，参考旧 S10 novelty known-band 规则，不更改真实 Coverage evaluator。评分 λ 固定 {0,0.1}、η=0.5，默认两路来自 **同一 checkpoint**。
- 旧 `experiments/exp6_v5_2/` 不覆盖；新路径 `experiments/exp6_v5_2_scratch70k/`，运行产物 `work_dirs/exp6_v5_2_scratch70k/`。

## 1. 此前结果与本次必须改善的短板

| 指标 | V5（Idea 1） | V5.2 旧 Pilot（Idea 2，λ=0） |
|---|---:|---:|
| MP3D AUC/Final | 0.333 / 0.501 | 0.238 / 0.310 |
| Outdoor AUC/Final | 0.711 / 0.891 | 0.608 / 0.781 |
| Outdoor G2 Top-1 | 0.540 | 0.411 |
| MP3D G2 Top-1 | 0.467 | 0.356 |
| 旧 V5.2 总参数 | — | 2,668,710（其中 HazardHead 13,953） |
| 评估数据 First-Hit NLL | — | Outdoor 3.56 / MP3D 4.11 |
| 预测 `p_nohit≈.98` 的射线，实际确实 No-Hit | — | Outdoor 59% / MP3D 8% |

**不要**把跨场景失败简单解释为“模型太小”；增大容量、Scratch、增强 2D 上下文、修改训练分布与几何约束同时发生，若结果改善，只能归因于这套**联合干预**，不能声称是单一变量的严格消融。本任务优先拿到强 Idea 2 的双域结果，而不是一次做完所有因果实验。

## 2. 目标容量与一版可实现的模型结构

### 2.1 参数估算（事先估计；必须实现后精确测量）

V5 报告中 `3D Encoder + Flow U-Net` 逐位迁移 24,745,890 参数，新增 V5RayDecoder 242,757 参数，**约 24,988,647 ≈ 25.0M**。旧 V5.2 报告 **2,668,710 ≈ 2.67M**，其中 HazardHead 13,953。

建议从旧 V5.2 **非 Hazard** 子结构形式复用（全部权重 Scratch）：约 **2,654,757** 参数（精确值依实际代码是否将未使用层计入而校正）；新增：

| 模块 | 初始设计 | 原型参数估计 |
|---|---|---:|
| 3D Scene Encoder + Ray Memory Query / Surface 基础 | 继承结构，不载入权重 | ~2.655M |
| **2D Spatial Context U-Net** | 4 尺度 channels `[96,192,384,384]`，每尺度下采样路径 2 个 ResBlock；上采样路径各 2 个 ResBlock；GN/SiLU/3×3 Conv、skip/add，输入/输出适配 64 channel | **21,723,136** |
| **Depth-Aware Decoder** | 深度 1D 序列宽度 192，6 个 DepthwiseConv + pointwise FFN×4 residual blocks，输入 ray feature + depth PE + 状态，输出 `surface`/`hazard`/可用区间细化特征 | **~2,021,955** |
| **合计** | 起始配置估计 | **~26,399,848 ≈ 26.4M** |

这些参数是对提议模块的本地 PyTorch 原型计数，不是仓库实际代码测量。原型假设 Ray features 64 channel，若真实 V5 接口不同应加 adapter，**最终必须输出真实参数表**，按训练实际模型校准，推荐总计 **20–30M**；超过 35M 或 8 GPU 下吞吐不可行时，优先减小 `Spatial U-Net` 的宽度/Block 数，而不是直接删除 Depth-Aware 模块。**不要为凑参数量牺牲数学正确性**。

### 2.2 数据流 / 计算成本

```text
P1_D100 → 3D U-Net (Scratch) → F0/F1
                                   Camera Intrinsics/Pose → Geometry-Guided Ray Query (Scratch)
                                   → [B,H=64,W=114,D_ray] per-camera rays
                                   → Spatial Geometry U-Net (new, 2D spatial context)
                                   → [B,H,W,D_ctx]
                                   → depth coordinates + valid range + P1 state
                                   → Depth-Aware 1D Decoder (per-ray ordered depth axis)
                                   → Multi-Surface marginal b(r,i) + conditional hazard h(r,i)
                                   → analytic FREE/BAND geometry constraints BEFORE transmittance
                                   → p_first(r,i), p_nohit(r)
                                   → 3D-world novelty/visibility aggregation + Potential
                                   → candidate scores → NBV
```

**注意**：Depth bins（例如 48 / 64 个）仅是输出 `b,h` 的 1D 概率积分区间；**不得**恢复旧 V4 固定 32 个 U-Net 特征采样。Depth 轴上采用单调、边界正确、近处较细的物理区间，保留单位/体素尺度；按 chunk 处理 Rays，不能构造 `H×W×D×D` 稠密注意力或一次性 3D 全 Token 注意力。3D memory retrieval 仍按 V5 局部+粗尺度。

Spatial U-Net 只做**确定性上下文融合**，不做 2D Flow、不引入时间步或噪声、没有 Depth Sampling。

### 2.3 First-Hit 物理约束：必须在累积可见性之前处理

定义 `ρ_i = softplus(raw_i)`，`h_i = 1-exp(-ρ_i·Δt_i)`，`T_i = ∏_{j<i}(1-h_j)`，`p_first_i = T_i h_i`，`p_nohit = ∏(1-h_i)`；浮点积累在 fp32，log-survival 稳定化，NoHit 与 FirstHit 概率和≈1。

- D100 **已确认 FREE** 的合法射线区间：`h_i = 0` **再计算** `T`；不能先算 `p_first` 再把 FREE 区间概率删除，否则前面虚假命中会剥夺后方 UNKNOWN 的透射率。
- D100 **BAND**：区分可信首次可见表面和具有厚度的近表面带，不能所有 BAND 体素都强行 Hit；依据实际 TSDF 表面交界 / 观测置信度提供软/解析约束。
- **UNKNOWN**：允许网络给出概率，不把未知当 FREE；不能在推理时直接用未来 GT 填未知区域。
- 真实 NoHit、超传感器范围、无效深度、数值离散区间边界要严格区分；所有未充分可信的标签用 mask。
- **Surface marginal `b(r,i)` 与 conditional hazard `h(r,i)` 分开定义**；`1-b` 不是可通行 FREE，墙 A 后墙 B 的信念不能因首次可见性计算而丢弃。
- 输出 `p_nohit` 可视化、校准、真实 NoHit 的 reliability diagram 及 UNKNOWN high-confidence false-nohit。

## 3. 训练（70k Scratch，全程自动）

### 3.1 数据与采样

- 只用 Macarons++ **Train-6** 多表面 Mesh / ray depth-peeling 标签，与历史 P1_D100 及已执行视角对齐。Train-6 内保留训练监控 S-c 轨迹；不能用 Test-3、Dev-3 或 MP3D 来选择权重、调损失权重或阈值（这些评估集历次使用，不是盲测）。
- 采样分层记录 `known-FREE`、`known-BAND`、`UNKNOWN+GT-hit`、`UNKNOWN+true-nohit`、遮挡后第二+表面、Near/Far 的份额。优先保证 UNKNOWN+GT-hit 和真正 NoHit 均有足够数量；不得把补采样造成的条件分布伪装成真实校准，**校准另用保持自然比例的 Train-6 holdout**。
- 训练中可用几何一致的数据增强（Camera pose / 距离尺度 / P1 采样状态的合法联合变化），所有标签同步变换，**严禁** GT scene geometry 作为网络条件。
- 固定数据版本、随机种子、global batch、loss reduction、AMP 精度、梯度累积、学习率 schedule；如果 DDP 使 global batch 与之前不同，报告 `seen samples`，不能只比较 step。

### 3.2 Loss / Optimizer

`L = L_surface + β·L_firsthit`（保留旧标签 valid mask，避免重复 loss、无效深度伪标签）；如需匹配 loss 尺度，在**仅 Train-6 监控**中校验梯度量级，冻结 β，不增加 coverage/ranking/flow loss。初始用 AdamW + Warmup + decay，实际学习率 / weight decay 由旧 repo 的 scratch training 经验选定并**训练前写入 FROZEN_PROTOCOL.md**，不能凭本文随意假设已存在相同配置。

`max_steps=70000`，冻结一次结构，推荐阶段检查点 `2k`（训练数值）、`10k`、`20k`、`40k`、`55k`、`70k`；每 `1000` steps 原子保存 `last.pt`（temp→rename）、每 `5000` steps 保存标记版本和训练集监控统计，保留 EMA/optimizer/scheduler/GradScaler/rng/per-rank sampler/cursor/step/git sha/config/parameter count，支持严格恢复（尤其重启后数据读取顺序与 loss scale）。

**最佳权重选择**：只用预注册的 Train-6 holdout `UNKNOWN firsthit NLL` + 已知物理违规率，且不能通过预测一律 NoHit 取巧；`last70k`、`best_train_holdout` 都保留。评估上的最好结果不作为选权重条件。

### 3.3 无人值守自动监测与分支（在作业开始前冻结）

1. **G0 先行**：参数量、一次前向、一次反向、两墙 / 九墙、PMF 归一、FREE-before-hazard、NoHit/截断、D100/S10 真值作用、表面去重、K=4 Flow 与 DistP1 调用计数为零、DDP 2 rank/small batch 与 resume 一致性检查；不通过则报告明确 blocker，**绝不消耗完整 70k GPU 时段**。
2. **吞吐预热**：200 个稳定 step 后记录 `steps/s`、显存峰值、样本速度、GPU util、NFS 等待和估算到 70k 所需 GPU 时长；允许仅在训练前按已定义规则调 batch/chunk/activation checkpoint，并记录变更。**不承诺 8 小时完成 70k**。
3. **数值容错**：nan/inf stop + diagnostic dump；OOM 可在预先规定的 **每卡 batch 减半 + 梯度累积加倍（保持 global batch）/Ray chunk 减半** 规则下最多重试 2 次，然后 fail-safe 停止；不得悄然跳过有问题数据使评价失真。
4. **定期 G1**：每 5k/10k steps 只用 Train-6 holdout 监测 NLL、Top1 interval、UNKNOWN/NoHit 细分、ECE、`p_nohit>.95` 的真实 Hit 比例、第二+ surface recall、远遮挡、FREE 虚假质量；保存一页 dashboard。训练不能仅按最小总体 NLL 选择（容易忽略 UNKNOWN）。
5. **固定里程碑 G2**：在 20k/40k/70k 对既有 Outdoor Dev-3 / MP3D VALSEL 计算**只读诊断**，不得根据它们调 λ/η、改 architecture、挑 best 或调 LR；按预注册指标判读是否朝正确方向发展。保存每步候选分数可审计分解。
6. **智能评估资源使用**：到 70k 或作业时限边缘，优先保存 checkpoint，然后固定选取 `last70k` / `best_train_holdout` 的 G2；若 Outdoor G2 Top-1 仍明显落后旧 V5.2 且 UNKNOWN false-nohit 未改善，自动只做 G3 smoke（4 Outdoor + 2 MP3D），节省时间；若达到预注册阈值（例如 Outdoor Top-1≥0.46、MP3D Top-1≥0.40，且 `p_nohit>.95` 误判比例较旧版本改善），开展 G3 smoke 后根据冻结的门槛决定 Full G3。**该门槛只决定是否运行昂贵评估，不能用于调超参/反复试 Test**。无论结局自动成文。
7. **优先保住训练权重**：距 Slurm time-limit 5–10 分钟时请求全 rank 同步保存 last/EMA 和原子 manifest，然后安全退出；若已完成 70k 尽量评估，未完成保留可恢复 continuation 脚本，报告精确 `steps_completed`；延续训练仅在合法重新提交时运行，不能假装训练完成。
8. **失败自动报告**：输出 `FINAL_REPORT.md`、`AUTO_DECISION.json`、`RUN_STATUS.json`（planned/running/checkpointed/completed/partial/blocked）、权重 SHA、Slurm jobID、所有日志、复现实验命令。不得把非盲测集称为 blind。

## 4. 评估：与此前完全一致，自动执行

| 类型 | 目标指标 |
|---|---|
| G1 Geometry | Surface 第二+召回、深遮挡召回、First-Hit NLL/ECE、每个距离段误差、UNKNOWN vs known、high-confidence false-nohit、FREE 概率违规 |
| G2 Fixed State | Outdoor Dev-3 275 states；MP3D VALSEL 90 states；被选 Camera 真实 Gain pp、Top-1、Spearman、相对 V5/V5.2/B2/E2 的配对分布；可审计 `visible / potential` |
| G3 smoke | 冻结少量指定 episode；两域 AUC、Final、Cov20、回访、停滞 |
| G3 Full | Outdoor 30×100、MP3D 10×150；仅按预注册资源门槛执行；分场景/楼层统计和 bootstrap CI |
| Scoring | **同一** 模型权重，λ=0 / λ=0.1；报告 potential 比例、翻转率、虚假潜力案例 |
| Efficiency | 参数量、模型单步训练耗时、Ray score 耗时、G3 wall-clock、8GPU DDP efficiency、GPU memory |

**G3 用户期望性能基准**：V5 Outdoor .711/.891、MP3D .333/.501；V5.2 旧 Outdoor .608/.781、MP3D .238/.310；强基线 B2 Outdoor .730/.906、E2 Indoor .364/.553。结果不可编造；绝不宣称没有检验的架构提升。

## 5. BSU HPC 资源与调度（官方资料已核实，live state 仍需查询）

官方 Borah L40 node：**4 × NVIDIA L40、64 CPU cores、512GB RAM**。`gpu` queue 支持最多 2 node/user，`gpu-l40` queue 最多 1 node/user。用 8 张 L40 需要从 `gpu` 申请 **2 node × 每节点4 GPU**，而不是 `--gres=gpu:L40:8` 单节点。60 CPU 总额分配为 **每节点 1 个 launcher task × 30 CPU**，由 launcher 再派生 4 个 `torchrun` GPU worker；不同于 `-c 60` 给每个 worker。`--mem=256G` 指**每节点 256GB**，不是两个节点总量。

在**登录节点轻量**执行：

```bash
sinfo -p gpu -o '%N %t %c %m %G'
squeue -u "$USER" -o '%.18i %.12P %.8T %.10M %.20R'
scontrol show partition gpu | grep -E 'MaxNodes|MaxTime|Allow'
```

**主提交**：调用附带的 `SLURM_BSU_8L40_TEMPLATE.sbatch`（由代码 AI 根据仓库实际环境填好 `PROJECT_ROOT`、依赖、真实 entrypoint）使用 Slurm 提交，不使用永久 `salloc`+`tmux`，避免用户休息时断线。`torchrun` 通过 `srun` 每节点一个启动进程，`nproc_per_node=4`，共 8 DDP ranks。必须在 `srun` 前核对每个节点 `nvidia-smi` 与 CUDA 实际可见 GPU 数；不要静默用 1/2 张 GPU 假称 8 GPU。

**8GPU 无法合法获得时**：若提交被拒绝，可选择同一模型 4GPU 单节点（`gpu-l40`，4 cards）并保持 global batch / LR 口径，报告降级原因和吞吐；若只是排队，**不能擅自抢占/取消其他用户作业**，保留排队任务并报告状态。`--time=12:00:00` 仅是本次作业 walltime 上限，不能保证队列开始时刻；集群计划维护/抢占都依赖 checkpoint / resume。

参考官方： https://hpc.boisestate.edu/en/latest/hpc_resources/ ; https://hpc.boisestate.edu/en/latest/scheduling/

## 6. 完成条件与真实交付

必须生成：

```
experiments/exp6_v5_2_scratch70k/
  ARCHITECTURE.md
  PARAMETER_AUDIT.md
  FROZEN_PROTOCOL.md
  G0_VALIDATION.md
  TRAINING_LOG_SUMMARY.md
  G1_FIRSTHIT.md
  G2_FIXED_STATE.md
  G3_SMOKE.md
  G3_FULL.md (若门槛通过，否则写明 skip)
  FINAL_REPORT.md
  AUTO_DECISION.json
  RUN_STATUS.json
  scripts/submit.sh / overnight_driver.py / slurm/*.sbatch
work_dirs/exp6_v5_2_scratch70k/
  checkpoints/last.pt / best_train_holdout.pt / step_*.pt
  logs/...
  eval/...
```

- 优先把代码及冻结协议提交 Git；产物保留于 scratch，禁止覆盖旧 checkpoint。
- 如果在作业结束前未能跑完 70k：必须写明精确步数，保留 last 完整状态和 resume 命令，不能写“已完成”。
- 若 G2/G3 明显退化：如实报告 Idea 2 当下仍有性能问题，不要自动切换到 Flow 或偷偷更改评分器。
- **代码 AI 必须立即开始自己能够操作的实施与提交，用户无需醒来追加确认**；但若不存在合法的 Slurm 资源/连接/权限，应停在可恢复状态，不能假称已经启动。
