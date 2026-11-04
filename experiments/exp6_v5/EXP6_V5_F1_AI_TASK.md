# EXP6 V5 / F1 — Geometry Query Isolation & Early-Coverage Evaluation

**交给代码 AI 的完整执行任务书 · 设计日期：2026-11-03**

> **核心要求：不要把实验写成 GO/NO-GO 筛选。** 在代码和资源允许时，对预注册的主对照至少各完成指定的 **100-step G3 Pilot**，实际报告 **AUC@20、AUC@100、Cov@20、Final Coverage (Cov@100)**；G1/G2 是用于解释结果，不是阻止闭环运行的门槛。所有结果如实保存，包括退化和失败。不要凭训练 Loss 推断 Coverage。此次是一个新的独立分支，不影响正在运行的 CVPR Table 1。

## 0. 工作位置与本任务边界

项目根目录中创建独立实验目录：

`experiments/exp6_v5/F1/`

本实验的研究问题：**从已训练且可靠的 V2-Mix（B0）出发，仅将 Legacy Ray32/Conv1D Geometry Conditioning 替换成 V5 风格的多尺度 Geometry-Memory Query，能否在同等数据、预算和评估环境下，提高候选视角未来深度预测、NBV Ranking，尤其提高 Outdoor AUC@20、AUC@100、Final Coverage？**

不要把 V5 原有的 Multi-Surface Belief 监督、多表面直接评分、Hazard/Direct Scoring、VAE、语义头、4³ 粗格、Ray64、R1/R2/R5 等额外修改混进 F1。F1 首次主要对照只包含旧 Ray Conditioning 与 V5 Query Conditioning 的差别。不要动任何既有 checkpoint、benchmark 原始结果或冻结协议文件。

### 0.1 既有事实（须重新核对项目源代码，不能仅相信本任务书）

- B0 与历史 B2 **是同一个 V2-Mix EMA@17.5K checkpoint**，B0 用 S10 inference + S10 scoring，B2 用 D100 inference + S10 scoring；不是不同架构或两份独立训练权重。
- 其父权重来自 V2 Original Outdoor S10 60K 再 Mixed FT（实际选中 17.5K）。需要核对完整路径与 SHA256，不能只凭文件名。
- B2/E2/V3/V4/V5 历史闭环均为 R10。最近 Refresh G3 没有显示更快刷新能稳定提高 Outdoor AUC，F1 保持 R10。
- 8³ coarse, V2 的 F0 32ch / F1 64ch；Ray32 均匀采样 + Conv1D → 64D；2D rectified-flow U-Net、K4、Euler10。
- V5 原 Geometry Query: F0 至多 24 token、F1 至多 12 token，对数深度选择，约 8 pooled coarse token 和 1 null token，cross-attention/SurfaceMLP，最后产生 64D ray condition。**必须以当前 `seen2scene/futureview/v5_model.py` 真实实现为准**；不要随意写一个外观相似却不同的 Transformer 冒充 V5 Query。
- 历史 V5 训练时还引入额外 surface/bridge losses，原 V5 vs V4C 不能归因为 Query 一项；F1 要隔离 Query 效应。
- 当时 V3-T0（自训练 VAE）及 V3-T1（VAE+Scene XAttn）已经完成；**不重训 V3**，只做轻量历史摘要，且不作为 F1 运行的前置依赖。

### 0.2 第一优先级（并行）

- **P0（独立持续）**：`experiments/cvpr_results/table1/` 现有全场景评估继续运行，F1 不修改、不重启、不覆盖它，也不独占 GPU。只读它的 `PROTOCOL`/registry/评估器（如果已存在）。
- **P1**：在 F1 内做 checkpoint/code/config 审计、实现两个主训练分支、启动小规模适配。
- **P2**：匹配数据/训练预算的 G1 + G2 诊断，输出逐 checkpoint 与逐状态比较。
- **P3**：**保证有完整 100 steps 的 G3 Pilot**（见下文），得到实际 Final Coverage；如算力允许再扩大场景/起点。
- **归档任务**：V3-T0/T1 历史配对实验只读整理，不重新训练、不重新计算昂贵 G3。

## 1. 实验目录结构

建议创建（可根据 repo 实际结构微调，保持语义一致）：

```text
experiments/exp6_v5/F1/
├── README.md                    # 目的、复现命令、状态
├── PROTOCOL.md                  # 正式冻结的口径（在第一次正式训练前）
├── CONFIG_ALIGNMENT.md          # B0/Query/官方 Table1 的配置对照
├── CHECKPOINT_AUDIT.md          # 原 checkpoint/EMA、SHA256、加载明细
├── V3_HISTORY.md                # T0/T1 历史结论，只读摘要
├── RUN_STATUS.md                # 运行进度，不写无依据的已完成
├── DEVIATIONS.md                # 运行偏差与处理
├── configs/
│   ├── f1_common.yaml
│   ├── f1_a_baseline.yaml
│   ├── f1_b_geometry_query.yaml
│   ├── f1_eval.yaml
│   └── manifest.yaml
├── scripts/
│   ├── audit_inputs.py
│   ├── smoke_compare.py
│   ├── train_f1.py
│   ├── eval_g1.py
│   ├── eval_g2.py
│   ├── run_g3.py
│   ├── aggregate.py
│   ├── plot.py
│   ├── validate.py
│   └── slurm/
├── results/
│   ├── g1_predictions.csv
│   ├── g2_ranking.csv
│   ├── g3_per_episode.csv
│   ├── g3_per_scene.csv
│   ├── g3_curves.csv
│   ├── checkpoints.csv
│   └── summary.json
├── figures/
│   ├── coverage_curve_0_100.png
│   ├── coverage_curve_0_20.png
│   ├── auc_horizon.png
│   ├── final_coverage_by_scene.png
│   └── prediction_ranking_diagnostics.png
├── reports/
│   ├── TRAINING.md
│   ├── G1.md
│   ├── G2.md
│   ├── G3_PILOT.md
│   ├── G3_EXTENDED.md
│   ├── RUNTIME.md
│   └── FINAL_REPORT.md
└── table1_export/
    └── f1_model_registry_entry.yaml
```

完整训练权重、原始深度/位姿、轨迹和大日志保存在 repo 已有 `work_dirs/` 或 scratch 标准位置，F1 用受版本控制的配置和索引引用，不把巨量数据复制进 Git。

## 2. Stage P1-A — 所有实验运行前的严格对齐审计

### 2.1 源模型检查

定位 B0/B2 所对应的 **V2-Mix EMA@17.5K** 实际 checkpoint。核验：

1. checkpoint file full SHA256、EMA vs raw、parent V2O EMA@60K、训练步数与原配置；
2. 是否严格加载原始 V2 Encoder、Ray32、Flow，缺失/意外键一律打印并判定；
3. 历史 B0 S10 inference+score，B2 D100 inference+S10 score，**F1 的主比较以 B0/S10 为父协议**；
4. 新增 Query 分支使用完全相同的 3D Encoder/Flow 起始张量；对保留的权重校验 `torch.equal`/hash；
5. 控制随机种子、样本流、shuffle、augmentation、label mask、batch、optimizer steps，做到配对；
6. 用已有固定状态先验证 B0 旧代码的预测与评分可回归历史，禁止直接拿不受控新 runner 开训。

### 2.2 固定训练及闭环协议

在 `PROTOCOL.md` 冻结：

| 属性 | 主对照要求 |
|---|---|
| Parent checkpoint | B0/B2 共用 V2-Mix EMA@17.5K（完整 SHA） |
| Train scene domain | Outdoor Macarons++ 原 V2 Outdoor Train-6（S-c 留出规则保持） |
| Train input P1 | **S10**，与 B0 一致 |
| Train labels | 现有 S10 的 b/y/UNKNOWN mask；两个分支完全相同 |
| Inference P1 | **S10**，与训练一致 |
| Scoring map | **S10 DistP1** |
| Coverage evaluation | 与 Table1 对齐的统一 S10 主口径；另可导出历史单调口径 |
| 3D coarse block | 8³，不变 |
| Encoder output | F0/F1，不变 |
| Flow backbone | 现有 V2 2D rectified-flow U-Net，不变 |
| Flow inference | K=4，Euler=10，不变 |
| Encoder refresh | R10，不变 |
| Camera candidates | 原 Greedy-NV 候选与行动合法性规则，不变 |
| Depth & resolution | 保持历史 256×456、64×114 rays |
| Training budget | **相同数量的 optimizer steps、相同训练 views 与采样 RNG** |
| Test discipline | Dev 作研究决策，Test/All-15 不用于调参 |

若某项无法与历史配置对齐，先在 `CONFIG_ALIGNMENT.md` 解释及修复，不能悄悄用 D100 的旧 V5 data shard 来训练 S10 F1-B。S10/D100 影响的不只是图像密度，也会改变 UNKNOWN mask、标签 b/y 与有效监督集合；只要主对照不同就不成立。

## 3. Stage P1-B — 两个主要实验分支

### F1-A：Baseline Continue（Matched Control）

- 以原始 B0 EMA@17.5K 作为初始化；保留 V2 的 3D Encoder、Ray32/Conv1D、Flow、S10 train/infer/score。
- 继续使用相同 Outdoor S10 view samples 进行短程训练，**不加入新模块**。
- 训练目标使用原始 Flow Matching Loss、无新增辅助监督。
- 在 step 0、2K、5K 保留模型与 EMA Checkpoints，并且独立保留父 B0。
- 这条分支要排除“只是多训练了一点，所以好像更好”的解释。

### F1-B：V5 Geometry Query Only（本轮核心）

- 完全相同的 B0 EMA@17.5K 初始化，同一 3D Encoder F0/F1、同一旧 Flow U-Net；仅把 Ray32/Conv1D 的几何条件抽取接口换成 **忠实于已有 V5 的 Multi-scale Geometry Memory Query**。
- 具体按原 V5 `v5_model.py` 代码审计其 gather/index、F0 24/F1 12、coarse8、null token、对数深度探针、Mask、XYZ/深度位置编码、输出 64D 的全部逻辑；复用经过验证的模块，而不是凭概念重写。
- **本轮不启用** Multi-Surface Belief 的独立输出损失 `L_surf`、V5 原 `L_bridge`、Hazard、Direct Coverage Scoring、额外 Teacher Loss；Flow Matching 是主要优化目标。
- Query 输出接口保持 `[V,64,64,114]` 或现有 cond 编码入口实际要求的等价 shape、dtype、mask/coordinate convention；若模型内部是 `[rays,64]`，以代码定义为准。
- 新 Query 权重可以随机初始化，但 Encoder/Flow 从 B0 精确复制，打印保留参数覆盖比例。**不要偷偷加载在 D100/不同损失下训练的 V5 Query 权重作默认初始化**。若测试 V5 预训练权重，只能开独立命名的辅助消融并说明分布差异。
- Query 起初可能让旧 Flow 输入特征分布突变。需要先做输出统计（均值/方差、norm、finite、掩码、新旧 cosine），并在保留模型结构的前提下实现轻量的 64D Feature Adapter 或稳定化层。
- **推荐训练两阶段**：前 2K 冻结 B0 Encoder/Flow，更新 Query + 必要 Adapter；后 3K 以较小 LR 联合解冻与其接口相关的层（例如 Ray-cond projection/Flow conditioning）与 Query。是否解冻更深 Flow 必须经轻量 smoke 和日志验证，不得默认整网大 LR 重训。总样本和总 optimizer steps 与 F1-A 相同，记录不同模块训练冻结表。
- F1-A 的 Continue 是必要的强对照；因为 F1-B 分阶段适配，而 F1-A 是原模块继续训练，两者构成“Query + 适配流程”的系统比较，而不是严格排除优化路径因素的架构纯消融。报告必须如实说明。

### 仅在资源允许时的辅助 F1-BR

如果 F1-B 直接替换导致明显条件分布断裂、Bridge 阶段难恢复，可以单独建 `F1-BR`（不是自动替换主实验）：保留旧 Ray32 branch，并将 V5 Query 作为**零初始化门控残差**注入 64D 条件中，以保护原模型行为。训练预算、数据与 F1-A 匹配。该方案是 Query-Augmentation 而非 Query-Replacement，必须独立命名；不要把它当作 F1-B 的结果。

> 如果硬替换失败是因为工程不兼容，修复工程问题；如果模型性能一般，继续按照下面 G3 计划完整评估，不能只留“没有通过门槛”的报告。

### 训练超参数

以原 `V2-Mix`、`E2-S10` 的已验证微调配方为起点（如 AdamW、bf16、EMA、gradient clip、global batch=64、stale augmentation），但**严格核对**实际可用 GPU、LR 日程和历史断点的 optimizer/EMA 信息。

- 两分支都从同一个 **model EMA** 初始化，优化器按新实验协议重新初始化；不能给其中一个分支加载不一致的历史 optimizer state。
- 可用作首选短程预算：**5K steps**，保存 `0 / 1K / 2K / 5K`，每次带完整 Config Hash 和 EMA。具体 LR、warmup、新旧模块 LR ratio 在正式运行前冻结、写入两个 YAML；不要看了结果再调整。
- 建议先在一批小样本上 Overfit Smoke 与 Gradient Flow 检查，再提交正式 Slurm batch。
- 对训练成本、时间、GPU memory、每千步 wall time 和最终参数量如实记录，不在运行前保证完成时间。

## 4. P1-V3 历史结论复核（快速，只读，不占 F1 的关键 GPU）

查找已有 `experiments/exp6_v3/` 的 T0/T1/V3.1 原始报告。形成 `V3_HISTORY.md`，给出：

- V3-T0 = 自训练 VAE + RayEncoder；V3-T1 = 同 VAE + Scene XAttn；两者的训练数据、步数、checkpoint 选择是否确实同条件。
- 原报告中的 G1 深度/No-Hit，G2 Top-1/Selected Gain/Regret，G3 AUC/Final/Cov@20 的数值；缺失则写“未提供”，不要推断或补造。
- V3-T1 与 V3-T0 的**配对差异**能否验证 Scene Cross-Attention 的独立价值。
- 明确 V3 自训练 VAE 不等同于预训练 Seen2Scene Sparse VAE。该支线本次不重启。

只读复核即可；不要等待此步骤完成才训练 F1。

## 5. Stage P2 — 快速预测与选点诊断（必须配对）

固定相同的数据状态、候选 Camera、GT Depth、起点和 Flow 噪声，比较 `B0 parent`、`F1-A 2K/5K`、`F1-B 2K/5K`（若有 F1-BR 单独列）。优先使用 Dev-3 历史固定状态（例如 275 states），确认 S10 重建和 evaluator 一致。可将 Dev-3 的 150 条 B0-state 与 125 条 B2-state 作为两层分别报告，再做统一汇总，避免状态来源不一致带来误解。

### G1 — Future Depth / Hit diagnostics

至少统计：

- UNKNOWN-ray sample count、hit prior、有效标签数和按场景/step bucket 分布；
- true-hit 的 Depth MAE/median 和长尾错误；
- Hit Precision / Recall / F1、False No-Hit、False Hit；
- No-Hit calibration（可用 ECE/Brier，给出 bin 分布）；
- Early states（t≤20）与后期（t>20）分开；
- 新 Query 是否只改变某些特定 Ray / 远深度 / 多段 UNKNOWN 上的预测；
- 参数量、条件 Query 推理成本、每个 Camera scoring cost。

### G2 — Candidate Ranking / real one-step coverage gain

至少统计：

- Top-1 accuracy（与 oracle 同一动作集）；
- Selected Gain、Oracle Gain、Regret；
- Spearman/秩相关（若可计算）；
- 对比 B0 parent / F1-A / F1-B 的 same-state paired Δ；
- **Early states t≤20 单独指标**，不能只汇总整轨迹；
- Scene-level bootstrap 95% CI，保留 scene count 与胜/负数；
- 预测精度变好而 G2 没变好时，分析是不是 DistP1 与 GT Coverage Gain 不对齐。

注意：G1/G2 用于解释和确定接下来重点调试哪里，**不是是否允许计算 G3 的拦截器**。即使 G2 不理想，也要按 P3 跑完整 100 步的已预注册 G3 Pilot，以获得真实 Final Coverage。

## 6. Stage P3 — 闭环验证：必须交付 Final Coverage

**最核心要求**：不要因为训练 Loss、G1 MAE 或 G2 Top-1 不达门槛就跳过所有完整 G3。对 `B0 parent`、`F1-A`、`F1-B` 在相同开发场景、相同起点运行 100 steps，计算实际 Final；只要模型数值有效、GPU/数据可用，就完成预注册的基本闭环。

### 6.1 先运行 100-step G3 Pilot

推荐最小固定 Pilot：Macarons++ **Dev-3 × 每场景 2 起点 = 6 episodes / 模型**，每条都是 **100 steps**（不是只跑 20 steps 就称 Final）。起点具体 ID 在训练前冻结。B0 parent / A / B 必须使用一模一样的 6 个初始状态和对应 Seed；如果已有 B0 parent 6 条有效历史轨迹，允许直接复用。

如资源允许，扩到 Dev-3 × 每场景 5 起点 = **15 episodes / 模型**。不使用 Test-3 的结果来调训练/选 checkpoint。

输出每条轨迹 `Coverage[t] (t=0..100)`，并计算：

**主要三项（始终展示）**

1. **AUC@20（Early Exploration）**；
2. **AUC@100（Overall Efficiency）**；
3. **Final Coverage = Cov@100（最终覆盖）**。

**完整 Table1 对齐指标**

- Cov@10/20/40/50/60/80/100；
- AUC@10/20/40/50/60/80/100；
- Final Coverage；
- 完整 Coverage vs Step 0–100 Curve；
- 运行耗时/决策时间、Peak GPU Memory；
- 各场景与逐起点结果。

AUC 的离散积分、C(0) 起点、End Step、Cov@100=Final 的一致性按 Table 1 统一标准检查；不能将旧脚本的 AUC@100 与新积分规则混用。若 Table1 采用 MAGICIAN-compatible 官方主口径，F1 正式结果也必须经同一个 evaluator；如 F1 开发期暂时使用单调 S10，必须同时记录口径并在最终结果前重新评价。

### 6.2 模型选择与继续训练

不要用简单 GO/NO-GO 标签。把结果视为以下三个可能方向：

- **若 Query 在 Early AUC、Overall AUC 或 Final 有收益**：进一步分析其优势是否稳定、是否带来额外时间成本；可以选择继续适配训练或增加 Dev 起点。
- **若 Query 整体接近 Baseline**：保留完整 Final 与 AUC，说明新 Query 并未自动带来更高探索效率，作为论文消融/设计依据；不需要为追逐微小波动连续更改配置。
- **若 Query 明显退化**：保存完整曲线和 checkpoint，诊断特征分布断裂、Hit/No-Hit 校准、样本/Mask 逻辑或过拟合；按资源允许选择 F1-BR 小消融或收束研究。**不要删除退化结果，也不要为了显得成功而改变测试场景或掩盖 Final。**

事先冻结要展示的 checkpoint（推荐 A/B 的 `2K` 与 `5K`；正式选择只在 Dev），不能看 Test/All-15 再选择 best。最终可用 Dev AUC@20 作优先选择依据，同时独立报告 AUC@100 与 Final；如多目标冲突，保留 Pareto/权衡说明。不要伪称某一个 checkpoint 同时在所有指标最佳。

### 6.3 Full 100-step G3 的拓展

如果 Pilot 后还有时间/算力，先完成 Dev-3 × 5 起点，对 selected F1 模型冻结 checkpoint；然后作为 Table1 的新模型使用**完整官方 15 场景 × 5 起点**的统一正式评估。不得单凭 6 episodes 的 Pilot 宣称 All-15 改善。

**关键**：CVPR Table1 是正式评估的唯一汇总数据源。F1 的 `G3_PILOT.md` 是开发报告，不直接拷贝到 Table1 充当正式数字。

## 7. 数据配对与统计标准

- 每种模型相同 Scene + Start + Action Budget + Candidate Set 规则 + Evaluator + R10 + Noise Seed，独立闭环会导致后续轨迹不同，这是合法的系统比较。
- G1/G2 固定状态配对；G3 比较同场景/同起点但不同模型自主探索轨迹。
- 场景先对 2 或 5 个起点取均值，再对场景等权平均；所有表必须标注 scenes / episodes。
- Scene-level paired bootstrap CI，开发 Pilot 只有 3 个 Dev 场景，CI 不稳定，强调探索性；不要把不同种子 episode 当成完全独立场景。
- Train 场景均值、Dev、Test 和 All-15 **必须分开报告**；Train 结果不能作为未见泛化；Test 已在历史研究中观察过，标注其已解封状态。
- 不允许后验挑选有利场景、起点、checkpoint，再声称统一全场景结论。

## 8. Slurm 运行与容错

1. 在 login node 用 `squeue -u $USER`、`sinfo`、`scontrol show job` 检查 Table1 运行及 GPU 额度，避免干扰 P0。
2. 使用当前集群实际可用 partition、GPU 类型、CPU、内存、时间；不要假定固定有 4/8 张闲置 L40。
3. 训练分支、G1/G2、G3 均使用独立输出目录与 `sbatch`，记录 Job IDs 和依赖。
4. 训练 checkpoint 包括模型、EMA、optimizer、scheduler、global_step、RNG、config hash、parent SHA；可以中断续训，续训不改变总 optimizer steps。
5. 避免重复运行已有对齐历史 Episode；设置 per-episode `done.json`，完成性校验、异常重试、不能重复聚合。
6. OOM 等运行环境错误可以在不改变科学配置的前提下重试；若必须改 batch、resolution、K、积分或数据则写在 `DEVIATIONS.md` 并另起配置版本，不静默修改。
7. 失败 Episode 单独登记，**不要用均值补值**；Partial 与 Completed 必须区分。

## 9. F1 → CVPR Table 1 接口

F1 的每个候选模型在正式评估时必须向 Table1 `MODEL_REGISTRY` 提交：

```yaml
model_id: F1-GQ-v1
model_family: V5-F1
parent_model: V2-Mix-B0
parent_ckpt_sha256: '<由代码核验>'
checkpoint_sha256: '<实际训练后填入>'
training_data: 'Outdoor Train-6 / S10'
training_steps_added: 5000
inference_density: S10
scoring_density: S10
refresh: R10
flow_k: 4
flow_euler_steps: 10
planner: Greedy-NV
architecture_difference: 'V5 Geometry Query only; no L_surf/L_bridge'
selection_split: Dev
status: PENDING_TABLE1_EVALUATION
```

注意 F1 的实验开发目录和 Table1 的汇总目录分离：**Table1 只读入结果与 provenance，不允许在聚合时直接依赖 F1 的训练脚本或手工复制浮点数**。如果协议一致且历史轨迹齐全，可由 Table1 自己复评；最终 Table1 的 All-15 结果由其统一 evaluator 产生。

## 10. 完成标准与要交付的实际结果

**必须输出以下 4 类报告，即使模型性能不理想：**

### A. 完整对照与训练说明

`CONFIG_ALIGNMENT.md`、`CHECKPOINT_AUDIT.md`、`reports/TRAINING.md`：谁继承谁、有哪些可训练参数、学习率、训练样本、step 数、哪些 head 被删除/保留、哪里存在不可避免的优化差异；必须能复现实验。

### B. G1/G2

`reports/G1.md`、`reports/G2.md`：固定状态配对结果，尤其 t≤20；哪类几何/No-Hit 错误导致决策变化，真实增益是否改善。

### C. **G3 100-step，包含 Final Coverage（不可省略）**

`reports/G3_PILOT.md`、`results/g3_per_episode.csv`、`figures/coverage_curve_0_100.png`、`figures/coverage_curve_0_20.png`。每个主模型至少一组预注册的 6 条 Dev 100-step episodes；如果暂时未完成，准确写明未完成的 Episode 与原因，不能将 `G2 skipped` 当作 Final 结果。

Markdown 主表至少如下列（数值必须由代码填，不能预填）：

| Model | Parent | Train/Infer/Score | Added Steps | Dev Episodes | AUC@20 | Cov@20 | AUC@100 | Final Cov@100 | Peak VRAM | s/Step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 parent | EMA@17.5k | S10/S10/S10 | 0 | ... | ... | ... | ... | ... | ... | ... |
| F1-A Continue | same | S10/S10/S10 | 5k | ... | ... | ... | ... | ... | ... | ... |
| F1-B Query | same | S10/S10/S10 | 5k | ... | ... | ... | ... | ... | ... | ... |

另外保存完整版 7 个 Cov@k + 7 个 AUC@k，以及逐场景数据和 CI。

### D. FINAL_REPORT.md

用实际数据回答：

1. 从 B0 直接换 Geometry Query 后，最终 AUC@20、AUC@100、Final 改变了多少？
2. 相对于 **同预算 B0-Continue**，有无独立的 Query 贡献？
3. G1 几何预测改善是否与 G2 真实选点收益一致？哪些错误限制性能？
4. 哪些场景/起点有收益、哪些退化？Early（0–20）曲线和后期（20–100）曲线怎样？
5. 相比 V5 历史版本，去掉 L_surf/L_bridge 后是否有不同趋势？注意训练条件不同，谨慎归因。
6. Query 的代价是多少（参数、显存、每步决策时间）？
7. 下一步应当保留、简化、调整什么；在两天时间和论文截止压力下，给出最值得做的**一个**后续实验建议。
8. F1 的哪些最终 Checkpoint 已登记到 Table1，哪些仍处于待评估状态？

**输出必须是数字与图像驱动的分析，不能只输出训练 Loss、G2 Top-1 或“实验未通过”。没有实测 Final 就明确写尚未运行；尽力完成 G3 Pilot 后再总结。**

## 11. 现在开始的执行顺序

1. 立即检查 Table1 当前工作负载，保证不干扰它。
2. 创建 F1 目录、只读审计、冻结协议、准备 B0 Parent Regression。
3. 并行做 V3-T0/T1 历史总结（低开销、非阻塞）。
4. 实现 F1-A 和 F1-B，smoke 后提交两条 matched 5K 训练作业并保存 0/1K/2K/5K。
5. 在计算允许时跑 Dev 固定状态 G1/G2，并完整运行预注册的 100-step G3 Pilot；**不使用 G1/G2 门槛自动跳过 G3**。
6. 根据真实 AUC@20 / AUC@100 / Final 和成本决定是否扩到 Dev-3 15 ep 以及进入 All-15 Table1。
7. 完成全部可完成输出，更新 RUN_STATUS、FINAL_REPORT 和 Table1 registry。不能虚构未完成项。

**现在可以直接执行上述任务。不需要每一步询问确认；有不可验证的依赖或科学协议冲突则记录并采取可重复的保守方案，绝不私自修改已有历史文件或结果。**
