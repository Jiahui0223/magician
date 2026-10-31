# EXP6 V3 — Ablation Study First（最新执行计划）

> **优先级更新（2026-10-31）**：用户明确决定 **暂停 V4 的架构实现与 U0/U1 训练**。先完成 `experiments/exp6_v3/ablation_study/` 的诊断实验；看到结果之后，再讨论 V4 的结构、是否从零训练、是否继承 B2 权重。**本计划取代先前 `EXP6_V4_AND_V3_ABLATION_AI_INSTRUCTIONS.md` 中的 “V4-first” 执行顺序**，但不删除、改写旧文件和历史结果。遵守仓库 `plan.md` 的合理通用工程规范；与本次明确要求冲突时，以本次任务优先并记录。

## 0. 核心研究目标与冻结范围

原始算法主链路：真实历史观测构建 Partial TSDF/P1 → 对同一历史状态枚举合法候选 Camera → 每个候选 Camera 通过冻结的 Future Depth/HIT Flow 模型生成 K=4 样本 → DistP1 估算新增覆盖 → Greedy 选下一步 Camera → **仅真实执行后的观测**更新 P1。不要改成完整世界生成、直接相机 Pose 回归、Beam Search 或额外训练模型。

本次只研究两个问题：

1. **Experiment A / GT-Depth Oracle Decomposition：** 现有模型选错下一相机，究竟主要由 Future Depth 预测、DistP1 评分 / 规划区域，还是候选空间限制导致？
2. **Experiment B / Predictive-Depth Disagreement Planner：** 利用同一候选视角 K=4 个随机 Future Depth/HIT 预测的差异选点，能否改善一步候选排序或闭环覆盖？

**绝对不要启动、排队申请或提交新的 V4 U0/U1 训练。** V4 scratch vs B2 warm-start 是留待本轮实验结束后共同讨论的决策；不得在本计划内替用户确定。

所有模型 Checkpoint 冻结：优先 **V2-B2、V3-T1** 做主分析；**E2-D100、V3-T0** 在缓存或算力足够时加入固定状态对照。保留 MAG-G、MAG-B、B0、B2、E2-D100、V3-T0、V3-T1 的既有 G3 参考表（统一比较，不把 MAG 当作能够输出 K=4 Future Depth 的同类模型）。不得以新实验名义重新训练任何模型。

## 1. 先审计并最大化复用 EXP5 A1/A2

实验前必须先阅读/定位：

- `plan.md`、`experiments/README.md`、EXP6 V3 的 `protocol.md`、`V3_COMBINED_REPORT.md`、各任务 G1/G2/G3 报告及 DEVIATIONS；
- 既有 **EXP5 A1 Prediction/Scoring**、**EXP5 A2 Stochastic Future Views** 报告及 `ABLATION_SUMMARY.md`；
- 其固定状态清单、Camera 候选列表、GT Depth 和 GT Gain 缓存、K=4 原始预测、score、seeds、scoring mask、物理 voxel size 和测试脚本。

EXP5 先前已经做过类似拆分，已记录 V2 模型的 GTDepth/DistP1/Oracle 分析、K=1/4/8/16 消融、室内区域裁剪、室外无关表面加分和样本分歧相关性。**禁止机械重跑已有结论。** 先写 `CACHE_REUSE_AUDIT.md`：哪些可以逐位复用；哪些因为 V3-T0/T1、新 checkpoint、输入地图、候选口径或协议差异必须补算。凡直接引用旧表，明确标注 `historical EXP5, not new V3 result`。

优先复用 EXP6 G2 已冻结的固定状态（若存在逐候选数据）；如果与 EXP5 的状态不一致，首先锁定一个所有待比模型共用的 `(dataset, scene, trajectory, step, candidate set, P1)` manifest，再重新计算必需的模型输出。不要把来自不同闭环轨迹的状态误当作严格配对。

## 2. 目录规范

```text
experiments/exp6_v3/
  ablation_study/
    README.md
    protocol.md                      # 在看正式测试结果前冻结
    STATUS.md
    shared/                          # 公共评分、指标、状态清单接口（优先复用正式包）
    tests/
    experiment_A_gtdepth_oracle/
      README.md
      configs/
      scripts/
      tests/
      report/
        CACHE_REUSE_AUDIT.md
        A_G0_VALIDATION.md
        A_FIXED_STATE.md
        A_FAILURE_CASES.md
    experiment_B_depth_uncertainty/
      README.md
      configs/
      scripts/
      tests/
      report/
        B_SCORE_AUDIT.md
        B_FIXED_STATE.md
        B_CLOSED_LOOP.md
        B_DECISION.md
    reports/
      V3_ABLATION_SUMMARY.md
      V3_ABLATION_COMPARISON_TABLES.md
      V3_ABLATION_DEVIATIONS.md
      V3_ABLATION_RESOURCES.md
work_dirs/exp6_v3/ablation_study/
  cache/  manifests/  eval/  logs/  figures/
```

尊重现有正式 Python Package 的模块布局；不要为每个实验复制 Dataset、Trainer、Flow 和 DistP1。`experiments/` 主要保留配置、入口、测试、报告。所有昂贵缓存、Checkpoint/大图和原始数据放 `work_dirs` 等 gitignored 位置。每份报告包含命令、真实 SHA、checkpoint 标识、数据 split、随机种子、实测成本、失败项。

## 3. Experiment A：GT Depth / DistP1 / Oracle

### A0. 同一状态与同一候选集的三条评分路径

对每个固定状态 `s`、其真实历史 `P1_s`、合法候选集合 `A_s`：

- **A1 / Pred+DistP1：** 对每个候选 `a`，从**冻结模型**生成 K=4 个 Future Depth/HIT，逐样本跑原始 `DistP1(P1_s, D_a^k)` 并以现有固定顺序/归约规则求均值，得到 `score_pred(s,a)`。复现冻结 B2 或 T1 评分结果作为回归测试。
- **A2 / GTDepth+DistP1：** 对每个同样的 `a`，基于真实场景和同一个深度传感器协议渲染**一张**真实 GT Depth/HIT，交给**完全相同**的 DistP1（相同 P1、10% scoring map、距离权重、体素去重、近远裁剪）。从这些分数选出 `a_gtdepth`。
- **A3 / OracleGain：** 根据真实执行该候选后的目标 Coverage 增量 `G_gt(s,a)` 对相同集合排序，选 `a_oracle=argmax_a G_gt(s,a)`。真实 Gain 必须使用既有 G3 目标/评价口径，不等价于 GT Depth 经 DistP1 的得分。

**GTDepth 为什么 K=1？** 固定真实静态场景和相机位姿、确定性渲染协议下只有一个真实观测结果；它是一个“完美预测”的评分器诊断，**不是**概率生成器，**不应该**人为复制或创造四个 GT 世界。A1 保留 K=4 来估计模型预测的期望 Coverage；A2 不要求估计多样性。若传感器本身定义了噪声，主诊断仍使用冻结的确定性 GT 传感器规则，不能临时改变这一目标。

**禁止 GT 泄漏：** GT 几何只能在离线渲染候选 Depth 和计算 Oracle Gain 时读取；不能注入 P1、模型输入、预测的 Camera 条件、合法动作生成或真实闭环 Planner。对每个候选只能渲染那个候选本来会观测到的可见区域，不能用完整 GT 几何直接修改预测可见表面。

### A1. 指标及可解释的误差拆分

```
G*(s)    = max_a G_gt(s,a)
a_pred   = argmax_a score_pred(s,a)
a_gt     = argmax_a score_GTdepth(s,a)
r_pred   = G*(s) - G_gt(s,a_pred)
r_gt     = G*(s) - G_gt(s,a_gt)
Delta    = r_pred - r_gt
```

- `r_pred`：当前模型 + DistP1 路径的一步 regret；`r_gt`：完美未来 Depth + 相同 DistP1 后仍剩下的 regret。
- `Delta`：替换 GT Depth 的**净收益**，可能小于 0；不要在逐状态把它谎称为必然非负的、可加的“纯预测误差”。A3 只给出当前合法候选集合内部的 one-step oracle，不是无限动作空间或长期规划的最优解。
- 输出 Top-1（明确 tie tolerance）、Spearman（含候选 gain 近乎相等处理）、regret、`Early/Middle/Late`、per-scene 和 scene-cluster paired bootstrap CI；记录 `r_gt` 在预先规定的状态组中的表现。
- 必须分析 **MP3D 中期停滞**（是否存在合法且实际高收益的 Camera；当前规划区域外的 GT gain 占比）和 **Macarons++ 高 regret**（虚假遮挡、距离权重、目标外/地面表面是否加分）。区域或评分算法改动只做 **counterfactual diagnosis**，不改变原始 A1/A2/A3 规则。

### A2. 检查清单（必过）

1. 真实执行过的候选 `GTDepth` 与时间 `t+1` Depth/coverage 对齐；Oracle Gain 与独立 Evaluation 实现一致。
2. 相机坐标系/位姿正逆、图像尺寸、近远平面、MP3D 与 Macarons++ 不同 voxel size / unit、TSDF 截断、BAND/EMPTY/UNKNOWN 语义正确。
3. A1 当前 score 与原闭环或 G2 缓存回归吻合；A2 与 A1 **只更换 Future Depth 来源**；A3 与 A2 分开。
4. 遍历同一候选集合时，候选顺序改变不应改变逐候选分数，argmax tie 使用冻结策略；所有缺失/无效样本显式记录。
5. 按**独立场景**计算配对 CI（MP3D floor，Macarons++ scene）；不要把几百个高度相关状态当作几百个独立场景。
6. EXP5 已经分析过的 Test-3 不能重新称为盲测；所有事后诊断都标为 exploratory。

**阶段交付 A：** `A_FIXED_STATE.md`、每状态/候选详细文件、`A_FAILURE_CASES.md`。形成一条简短结论：分别在室内/室外，是预测误差、DistP1 或候选视野限制占主导；并列出可信证据和例外。

## 4. Experiment B：Depth Uncertainty Planner

### B0. 保留模型、P1 和候选集合，只改变 Planner Score

原始 P0 基线：

```
S_coverage(a) = mean_k DistP1(P1, D_a^(k))
```

新增 **P1 / Unknown-Region DepthU**：**只比较同一个候选 Camera、相同像素/射线、不同 k 样本**；不跨 Camera 比较不同射线。观测状态来自真实 P1 的射线遍历，不来自 GT。采用：

- `U_hit(p)`：K 个样本 HIT vs NO-HIT 的 Bernoulli entropy；需要清楚定义真正的 valid HIT 以及 valid NO-HIT，不能混入 invalid / padded rays。
- `U_depth(p)`：在至少两个有效 HIT 的样本中计算有界、单位归一化的深度差异（建议 std normalized by valid depth span；不得把 NO-HIT 替代为 `far` 再算方差）。
- `M_unknown(p)`：该射线在候选的可观测、有效深度区间中是否接触 P1 未观测区域，并且没有被一个确定的**已知前方表面**遮住。沿射线 UNKNOWN 的几何边界与当前 P1 相同；拒绝给已经确认的完全已知区域虚高 U。

可用预先冻结的聚合式：

```
S_depthU(a) = mean_over_valid_image_rays(
    M_unknown(p) * [alpha * U_hit(p) + (1-alpha) * U_depth(p)]
)
```

权重及 DepthU 归一化在 `protocol.md` 写明，不得根据测试集自行选择。DepthU 是 predictive disagreement proxy，**不等于** calibrated epistemic uncertainty，也不是严格 Bayesian information gain。

**P2 / Coverage + DepthU**：将 P0 与 P1 的分数在每个状态的候选集合内用预先规定的归一化做比较，然后以预先固定/仅 dev 选定的少量权重融合。P0、P1、P2 都是核心组；可选 `RiskAware = mean(G_k) - lambda * std(G_k)` 只作为低成本探索性补充。P2 改变的是 Scorer，不是模型、不变 P1，也不额外训练。

### B1. 先固定状态 G2，再决定 G3

主模型 B2 与 V3-T1；若已有缓存则添加 T0/E2-D100。相同状态/相机/seed/K，记录以下内容：

- `Regret / Top-1 / Spearman`，分 Early/Middle/Late，室内/室外逐场景与 paired CI；
- DepthU 与 GT Future Depth Error、实际 Oracle Gain 的关系，重点区分“预测很不确定但实际无收益”的坏例子；
- K=4 在不同随机种子集下的选点稳定性，以及额外评分时间；必要时少量 K=8 **仅诊断**，不要事后扩大搜索空间。

**预注册 Gate（由代码 AI 在运行正式验证集前冻结，须可审计）：** P1/P2 至少有一个在 dev 的跨场景 Early Ranking 或 Regret 上显示有意义的改善、且不引入明显的跨场景崩溃，才按冻结协议开展对应评分器的完整 G3；若 Gate 不通过，报告否定结果，最多做少量标记 `exploratory` 的闭环。不得为了让 Gate 通过反复调 λ、场景、时间点或 K。

如执行 G3：用同一已冻结的合法候选生成、真实观测更新 P1 与 G3 初始状态；报告 MP3D AUC@150/Final、Macarons++ AUC@100/Final/Cov@20、每场景曲线、配对 CI 和选择分歧案例。GT/Oracle 数据只在离线评估，不在运行时 Scorer。`MAG-G/MAG-B` 的既有同协议闭环结果可放入对照表，但不把它们硬塞进基于 Flow K=4 的 DepthU 计算。

## 5. 执行顺序、GPU/Slurm 与自动推进

**最先做：** Git/资源及缓存审计 → 创建 V3 Ablation 的目录与 protocol → Experiment A GTDepth/Oracle 校验并完成固定状态 → 简短阶段回复 → Experiment B 的 DepthU Scorer + 测试 → G2 固定状态 → 依据 Gate 决定 G3 → 完整报告 → 向用户提出 V4 设计证据及可选方案（**不自行启动 V4**）。A 的 CPU 数据准备、GT 缓存索引与 B 的纯函数单测可并行，但优先保证 A 先产出科学结果。

资源原则：

1. 实际检查用户现有 Slurm 资源：`squeue -u "$USER"`、`scontrol show job <id>`、节点进程与 GPU 使用；旧截图里的 3429774 / 3435033 / 3435072 **只是历史 ID，不能假定仍在运行或属于空闲资源**。
2. **本轮无正式训练**，不为了“用满 8 张 GPU”而申请八卡；已空闲/合法持有的 L40/L40S 可在评估阶段按状态/场景划分成 1-GPU/2-GPU 推理 worker，优先复用 cache，按显存和吞吐实际测量决定并发，避免共享 NFS IO 过载。
3. CPU 优先：审计、状态索引、评分算法单测、已有 K4 cache 归约、报告；GPU 仅用于缺失的真实 GT 渲染（若必要）、缺失 Future Depth 推理及最终闭环。GT 可以 CPU 渲染时先测吞吐/质量，不默认 GPU 越多越快。
4. 每次并发确保 worker 负责互斥 `(scene,state,model,seed)`，采用原子缓存写入、校验完整度、检查 checkpoint SHA/config hash，失败重试和 Slurm Resume；不能让两个 worker 同时覆盖一个文件或同一份输入状态。
5. 若有 Pending 的冗余 GPU 申请，也要先核实 owner、实际用途、是否属本次任务；不未经授权擅自 `scancel` 正在运行的交互作业或已分配资源。
6. 不虚报 GPU-hours 或 ETA；测到实际吞吐后再估计。如无资源可用，不声称已经开始 GPU 评估；完成 CPU 部分并明确当前 blocker。

## 6. 交付和自主权限

- **阶段更新一：** Experiment A 完成时，简短说明缓存复用率、三条路径 A1/A2/A3 的室内外 regret、GT 单样本的语义、主失败因素、测试状态、总成本；然后自动进行 B。
- **阶段更新二：** Experiment B G2 完成时，报告 P0/P1/P2 的固定状态结果、跨场景稳定性、Gate 结论与是否启动 G3；根据预注册 Gate 自动继续。
- **最终交付：** `V3_ABLATION_SUMMARY.md`、per-state/per-scene 统计与 CI、测试日志与配置、资源账本、误差案例、所有偏差和阴性结果；给出下一轮 V4 是更值得走 U-Net+XAttn 还是其他方向的**建议**，并明确训练 Scratch vs Warm-Start 仍由用户另行确认。

允许代码 AI 自主修复**实现/测试/缓存/资源**问题；调整并发、mini-batch、无效射线处理、可复现性；但不能改已冻结的历史数据/模型/评估目标，不得在 Test 上选参数、不新增 V4 训练、不改成 Beam Search 或 Pose Policy、不擅自扩大实验矩阵。任何必须改变定义才能跑通的情况，先在 `DEVIATIONS.md` 留痕并报告；GT 泄漏、无效坐标、未能复现冻结基线则 STOP。

提交规范优先遵守本次仓库已有授权和 `plan.md`：建议 `[V3][ablation] ...`，无 Co-Authored-By，不 force-push，不提交大缓存/原始数据/权重。Push 之前核对真实 Remote/Branch 和已授予权限；如果权限不明，保留本地 Commit 并报告，不猜测。

**现在立即开始 EXP6 V3 Ablation 的缓存审计和 Experiment A。不要实现或启动 V4。**
