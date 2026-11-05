# Experiment F — Stochastic Future Views × Coverage-Aligned Decision Study

> **归档建议（在仓库内）**：`experiments/exp6_v3/ablation_study/experiment_F_stochastic_coverage_decision/EXPERIMENT_PLAN.md`  
> **版本**：F-v1.0（运行前冻结）  
> **基线**：V3-T1 40K + D-v2；D100 输入、S10 Novelty、已观测几何可见性约束、K=4、Euler=10、R10、两步前瞻、风险系数 λ=0。  
> **状态**：待执行。本文是实验设计，不代表任何实验已运行、有效或取得提升。

## 0. 研究目的与不可变原则

本轮不是追求更多模型容量或简单地调参，而是验证：**多个随机 Future First-Hit Depth 样本，能否借助更贴近真实新增 Coverage 的价值函数和时间折扣，转化为更正确的 Camera Decision？**

保留 V3-T1 40K 权重、D-v2 的原始实现和全部既有 Table 1 数值。不给单独的 Gain Head，不新增训练，不改 ROI、GT 目标、Evaluator、相机合法动作、100-step 预算、真实 P1 更新和遮挡约束。真实 GT 只用于**离线** Oracle 计算，不得进入在线规划。预测第一/第二步的模拟观测不得写入永久 P1。

**关键背景**（来源：`Multiview_coverage_study_FINAL_REPORT.md`；本实验启动前由代码 AI 核对原始文件与代码）：395 个固定状态，原始 D-v2 的官方一步 Top-1 约 0.549，Regret 0.221 pp；K=4 相比 K=1/2/8 没有显著更低 Regret，但换样本造成的相机选择不一致率从 K=1 的约 53% 降到 K=4 的约 16%；同一评分函数喂 GT Depth 时 Top-1 约 0.438，直接使用 GT 目标表面 Coverage 时约 0.803–0.846。以上仅为历史**离线**发现，不得据此预设本轮成功。

### 不可混淆的符号

- `λ_risk`：旧版 H3 的风险惩罚。**本轮一律为 0。**
- `γ_horizon`：本轮新引入的**第二步边际收益折扣**，属于规划时域权重；不可称为旧 `λ` 或覆盖已有配置。
- `K`：同一个候选 Camera 的独立随机未来 Depth 样本数，不是 K 个跨 Camera 配对、共享的完整三维世界。
- `σ_soft`：Soft Surface Support 的空间核尺度，必须报告**运行时世界坐标单位**和与 Coverage 容差/体素分辨率的比值。
- `G_1`：第一步新增价值；`ΔG_2`：**第一步之后的第二步边际**预测价值，不能重新奖励第一步已包含的部分。

## 1. 文件布局、编号与执行产物

新建目录，**不要覆盖** `experiment_E_stability_plateau/` 或先前 `multiview_coverage_study/`：

```text
experiments/exp6_v3/ablation_study/
└── experiment_F_stochastic_coverage_decision/
    ├── EXPERIMENT_PLAN.md         # 本文，运行前冻结
    ├── FINAL_REPORT.md            # 用户只需阅读这一份结果总结
    ├── configs/                   # 冻结配置及 config hashes
    ├── scripts/                   # 复用已有脚本，新增最小组件
    ├── results/                   # JSON / CSV / 机器可读统计
    │   ├── F0_regression/
    │   ├── F1_horizon_discount/
    │   ├── F2_soft_coverage/
    │   ├── F3_adaptive_K/
    │   ├── F4_interaction/
    │   └── F5_closed_loop/
    ├── figures/                   # 决策曲线、真实几何对照、场景图
    └── logs/                      # Slurm、错误、重试、运行记录
```

**编号**：F0 = Baseline & scoring audit；F1 = Discounted Lookahead；F2 = Soft Coverage；F3 = Adaptive Stochastic Sampling；F4 = 组合与交互；F5 = 受控闭环。只产生一份汇总性的 `FINAL_REPORT.md`。所有既有 E 系列编号保持不变。

代码 AI 必须先定位真实仓库根目录，核对当前文件结构后创建对应目录。本文中路径是**建议目标路径**；不要将推测模块名当作已存在文件。原代码和 checkpoint 的绝对路径/哈希从现有配置读取并记录。固定状态、相机候选、K 随机种子、预测深度/表面集合、P1 有效域和 D-v2 可见性规则应复用现有经过回归的执行入口，避免另写一套近似算法。

## 2. 统一的数据分层与指标

### 状态与独立性

1. **探索层**：优先复用此前 395 个固定状态（E3 的 120 + G2 的 275）进行诊断和参数粗筛。它们已被历史研究反复查看，不能称为盲测。
2. **确认层**：在**启动 F 实验前**冻结未参与本轮参数选择的额外 episode/state 清单，尽量覆盖不同场景与 Early/Middle/Late，不使用 Test-3 调参。不能仅将同一轨迹的不同 step 当作独立样本。
3. **闭环层**：仅当 F1/F2/F3 有明确信号后使用预注册 Dev-5 或其他协议指定场景做受控、配对 100-step rollout；只在明确通过门槛后跑 All-14。以前 Dev/Test 已参与其他研究，应在论文中如实披露，不叫“严格盲测”。
4. 最终 CI 采用**按场景分层/聚类**的 paired bootstrap，并单独报告按 episode 的方差；状态间、同起点不同 noise seed 间依赖不能当作独立场景。

### 必报结果

- **固定状态**：官方一步 Top-1、官方一步 Regret（pp）、选中 Camera 的真实一步 Coverage Gain（pp）；几何两步 Oracle Regret / Top-1；真实二步并集；场景级胜负；Early `t<20`、Middle `20≤t<50`、Late `50≤t<100` 分组。
- **闭环**：官方 AUC@20、AUC@50、AUC@100、Cov@20、Final；辅以 monotonic/mono Coverage（**仅作为补充**），逐场景/逐起点配对差和 95% CI；运行时间、预测次数、峰值显存、失败/恢复。
- **机制**：第一步收益 `G1`、第二步边际 `ΔG2`、二者比例、Top1-Top2 分数间隔、K 样本方差、选择不一致率、软表面预测重叠、重复计分；GT 仅离线。
- 分别提供 `paired delta` 与 CI，不要凭肉眼较高均值/单个场景提升判显著。

### 共同约束

- 审计已确认 Official Coverage 对点数封顶和随机替换敏感且不严格单调，所有主表仍沿用原官方协议；Mono 作为敏感性检查。**不允许改官方 Evaluator**。
- ROI 不变：只给 ROI 有效域内的预测新表面计分；ROI 外历史已观测表面依然可以遮挡视线；保留 D-v2 的射线束物理过滤。
- 预测前瞻中的假想状态仅在内部评分使用，永久 P1 在调用前后校验和不变。
- 一切改变均可通过开关关闭，关闭后的 Camera 排序、score、轨迹应与冻结 D-v2 回归一致。

## 3. F0 — 冻结与可靠性门槛（先做）

**F0.1 代码/数据审计**：找到 `value_cov`、第一步/第二步价值集合、去重规则、K 样本均值、随机种子、模型/配置哈希和 D-v2 可见性过滤。核对报表里所谓 GT Depth + DistP1 表现更差时，两者经过的 UNKNOWN gating、First-Hit 处理、ROI、可见性、S10/BAND、单元映射是否同口径；分离“输入分布差异”与“value function gap”，不对历史结论做过度归因。

**F0.2 逐位回归**：使用已有 395 状态中可重放的样本，关闭新增功能，逐状态重建原 D-v2 的一层与两层分数及最终选择。最好保存每个 `(episode,step,c1,c2,k)` 的世界坐标命中、稀疏或紧凑的表面集合。检测出缓存不全则补算，且需保存 seed/hash。防止重新实现 `value_cov` 造成评分漂移。

**F0.3 映射单位检查**：数值比对世界/模型坐标、GT coverage 容差（ROI 审计：运行时 1.0）、P1 网格边界、预测命中点采样分辨率。确保 Soft Scoring 的半径用**真实世界距离**。

**F0 失败则停止**：不允许在 baseline 无法复现时继续做优化筛选。

## 4. F1 — Second-Step Discount（优先，预计成本低）

### 假设

D-v2 两步路径的最优第二步收益过大、或第一步候选共享近似相同的最佳第二步时，决策可能被噪声和未来幻觉收益支配。对尚未执行的第二步边际收益折扣，可能提高一步真实 Coverage 而保留必要的远见。

### 公式

从规划器原始数据拿到每个合法路径 `(c1,c2)` 的第一步值 `V1(c1)` 和**去重后的**两步并集 `V12(c1,c2)`；只有当原 `value_cov` 支持差分时才定义

```text
DeltaV2(c1,c2) = max(0, V12(c1,c2) - V1(c1))
J_gamma(c1,c2) = V1(c1) + gamma_horizon * DeltaV2(c1,c2)
J(c1) = max_{c2 legal after c1} J_gamma(c1,c2)
c1* = argmax_{c1} J(c1)
```

如果原始分数使用非线性权重或集合级 max，使 `V12−V1` 与真正的第二步边际不相等，须用**同一个 value functional**在第一步集合/并集上计算差分，并给出逐位回归，绝不可直接把未去重的原第二步 raw score 乘 γ。

### 参数与对照

- 固定 `γ_horizon ∈ {0,0.25,0.50,0.75,1.00}`；`γ=1` 应重现 D-v2；`γ=0` 对应同模型同评分的一步选择，但不能假装是原 V3-H0（H0 的评分协议不同）。
- 所有 γ 从**同一缓存**中的预测结果重打分，使用共同随机数（CRN），不重新采样以规避噪声混杂。
- 重点分析：`ΔV2/V1`、不同 c1 共享最佳 c2 的比例、路径分数 margin、Barts 早期错误示例、Colosseum/Pantheon 可能需要绕行的反例。
- 保存每个 γ 的 Early/Middle/Late 改善、regret、真实两步 oracle 和代价；不同评价目标（一/两步）要同时报告，避免奖励纯贪婪策略。

**F1 停止规则**：若任何 γ<1 都不能在预先冻结的确认层降低官方一步 Regret，或真实两步收益出现明显恶化，标记 FAIL，仍保留其诊断价值，不进入 F5。

## 5. F2 — Soft Coverage Surrogate（研究潜力最高）

### 假设

用硬 Novel Cell 去重会使生成几何的微小随机位移被视为不同表面。用有界、可解释的**局部 Soft Surface Support**近似每个 Camera 对空间单元的支持程度，有可能更合理地利用 K 个随机视图，并改善真实 Camera Ranking。

### 核心算法（无需训练）

对于候选 Camera `c` 的第 `k` 个样本，先复用 D-v2 的物理可见性过滤和 ROI 有效域约束，得到世界坐标预测命中点集 `S[c,k]`。采用稀疏空间网格 `v`（可从现有 scoring grid 派生），在局部邻域散射高斯或紧支撑核：

```text
q[c,k,v] = max_{x in S[c,k]} kernel(distance_world(v,x), sigma_soft)
qbar[c,v] = mean_k q[c,k,v]
novelty[v] = known-observation-aware unseen weight computed ONLY from S10/P1
V_soft(c) = sum_{v in valid ROI} qbar[c,v] * novelty[v] * volume_or_area_weight[v]
```

- `kernel` 先用高斯 `exp(−d²/(2σ²))`，对影响半径截断；`q∈[0,1]`，不把它称为已校准的真实概率。
- `σ=0` 或专门的 `hard` 开关必须**严格重现原 D-v2 hard scoring**，以便排除新公式/单元权重本身的改变。
- 对比仅平滑、不改变单元权重；与近似面积权重两个版本。**不要同时改多个因素后宣称收益来自 Softness。**
- 软表面支持必须在 ROI 内计分；预测落在已知遮挡后、已知占用、或按原法被判定不可新增的点不得因卷积外溢再次获得分数；使用已观测几何 / S10 的相同 novelty gate。
- 跨 Camera 的第二步 Soft Union 可使用 `max(q1,q2)`、`clip(q1+q2,0,1)` 或 `q1+q2−q1*q2` 等保守/独立近似**作为不同消融**。后者包含独立性假设，不能声称不同 k 对应同一个联合世界；记录每种假设的局限。
- 不允许通过依赖 GT 的真实目标点位置给在线 `volume_or_area_weight` 赋权；可采用仅由几何分辨率、射线投影或已知视锥决定的面积代理，并与原深度平方权重分开验证。

### 建议最小参数与拆解

- 首先只测 `hard`、`σ_soft ∈ {0.25,0.5,1.0,2.0} × τ_cov`，其中 `τ_cov` 由代码核对（ROI 审计中运行时为 1.0）；避免巨大 σ 将不同墙面无差别合并。冻结确认层之前不得对 Test 参数调优。
- 对每种配置报告：Soft Support 总质量、与真实可见表面空间分布的**离线**近邻诊断、跨相机重叠差、路径排名、官方一步/几何两步 Regret、成本。
- 对照：Hard + 原距离权重；Hard + 不带距离权重；Soft + 原距离权重；Soft + 面积代理。必要时对 Soft 的分数做只使用训练/开发状态的归一化，以免尺度影响 γ。
- 注意：Softness 不能修复并不存在的真实墙。必须测量幻觉扩散及过度平滑导致的负面案例。

**F2 停止规则**：在冻结确认层上没有真实一步 Regret 或真实两步排名的稳定改善，或多个场景出现显著幻觉扩散，则不进入 F5。

## 6. F3 — Time/Confidence-Adaptive Stochastic Sampling（K 的价值）

### 原则

**不因为步数增加就机械减少 K，也不直接降低 Flow 噪声幅度。** 先检验“后期样本间分数波动相对候选间差距更大”的现象；自适应的是**生成预算与决策时的置信处理**，保持原训练分布与采样过程不变。

### 数据与固定基线

- 用与历史 Multi-View 研究相同或可验证一致的种子生成 `K_max=8`，依预定义随机顺序构造 `K=1,2,4,8` 的前缀样本。
- 若 K=8 与 K=4 的 batch size 造成数值变化，必须在同一 K=8 缓存内比较各 K，并报告与原 K=4 的回归差异（历史 0.3% 左右），不能混入方法收益。
- 控制不同 K 的相机样本共享同一初始噪声顺序；报告总随机生成次数、Euler 步数、耗时。

### 必须区分三个机制

1. **采样均值方差**：固定状态/Camera 的 K 个得分的样本方差、均值不确定性、Top1-Top2 margin。
2. **状态是否难区分**：例如 `margin / sqrt(SE_top1²+SE_top2²)`；该比值仅是启发式决策信号，不是校准的置信区间。还须处理共用第二步和样本相关性。
3. **动态生成预算**：默认先 K=4；当排名间隔较小、或 bootstrap 重采样导致 top-1 不稳定时，仅对**可实现且保持两步路径公平性的候选**补足到 K=8。与固定 K=4、固定 K=8 和等预算随机补样做对照。所有候选在比较前须用相同的效用估计规则，不能让高 K 自动占便宜。

### 测量

- Early 0–20 / Middle 20–50 / Late 50–100，各分组报告：噪声方差与跨相机方差比、选择切换率、Top1、regret、真实收益、额外采样比例及均值时间。
- 固定 K1/K2/K4/K8，`K4→8 on ambiguity`，等计算预算随机加样，使用统一的 Hard/D-v2 评分；若 F2 成功，再在相同预算下测 `Soft×K` 交互。
- 检测是否存在“更稳定但更错误”的选择；不能把低 flip-rate 直接当成准确率提升。

**F3 停止规则**：若 Adaptive K 不能在同等预测预算下实质降低 regret，或者时间成本更高但质量仅为趋势，记录负结果，不进入 F5。

## 7. F4 — 组合与交互（只做小型确认，不做指数爆炸搜索）

执行前从 F1/F2/F3 各选**最多 1 个**通过冻结确认层的候选，比较：

- D-v2 原版（不可省略）；
- F1 单项；F2 单项；F3 单项；
- F1+F2；F2+F3；全部组合（仅当相应单项通过）。

重点检验：**Soft Coverage 是否让 K>1 的随机样本产生更大的决策收益？** 报告 K=1 与 K=4 在 Hard 和 Soft 下的交互差异：

```text
Interaction = [Gain(Soft,K4)−Gain(Soft,K1)] − [Gain(Hard,K4)−Gain(Hard,K1)]
```

Gain 使用同一批固定状态的真实选中 Coverage 或 `−regret`，按场景配对计算 CI。不要把“多采样稳定”冒充“多世界提升 Coverage”。旧 H3 风险项依然固定为 0。

## 8. F5 — 分阶段闭环（仅在离线门槛满足后）

**可进入 F5 的条件**：F0 PASS；某一候选在新冻结的确认层对真实一步 Coverage/Regret 有明确正向效果、几何两步收益没有明显下降；成本与 GT 信息边界合格；已冻结参数与版本号。对历史探索状态的改善不够。

1. **Smoke / 回归**：1 个既有失败起点 + 1 个一般场景；先验证 OFF 完全复现 D-v2；ON 能完成 100 steps、内存稳定、无 GT 泄漏。
2. **Dev-5**：每场景官方 5 起点，共 25 条/每策略；相同相机起点、噪声键、评估器。最多选择 1–2 个通过 F4 的版本；若需要新配对 D-v2 则补跑，不要混入未配对结果。
3. **预注册 Go/No-Go**（参数选择之前写入 configs，实验不能临时修改）：建议主指标 AUC@20、AUC@100、Final。至少真实 AUC/Final 无实质退化，收益不是只来自一个场景，运行成本可接受，并有场景层面的配对 CI/胜负表。可按实际量纲制定*具体门槛*，但必须在 Dev 运行前冻结；不允许看到结果后再定。
4. **All-14**：仅当 Dev 满足门槛且复现稳定，调用现有 Table 1 正规流水线，以**新方法 ID** 注册并保留所有旧行；All-15 仍在 Sofia Church 缺失时标记 incomplete。Test-3 不得用作超参数筛选，也不能称严格盲测。
5. **失败时停止**：无需为了“跑满 GPU”追逐无依据的参数改动。提供明确失败原因、负例与最经济的下一步。

闭环中区分 `true` 第一步观察只用于更新实际 P1，与内部想象第二步严格隔离。每次两步做完只执行第一步、获取真实 Depth、重规划。

## 9. GPU / Slurm 执行策略（面向 BSU HPC）

这份文档是供**有实际仓库与 HPC 权限的代码 AI**执行的；不要声称聊天助手已经启动作业。优先采用 **F0→F1/F2/F3 可并行→F4→F5** 的作业依赖图：

- F0 CPU 为主，必要时少量 GPU 做严格 baseline replay；F0 FAIL 应停止其余阶段。
- F1 复用预测缓存，CPU/少量 GPU 即可；不要抢占必须做推理的卡。
- F2 稀疏 Soft Map 构建可按场景分片；进行体素-neighbor 查询时限制候选距离和临时张量大小，避免 dense 巨型 3D occupancy 与爆显存。
- F3 K8 预测最耗 GPU：优先生成一次并缓存，所有 K 子集共享该缓存，避免重复生成。
- F4 仅对 F1/F2/F3 通过门槛的少数配置合并，不做大网格搜索。
- F5 闭环独立 job、逐 episode checkpoint，复用既有资源控制，不同 job 不要在同卡争抢显存。

Slurm 资源 (`--gres`, `--mem`, `-c`, partition, time) 必须按照集群当时可用资源及旧项目可靠脚本设置，不要臆造当前空闲节点。保留 `sbatch --dependency=afterok:<jobid>` 的逻辑依赖。支持断点续跑、原子写文件、每个 episode 的 seed 与配置哈希。若设置约 8 小时预算，优先保证 F0/F1/F2/F3 的**可复现诊断结果完整**，而不是把 GPU 耗尽；不超时承诺自动交付。

如脚本实际运行出现 OOM、数值错误、无法复现 baseline 或数据口径不一致：保存日志，停止相关分支，不要私自改变科学设置掩盖错误。

## 10. 结果判定与论文叙事

- **证据等级**：已经测得 / 有统计支持 / 趋势 / 机制假设 / 未验证。
- 可以把 K=4 称为“每个候选 Camera 的四个随机 Future Observations / hypotheses”；**不得**称之为四个跨 Camera 联合一致的完整 `Worlds`。样本编号相同并不代表来自同一个世界。
- 不允许用固定状态 Top-1 改善替代闭环 AUC；不允许用 GT-only Oracle 值作为可部署 Planner 成绩。
- 若 F1 成功，解释“较远时间层的想象价值应折扣”；若 F2 成功，解释“生成分布的空间支持比硬体素数更适合预测 Coverage”；若 F3/F4 成功，解释“多个随机 Future Views 实际带来决策收益”。
- 若结果仍是负面，保留 D-v2 为默认主方法，诚实报告失败与边界；不要强行改模型或进入大规模训练。

## 11. 最终交付（严格）

仅生成**一份**高信息密度面向用户的 `FINAL_REPORT.md`；本文 `EXPERIMENT_PLAN.md` 为原始实验协议，不算报告。其他过程放 JSON/CSV/PNG/logs，不另出一堆重复 Markdown。

`FINAL_REPORT.md` 按下面结构：

1. **EXECUTIVE RESULT**：F0…F5 状态、最佳版本（或维持 D-v2）、是否实际完成闭环；禁止把未跑实验写成已完成。
2. **BASELINE/REPRODUCTION**：checkpoint/config hash、395 状态复现是否成功、ROI/GT 隔离是否正确。
3. **F1**：γ 消融、Early/Middle/Late 真正 Camera Gain/Regret、两步正确性、失败案例。
4. **F2**：Hard vs Soft、核尺度、Soft Union、物理过滤与扩散风险、计算成本。
5. **F3**：K/Adaptive K、噪声随步数的作用、同预算比较、是否存在 K×Soft 交互。
6. **F4/F5**：哪些组合进入真实闭环；Dev/All14 AUC@20、AUC@100、Final、配对 CI 与时间（若执行）。
7. **CONCLUSION**：三项是否真的产生增益；是否支持多样本 Future View 论文主张；下一步具体阻塞点。
8. **REPRODUCIBILITY**：指向 `configs/`、`scripts/`、`results/`、`figures/`、`logs/` 的相对路径，附任务 ID/哈希，避免长日志粘贴。

核心结果至少有一张 **F1/F2/F3/Interaction × 固定状态真实一步 Regret × 两步收益 × 成本** 表，以及（有闭环时）一张 **D-v2 vs 候选方法 Dev/All14** 表。结论写清楚 sample counts/独立场景数、CI 计算的聚类单位。

---

### 最后提醒

这次**不能为了跑满 8 GPU 或优化一张表，而修改冻结的 D-v2、随机种子、GT 目标或已有实验结果**。本实验真正要回答的是：在物理一致的 ROI 预测中，随机 Future View 样本是否蕴含能够转化成正确 Coverage Decision 的信息；若没有，明确告诉用户原因。
