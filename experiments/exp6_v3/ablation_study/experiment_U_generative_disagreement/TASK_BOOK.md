# Experiment U — 任务书原文（用户 2026-11-13 粘贴；逐字保存，不修改）

# Experiment U — Generative Disagreement for Active View Planning

## 0. Background and Objective

我们目前的 Active 3D Mapping 主系统为 **V3-T0@40K + D1 Planner**。

当前模型使用：

- Partial TSDF → 3D VAE Encoder → Latent Geometry \(\mu_t\)
- Candidate Rays → Latent Ray Encoder → Condition Map \(C_c\)
- 2D Conditional Rectified Flow U-Net
- K=4 Future First-Hit Depth Hypotheses
- Coverage-Aware Two-Step Planning（D1）

目前 D1 利用多个未来深度假设估计候选视角及其后续视角的联合新增覆盖。

我们希望研究另一种规划策略：

**Generative Disagreement-Driven Exploration**

其核心是：对于相同的部分地图和候选相机，模型生成 K=4 个可能的未来几何观测。如果这些假设之间存在较大差异，说明该视角下未知几何具有较强的预测歧义，真实观测可能帮助消除这种歧义。

本实验希望检验：

> Does maximizing disagreement among generated future surface hypotheses improve active mapping efficiency compared with coverage-driven planning?

注意：当前 V3-T0 生成的是视角条件化 First-Hit Depth Hypotheses，并不是完整的 3D SDF Scenes。因此实验应基于生成深度及其反投影表面计算几何分歧，不得在报告中声称生成了完整三维场景。

## 1. Experiment Directory

新建实验目录：

`experiments/exp6_v3/ablation_study/experiment_U_generative_disagreement/`

建议结构：

```text
experiment_U_generative_disagreement/
├── README.md
├── configs/
├── scripts/
├── reports/
├── results/
├── figures/
└── logs/
```

实验名称为：

**Experiment U — Generative Disagreement for NBV**

所有新增代码、配置、日志与报告均存储在该目录。

尽可能复用已有 V3 模型、官方 Coverage Evaluator、模拟器与 D1 Planner。

**不要修改已有的 V3-T0 Checkpoint，不要覆盖任何历史实验文件，也不要修改冻结的 D1 主实验结果。**

如需复用已有模块，优先通过独立的 Planner 策略适配层实现。

## 2. Phase A — Existing Pipeline Audit

首先进行代码审查，不要直接运行完整实验。

需要明确以下信息：

1. V3-T0@40K 的实际 Checkpoint 路径及 SHA256。
2. K=4 的四组噪声如何生成、缓存及在候选视角之间复用。
3. 模型如何得到 \((\hat b,\hat y)\)。
4. 如何将生成结果解码为 First-Hit Depth。
5. 如何得到每条射线的 UNKNOWN Mask、Known Depth 和 Validity Mask。
6. D1 如何计算新增覆盖单元，以及如何选择第二步候选视角。
7. 当前 Coverage Evaluator 的接口、统计方法及 All-14 场景名单。
8. 是否可以在不重新运行生成模型的情况下复用候选视角的 K 个预测结果。

重点查看现有的：

- `seen2scene/v3/`
- `seen2scene/futureview/geometry.py`
- `experiments/exp6_v3/shared/`
- `experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/`

同时参考 `V3_T0_D1_MODEL_ARCHITECTURE.md`。

审计完成后输出：

`reports/U0_PIPELINE_AUDIT.md`

必须提供真实代码路径、函数名、关键参数以及当前实现中的潜在问题。

## 3. Phase B — Disagreement Metrics

本实验首先设计三个候选评分指标。

### U1 — Hit Disagreement

对于候选视角 \(c\)，模型生成 K=4 个未来观测假设。

首先根据当前 Depth Decoder 的实际规则，确定每条有效 UNKNOWN 射线在不同假设下是否预测到新表面：

\[
h_r^{(k)}\in\{0,1\}.
\]

定义平均命中概率：

\[
p_r=\frac1K\sum_{k=1}^{K}h_r^{(k)}.
\]

使用 Bernoulli Variance 衡量命中分歧：

\[
U_{\mathrm{hit}}(c)
=
\frac{1}{|\mathcal R_c|}
\sum_{r\in\mathcal R_c}4p_r(1-p_r).
\]

其中 \(\mathcal R_c\) 为当前地图下存在可观测 UNKNOWN 区域的有效射线集合。

该指标在四个假设一致时接近 0，在各有一半预测命中和不命中时达到最大值。

需要特别注意：

- 不将 padding rays 计入分歧。
- 不将无效、已知区域或没有 UNKNOWN 区域的射线计入。
- 仅比较真实生成的不确定部分。
- 不允许直接使用 GT Future Depth。
- 对不同候选视角的有效射线数量进行正确归一化。

### U2 — First-Hit Depth Disagreement

仅考虑至少两个假设预测出有效未知表面命中的射线。

定义：

\[
U_{\mathrm{depth}}(c)
=
\frac{1}{|\mathcal R_c^{2+}|}
\sum_{r\in\mathcal R_c^{2+}}
\operatorname{Var}
\left(
\{\hat d_r^{(k)}:h_r^{(k)}=1\}
\right).
\]

注意：

- 使用同一套坐标单位或合理的距离归一化。
- 不允许把 No-Hit 随意表示为最大深度后直接计算方差。
- 至少两个有效命中样本才能计算命中位置分歧。
- 若没有合格射线，该指标返回 0。
- 检查远距离表面是否可能因深度尺度而获得不合理的高分。

此外，记录使用 UNKNOWN-Arc 相对坐标计算 Depth Disagreement 的结果，作为离线诊断，比较其与实际三维距离方差的差异。

### U3 — Geometric Surface Disagreement

本实验最值得关注的指标。

对每个假设的有效预测深度进行反投影，得到候选视角下的三维表面点：

\[
\hat{\mathbf p}^{(k)}_{c,r}
=
\mathbf o_c+
\hat t_{c,r}^{(k)}
\mathbf d_{c,r}.
\]

随后建立各假设对应的预测表面集合。

建议优先使用现有 D1 的三维覆盖单元定义，将预测表面映射到统一空间网格。

构建跨假设的表面占用频率，计算几何分歧。

对于某个预测覆盖单元 \(v\)，定义：

\[
p_v=\frac1K\sum_k\mathbb I[v\in\mathcal S_c^{(k)}].
\]

可以使用：

\[
U_{\mathrm{geo}}(c)
=
\sum_{v\in\mathcal V_c}
4p_v(1-p_v)
\]

作为初始候选指标。

但这个定义仍需要进行合理归一化和有效区域审查，尤其要防止视野更大、预测点更多的候选仅因为单元数量更多而获得更高分数。

建议同时统计：

- 分歧单元总数；
- 平均单元分歧；
- 预测覆盖单元总数；
- 各假设间的三维表面距离；
- 分歧与预测覆盖大小的相关性。

优先采用能够在不同候选视野和不同 UNKNOWN 区域规模之间进行公平比较的定义。

在 U0 报告中确定最终的 Normalization Rule，避免根据最终 AUC 结果反复修改。

## 4. Phase C — Controlled Planner Comparison

首先保留以下四种策略。

| Policy | Scoring | Lookahead |
|---|---|---|
| D1 Baseline | Expected Novel Coverage | Two-Step |
| U1 | Hit Disagreement | One-Step |
| U2 | Depth Disagreement | One-Step |
| U3 | Geometric Disagreement | One-Step |

这里有一个重要的实验设计要求：

**U1/U2/U3 先采用 One-Step，用于判断各分歧指标本身是否有探索价值。但这些结果不能直接证明 Disagreement 比 Coverage 更好，因为 D1 使用 Two-Step。**

因此还需增加一个严格受控的对照：

| Policy | Scoring | Lookahead |
|---|---|---|
| C1 | Expected Novel Coverage | One-Step |
| U-best | Best Disagreement Metric | One-Step |

如果 Disagreement 在相同规划深度下优于 Coverage，才能说明评分信号本身具有优势。

在此基础上，再研究：

**U4 — Two-Step Disagreement**

沿用 D1 的合法第二步扩展机制，并在两步规划中评价预测分歧。

注意，必须明确区分逐视角分歧和两个视角的联合分歧。不能简单将 U1 或 U2 分数相加后声称它等价于 Joint Information Gain。

如果需要，可以在多个候选视角的共同三维覆盖单元上构造联合分歧分数。

但这需要首先验证跨视角假设的几何一致性，不能假设相同噪声索引必然代表同一个完整三维世界。

如果缺乏合理的联合定义，允许先采用明确定义的两步启发式分歧评分，并在报告中注明其局限。

## 5. Phase D — Fixed-State Diagnostic

正式闭环运行前，先对已有固定状态进行离线分析。

优先复用此前 EXP5/EXP6 中已经缓存的候选状态和 GT Coverage Oracle。

对于每个候选视角记录：

- Predicted Coverage Score
- Hit Disagreement
- Depth Disagreement
- Geometric Disagreement
- GT Coverage Gain（仅离线分析）
- Candidate Ranking
- Disagreement 与 GT Coverage Gain 的相关性
- Disagreement 与 Oracle Choice 的关系

目的不是要求 Disagreement 与 GT Coverage 完全一致，因为二者衡量的目标不同。

而是检查：

1. Disagreement 是否能够区分不同候选视角？
2. 是否出现几乎所有候选的分歧都接近 0？
3. 高分歧是否主要来自生成噪声、无效深度或伪表面？
4. 高分歧视角是否通常对应有意义的未知区域？
5. 分歧是否与当前地图的未知区域规模高度相关？
6. 是否存在 Disagreement 高、实际新增 Coverage 很低的典型案例？

生成：

`reports/U1_FIXED_STATE_ANALYSIS.md`

并输出相应的候选排序图与预测深度分歧可视化。

## 6. Phase E — Full Closed-Loop Evaluation

完成固定状态分析并确认分歧指标实现正确后，运行完整主动建图闭环实验。

### Dataset

使用当前冻结的 Macarons++ 协议：

- All-14 Scenes
- 每场景 5 个相同起点
- 每条轨迹 100 Steps
- 官方 Coverage Evaluator
- 场景等权聚合

另外分别报告：

- Train-6
- Dev-5
- Test-3
- Dev + Test

不要把 All-14 全部称为 Unseen Scenes。

### Metrics

主要评价：

- AUC@20
- Coverage@20
- AUC@50
- AUC@100
- Final Coverage

同时保存完整的逐步 Coverage 数组，以便计算其他时间节点。

### Fair Comparison

实验中保持以下条件不变：

- V3-T0@40K Checkpoint
- 3D VAE、Ray Encoder、Flow U-Net
- K=4
- Euler Sampling = 10 Steps
- 当前 Partial TSDF 更新规则
- 候选动作集合和可行性检查
- 共享噪声生成规则
- 官方 Surface Coverage
- 每场景相同的五个起点

除 Planner Scoring 和明确指定的 Lookahead 设置外，不应更改其他变量。

GT 只能用于离线标签、模拟器已有合法性检查和最终 Coverage Evaluation，不得进入 Disagreement Planner。

建议先运行每种策略的少量 Smoke Episodes，检查没有异常后再执行 All-14。

## 7. Experiment Tables

### Table U1 — Full Closed-Loop Performance

| Method | AUC@20 | Cov@20 | AUC@50 | AUC@100 | Final |
|---|---|---|---|---|---|
| D1 Coverage (Two-Step) | 0.420 | 0.646 | 0.633 | 0.763 | 0.911 |
| C1 Coverage (One-Step) | TBD | TBD | TBD | TBD | TBD |
| U1 Hit Disagreement | TBD | TBD | TBD | TBD | TBD |
| U2 Depth Disagreement | TBD | TBD | TBD | TBD | TBD |
| U3 Geometric Disagreement | TBD | TBD | TBD | TBD | TBD |
| U4 Two-Step Disagreement | TBD | TBD | TBD | TBD | TBD |

其中 D1 一行是此前已归档的 All-14 基线结果，必须核对协议后复用；不得重新编造或覆盖。

U4 是条件实验，只有在 Phase D 中形成合理的两步分歧定义后才纳入正式比较。

### Table U2 — Held-Out Scene Comparison

单独比较 Dev/Test 上最有代表性的 Coverage 与 Disagreement 策略。

至少报告：

- AUC@20
- AUC@100
- Final Coverage

### Table U3 — Decision Quality Diagnostics

比较候选排序、GT Oracle Ranking 或 Regret，以及各分歧指标与实际覆盖收益的关系。

这一部分只使用真实存在的 Oracle 标签。

如果数据无法支持某个指标，应明确记录为 NOT_AVAILABLE，不得推断填充。

## 8. Visualization

至少设计以下三组图。

**Figure U1 — Coverage Curves**

展示 D1、One-Step Coverage 及主要 Disagreement 策略的完整 Coverage vs. Steps 曲线，并放大前 20 步。

**Figure U2 — Geometric Disagreement Visualization**

选择若干真实状态，显示：

- Current Partial Map
- Candidate Camera Pose
- K=4 Predicted Depth Hypotheses
- Back-Projected 3D Surface Points
- High/Low Disagreement Regions
- Selected NBV

重点展示分歧来自哪些未知表面区域。

**Figure U3 — Coverage vs. Disagreement**

展示不同候选视角的 Coverage Score、Disagreement Score 和对应 GT Coverage Gain，并分析两种评分策略何时作出不同选择。

所有图片必须来自真实模型输出，不允许使用示意数据冒充实验结果。

## 9. Statistical Evaluation

对 All-14 使用场景级等权统计。

对于各策略与 D1 的比较，使用相同场景与起点进行配对分析。

建议报告：

- Mean Difference
- Scene-Level Bootstrap 95% CI
- Per-Scene Win / Loss
- Early-Stage AUC Difference

如果分歧策略在 AUC@20 上取得优势，但在 Final Coverage 上下降，需要如实展示。

如果 Disagreement 在所有指标上都不如 Coverage，也必须保留实验结论，并分析失败原因。

## 10. Final Report

输出：

`reports/EXPERIMENT_U_FINAL_REPORT.md`

报告至少包含：

1. 实验目的与教授提出的研究假设。
2. 当前 V3-T0 模型实际生成对象的说明。
3. 各 Disagreement Metric 的数学定义及代码实现。
4. 所有实验配置、Checkpoint、随机种子、代码版本。
5. Fixed-State Diagnostic。
6. Full Closed-Loop Results。
7. Train/Dev/Test 分组结果。
8. AUC@20、AUC@100 和 Final Coverage 比较。
9. 真实可视化。
10. Disagreement 是否优于 Coverage，以及在哪些场景下有效。
11. 对 CVPR Ablation Study 的建议。
12. 是否值得进一步研究 Hybrid Coverage + Disagreement Planning。

最终结论必须建立在真实日志、原始 Coverage Curves 和统计结果基础上。

## 11. Execution Requirements

请按阶段执行：

**Phase A → Phase B → Phase C → Phase D → Phase E。**

先完成实现审计与评分定义，再进行离线测试，最后运行完整闭环。

不要为了预期结果去调节评分函数。

不要修改主模型权重。

不要将预测分歧直接称为经过校准的 Information Gain。

如果某项实验无法满足公平性要求，必须在报告中明确说明。

整个实验的最终目标是判断：

**Whether generative disagreement is a more effective exploration signal than expected geometric coverage, particularly during the first 20 mapping steps.**
