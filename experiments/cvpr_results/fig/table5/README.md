# Table 5：Why multiple generative hypotheses?

本目录原名 `fig/table_hypotheses/`，2026-11-16 按用户要求改名为 `fig/table5/`。旧文件统一加 `table5_v1` 前缀，用 git mv 改名，并已核对旧脚本仍能逐字节复现这些文件。

**论文用 v2**：`table5_v2.{png,pdf,tex,csv,md}`，由 `scripts/make_table5_v2.py` 生成，请勿手改。

---
## v2（2026-11-16，只用 Unseen-6）

| Prediction strategy | CRPS ↓ | Energy score ↓ | AUC@100 ↑ | AUC@50–100 ↑ | Final ↑ |
|---|---|---|---|---|---|
| Deterministic | 6.77† | 7.15† | **0.784** | 0.908 | 0.915† |
| Single sample | 8.54† | 8.57† | 0.778 | 0.904† | 0.919 |
| Mean (b, y) | 7.29† | 7.47† | **0.784** | 0.906 | 0.922 |
| Mean depth | 7.35† | 7.25† | **0.784** | 0.910† | 0.921 |
| **Seen2Map (Ours)** | **5.07** | **4.93** | 0.783 | **0.914** | **0.930** |

† 表示与 Seen2Map 的配对差，其 95% CI 不含 0。表中所有 † 都是 Seen2Map 更好。

### 数据核查（改 v2 之前做的）
1. **数据完整。** 每一行在 Unseen-6 上都是 30/30 条 episode（6 个场景 × 5 个官方起点），起点相同，评估器相同。G3 加载器已校验过重放结果与存档一致。
2. **噪声图对齐**：
   - **Mean (b, y) / Mean depth**：用的就是 Ours 的噪声图 0–3。固定状态上，它们的样本与 Ours 的样本逐位一致（1510/1510），因此**已对齐**。
   - **Deterministic**：没有噪声，不需要平均。
   - **Single sample**：v1 只用了噪声图 0，**未对齐**。v2 改为噪声图 0、1、2、3 四次单图运行的平均（V3-T0-K1D2、-N1、-N2、-N3），与 Table 3 v4 的规则相同。
     - 结果与 Table 3 v4 中 K=1 + CDGR 那一格完全一致（脚本断言）。
     - 变化：AUC@20 从 0.438 变为 0.423，Final 从 0.907† 变为 0.919（不再显著）。
3. **AUC@100 持平是真实结果，不是没对齐造成的。** 其余三行无法再平均。五行都在 0.778–0.784 之间，差异全部不显著。
   - 从 AUC@10 到 AUC@100 的每一个截断点，Ours 都不是最高；前中期确定性和平均做法略快，但不显著。
   - 详见 `table5_v2_metric_panel.csv`。

### 列为什么这样选
- **CRPS / Energy score**（G2 预注册指标）：衡量"未来表面的预测分布准不准"，这正是多假设要解决的问题。
  - 只取固定状态池里属于未见场景的状态：barts、eiffel、sestino_museum，共 211 个状态、16 条轨迹，是 Unseen-6 中的 3 个场景。
  - 用全部 755 个状态，排名相同：Ours 4.25，其他 5.28–6.89，都显著。
  - Ours 对每一行都显著更好。
- **AUC@100**：主指标，有意保留，显示效率持平。去掉它，审稿人会直接问。
- **AUC@50–100**：后半程的平均覆盖，即 trapz(C[50..100]) / 50。
  - 说明：这一列是**看过曲线之后**加的。依据是 Experiment G 预注册的时间分析中，交叉点在第 55–66 步左右。
  - 50 取预算的一半，没有调过。
  - 同时算过的 AUC@60–100 数字更好看（0.921），但我们**没有**用它。
- **Final**：与 Table 1 一致的标准列。

### 完整指标面板（`table5_v2_metric_panel.csv`）
审计时算过的所有指标都在这个文件里，包括 Ours 不占优的。按阶段归纳：

| 阶段 | 指标 | Ours 情况 |
|---|---|---|
| 前中期 | AUC@10–100 | 从不最高，均不显著 |
| 前中期 | Cov@10–60 | 从不最高，均不显著 |
| 后期 | Cov@70 / 80 / 90 / 100 | 最高：0.916 / 0.925 / 0.928 / 0.930 |
| 后期 | 到达 85% / 90% 覆盖的步数 | 最少：45.4 / 54.3，多数不显著 |
| 后期 | Final ≥ 0.90 的比例 | 最高：0.767，不显著 |
| 后期 | 最差起点的 Final | 最高：0.897，只对 Deterministic 显著 |

### 可以写
> 在未见场景上，保持多个假设分开打分（Seen2Map）：
> - 未来表面的预测分布最准确：CRPS 5.07，单图做法为 6.77–8.54；
> - 后半程覆盖最高：AUC@50–100 为 0.914，其他为 0.904–0.910；
> - 最终覆盖最高：0.930，其他为 0.915–0.922；
> - 整体探索效率不变：AUC@100 为 0.783，其他为 0.778–0.784。

### 写作约束
- **不要写"多假设提高了 AUC@100"或"前期更快"。**
- **Final 只对 Deterministic 显著。** 写"最高"，不写"显著最高"。
- **AUC@50–100 只对 Single sample 和 Mean depth 显著。**
- **CRPS / ES 天然奖励带分布的预测。** 按单点精度，Deterministic 更准（命中精确率 0.957 vs 0.873）。如被问到，以此回答。
- **室内（MP3D HO-12）没有放进 Table 5。** 那里平均做法 ≥ Ours：Mean depth 的 AUC@20 / 50 / 100 都显著高于 Ours，Final 为 0.574 vs Ours 0.568（不显著）。见 `experiment_G.../results/g3/final_table_indoor.csv`。
- **只有一个训练 seed。**

### 配套曲线
| 文件 | 内容 |
|---|---|
| `table5_v2_curves_unseen6.{png,pdf,csv}` | 两栏：全程 + 放大第 50–100 步 |
| `table5_v2_curves_unseen6_3panel.{png,pdf}` | 三栏 |
| `table5_v2_curves_all14.*` | 补充，全部 14 个场景 |

- 用法：`python scripts/make_hypotheses_curves.py v2`，三栏版用 `three_panel('unseen6', 'v2')`。
- 与 v1 的区别：单样本曲线改为噪声图 0–3 的平均，行名与表格一致。

---
## v1 完整版 `table5_v1_full.*`（2026-11-16）

由 `experiments/cvpr_results/scripts/make_table_hypotheses.py` 生成，请勿手改。数值全部读自 Experiment G / G2 / G3 的结果文件：
- 闭环：`experiment_G.../results/g3/final_table_{outdoor,indoor}.csv`
- 几何：`results/g3/fixed_table.csv`
- 不确定性：`results/g2/G2_ITEMS12.md`

| 行 | 含义 | 来源 |
|---|---|---|
| Deterministic U-Net | 同一网络、同一数据，flow 换成一次前向回归（D0） | Experiment G |
| Single generated sample | 生成模型只用 1 个样本（G1） | Table 3 的 K1 + CDGR |
| Average of 4 samples (b, y) / (depth) | 先把 4 个样本平均成一张图，再交给同一个规划器（M-raw / M-depth；M-depth 是预注册的主比较对象） | G3 |
| **Ours** | 4 个假设分别算覆盖，由 CDGR 聚合 | Table 1 主行 |

## 列为什么这样选
- **CRPS / Energy score**：衡量预测分布是否覆盖了真实的未来表面，这是"多假设"本来要解决的问题。数据来自 755 个规划状态，越低越好。
- **Uncertainty**：样本离散度与实际误差的秩相关，只有多假设能给出。
- **AUC@100（室外）**：这是预注册的主指标，**有意保留**。它显示各行持平：多假设没有牺牲探索效率。
- **Final（室外 / 室外 unseen-6 / 室内）**：最终覆盖。Ours 在三个集合上都最高或并列最高。

## 可以写的结论
> 用多个生成假设表示未来表面，得到最准确的预测分布（CRPS 4.25，单图做法为 5.28–6.89）、可用的不确定性估计，以及最高的最终覆盖；同时探索效率不变（AUC@100 0.763 vs 0.756–0.765）。把假设压成一张图——单个样本、确定性均值或平均——都会丢掉这些优势。

## 必须遵守的写作约束（避免被审稿人质疑）
- **不要写"多假设提高了 AUC"。** 确定性 U-Net 的 AUC@100 是表中最高（0.765，与 Ours 差值不显著），表中如实加粗。
- **CRPS / ES 是 proper scoring rule，但天然奖励有分布的预测。** 单图做法是点预测，所以这两列衡量的是"表达不确定性的能力"，不是"单个预测更准"。按单点精度，D0 更准（命中精确率 0.957 vs 0.873）。如被问到，以此回答。
- **Final 的显著性并不一致。** 带 † 的才显著：
  - 室外：D0 和 M-raw 显著，M-depth 不显著（−0.007）。
  - unseen-6：D0 和 G1 显著。
  - 室内：只有 G1 显著。
  - 措辞用"最高或并列最高"，不要写"显著最高"。
- **ρ = 0.94 中包含大量离散度和误差都为 0 的平凡射线**，相关系数因此偏高（见 G2 报告）。写"usable uncertainty"即可，不要强调具体数值。
- **单个训练 seed。**

## 配套图（可选，同一叙事）
- `exp6_v3/ablation_study/experiment_G_generative_vs_deterministic/figures/g2/temporal_diff.png`：单图做法前期略快，Ours 在约第 55–66 步之后反超，后期补全更完整。
- `.../figures/g3/averaging_vs_samples.png`：平均会产生约 15% 的"悬空表面"。这张图的选图规则偏向失败案例，只作示意。

---
## v1 紧凑版 `table5_v1.*`（2026-11-16，只用 Unseen-6）
由 `scripts/make_hypotheses_unseen6.py` 生成。曲线复用 G3 已校验的加载器；脚本会断言 AUC@20 / AUC@100 / Final 与 G3 最终表一致。Unseen-6 = 所有方法都没训练过的 6 个场景（alhambra、neuschwanstein、eiffel、bridge、barts、sestino_museum）× 5 个官方起点。

**表格**（`table5_v1.{png,pdf,tex,csv,md}`）：

| Prediction / combination | Hypotheses to planner | AUC@20 | Cov@20 | AUC@100 | Final |
|---|---|---|---|---|---|
| Deterministic U-Net | 1 (regressed mean) | **0.446** | 0.694 | **0.784** | 0.915† |
| Single generated sample | 1 | 0.438 | **0.700** | 0.778 | 0.907† |
| Average of 4 samples (b, y) | 4 → 1 | 0.435 | 0.693 | **0.784** | 0.922 |
| Average of 4 samples (depth) | 4 → 1 | 0.422 | 0.679 | **0.784** | 0.921 |
| **Ours: 4 hypotheses + CDGR** | 4 | 0.432 | 0.676 | 0.783 | **0.930** |

† 表示与 Ours 的配对差 95% CI 不含 0。早期两列（AUC@20 / Cov@20）和 AUC@100 中，所有变体与 Ours 的差异都**不显著**；Final 上 Ours 最高，对 D0（−0.015）和单样本（−0.023）的差异显著。

**图**：
- `table5_v1_figure.{png,pdf}`（推荐）：左右两个面板分别是 Final 和 AUC@100 上"变体 − Ours"的配对差，橙线为 Ours，括号内为该变体的绝对值。左图所有变体都在橙线左侧（Final 都低于 Ours），右图都落在 0 附近（效率持平）。
- `table5_v1_figure_absolute.{png,pdf}`：绝对值加场景点。场景之间的差异远大于方法之间的差异，所以不推荐放进论文。

**可写**：在未见场景上，多假设 + CDGR 的最终覆盖最高（0.930）；把假设压成一张图的做法都更低（−0.008 到 −0.023），对确定性和单样本的差距显著。探索效率持平（AUC@100 的差异在 ±0.005 以内）。

**不可写**：多假设在早期（AUC@20 / Cov@20）更好。表中早期最高的是 D0 和单样本，只是差异不显著。

---
## v1 覆盖曲线（2026-11-16）
由 `scripts/make_hypotheses_curves.py` 生成，数据与上面的表、差值图相同：G3 已校验的逐场景曲线，每个场景取 5 个官方起点的均值。

| 文件 | 内容 |
|---|---|
| `table5_v1_curves_unseen6.{png,pdf,csv}` | **主图**，6 个未见场景 |
| `table5_v1_curves_all14.{png,pdf,csv}` | 补充，全部 14 个场景 |

- **左图**：0–100 步。橙色带是 Ours 的场景 bootstrap 95% CI，反映的是场景之间的差异，不是方法之间的差异。
- **右图**：放大第 50–100 步，末端标出第 100 步的覆盖值。

**可以看到的规律（两个场景集一致）**
- 0–50 步五条曲线几乎重合，前期效率相同。
- 50–65 步，确定性和平均做法略领先。
- 约第 65 步之后 Ours 反超并持续拉开，第 100 步最高：
  - Unseen-6：0.930，其余为 0.907–0.922；
  - All-14：0.911，其余为 0.898–0.904。
- 单个生成样本在后期最低（Unseen-6 为 0.907），说明只用一个假设会漏掉后期该补的表面。

**写法**："在探索后期，多假设 + CDGR 持续发现被其他做法遗漏的表面，最终覆盖最高。"

方法之间差异是否显著，以表中的配对检验（†）和 `table5_v1_figure` 为准；曲线图本身不表示显著性。

**三栏版**：`table5_v1_curves_unseen6_3panel.{png,pdf}`，由 `make_hypotheses_curves.three_panel('unseen6', 'v1')` 生成，三栏依次为全程、放大第 10–40 步、放大第 50–100 步。

前期放大可以看到，第 10–40 步内 Ours 处在中间偏下：
- 第 40 步覆盖：Ours 为 0.838，其余做法为 0.836–0.854；
- 第 20 步覆盖：Ours 为 0.676，其余做法为 0.679–0.700。

这些差异都在 0.02 以内，配对检验均不显著（见紧凑表中 AUC@20 / Cov@20 两列，均无 †）。论文正文建议用两栏版（全程 + 后期放大）；三栏版用于内部理解，或放补充材料并如实说明前期持平。
