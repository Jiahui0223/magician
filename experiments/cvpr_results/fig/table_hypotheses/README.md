# Table：Why multiple generative hypotheses?（2026-11-16）

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
## 紧凑版（2026-11-16，只用 Unseen-6）
由 `scripts/make_hypotheses_unseen6.py` 生成。曲线复用 G3 已校验的加载器；脚本会断言 AUC@20 / AUC@100 / Final 与 G3 最终表一致。Unseen-6 = 所有方法都没训练过的 6 个场景（alhambra、neuschwanstein、eiffel、bridge、barts、sestino_museum）× 5 个官方起点。

**表格**（`unseen6_table.{png,pdf,tex,csv,md}`）：

| Prediction / combination | Hypotheses to planner | AUC@20 | Cov@20 | AUC@100 | Final |
|---|---|---|---|---|---|
| Deterministic U-Net | 1 (regressed mean) | **0.446** | 0.694 | **0.784** | 0.915† |
| Single generated sample | 1 | 0.438 | **0.700** | 0.778 | 0.907† |
| Average of 4 samples (b, y) | 4 → 1 | 0.435 | 0.693 | **0.784** | 0.922 |
| Average of 4 samples (depth) | 4 → 1 | 0.422 | 0.679 | **0.784** | 0.921 |
| **Ours: 4 hypotheses + CDGR** | 4 | 0.432 | 0.676 | 0.783 | **0.930** |

† 表示与 Ours 的配对差 95% CI 不含 0。早期两列（AUC@20 / Cov@20）和 AUC@100 中，所有变体与 Ours 的差异都**不显著**；Final 上 Ours 最高，对 D0（−0.015）和单样本（−0.023）的差异显著。

**图**：
- `unseen6_figure.{png,pdf}`（推荐）：左右两个面板分别是 Final 和 AUC@100 上"变体 − Ours"的配对差，橙线为 Ours，括号内为该变体的绝对值。左图所有变体都在橙线左侧（Final 都低于 Ours），右图都落在 0 附近（效率持平）。
- `unseen6_figure_absolute.{png,pdf}`：绝对值加场景点。场景之间的差异远大于方法之间的差异，所以不推荐放进论文。

**可写**：在未见场景上，多假设 + CDGR 的最终覆盖最高（0.930）；把假设压成一张图的做法都更低（−0.008 到 −0.023），对确定性和单样本的差距显著。探索效率持平（AUC@100 的差异在 ±0.005 以内）。

**不可写**：多假设在早期（AUC@20 / Cov@20）更好。表中早期最高的是 D0 和单样本，只是差异不显著。

---
## 覆盖曲线（2026-11-16）
由 `scripts/make_hypotheses_curves.py` 生成，数据与上面的表、差值图相同：G3 已校验的逐场景曲线，每个场景取 5 个官方起点的均值。

| 文件 | 内容 |
|---|---|
| `curves_unseen6.{png,pdf,csv}` | **主图**，6 个未见场景 |
| `curves_all14.{png,pdf,csv}` | 补充，全部 14 个场景 |

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

方法之间差异是否显著，以表中的配对检验（†）和 `unseen6_figure` 为准；曲线图本身不表示显著性。
