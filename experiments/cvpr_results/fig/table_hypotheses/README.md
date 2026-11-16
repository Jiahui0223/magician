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
