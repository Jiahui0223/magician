# H1_ANALYSIS — Coverage-Aligned Scoring（要点；完整分析见 FINAL_REPORT §2 Q1）

- **定义**：已见半径从 ±1 个 P1 体素（S10 `band_dil`）改为 eps_cov = 1.0，按智能体自己的 S10 观测点判断；去重尺度从 0.011 模型单位改为 1.0 运行时单位的覆盖格子；w(z)、z0、K 均值都不变。
- **G2**（Dev-3，275 个状态）：
  - 所选真实增益 1.459 vs 1.400 pp（t ≤ 20 时 2.880 vs 2.756）；
  - Top-1 0.556 vs 0.549（t ≤ 20 时 0.520 vs 0.472）；
  - 36% 的状态选择改变。
- **G3**（All-14，H1 − H0）：ΔAUC@20 +0.013 [−0.004, +0.029] 9/5；ΔAUC@100 −0.001；ΔFinal −0.009。
  - Test-3 的 ΔAUC@100 +0.023 ✱；
  - Train-6 的 ΔFinal −0.014 ✱。
- **副作用**：全零回退步从 1.0% 增加到 7.4%，集中在已接近饱和的 barts；colosseum 的 Final 0.764，H0 为 0.824。
- 来源：`G2_FIXED_STATE.md`、`G3_*.md`、`results/decisions.csv`。
