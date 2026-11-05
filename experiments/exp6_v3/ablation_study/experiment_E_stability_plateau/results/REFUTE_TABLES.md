# REFUTE_TABLES — “反证记忆”离线重打分（`scripts/refute_analysis.py`；诊断，不是闭环结果）

- episode：26 条（alhambra H0 s0, colosseum D1 s0, colosseum D1 s1, colosseum D1 s2, colosseum D1 s3, colosseum D1 s4, colosseum H0 s0, colosseum H0 s1, colosseum H0 s2, colosseum H0 s3, colosseum H0 s4, colosseum H3 s0, colosseum H3 s1, colosseum H3 s2, colosseum H3 s3, colosseum H3 s4, colosseum YBEST-H3 s0, colosseum YBEST-H3 s1, colosseum YBEST-H3 s2, colosseum YBEST-H3 s3, colosseum YBEST-H3 s4, pisa D1 s1, pisa H0 s1, pisa H0 s4, pisa H3 s1, pisa H3 s4）。
- 原始 = 规划器的预测均值（pred_value）；并集 = 同一预测按样本并集（每 cell 取最大权重）的值；反证 = 并集值去掉被此前已执行视图“看过却没看到”的 cell。
- Top-1 与所选真实增益都对照 GT 一步覆盖增益（pp）；GT 只用于评分，不参与重打分。

| 步集合 | 步数 | 最佳一步增益 pp | Top-1：原始 / 并集 / 反证 | 所选真实增益 pp：原始 / 并集 / 反证 | ρ：原始 / 并集 / 反证 | 执行视图预测 cell 的持续率 | 执行视图预测值中被反证的比例 |
|---|---|---|---|---|---|---|---|
| colosseum 低平台 episode 的平台阶段 | 674 | 0.12 | 0.35 / 0.32 / 0.45 | 0.05 / 0.05 / 0.08 | 0.14 / 0.14 / 0.28 | 0.92 | 0.88 |
| colosseum 低平台 episode 的平台前 | 126 | 4.03 | 0.37 / 0.45 / 0.53 | 3.33 / 3.42 / 3.57 | 0.63 / 0.67 / 0.74 | 0.23 | 0.16 |
| colosseum 成功 episode t < 20 | 240 | 3.36 | 0.47 / 0.52 / 0.59 | 2.79 / 2.94 / 3.03 | 0.64 / 0.66 / 0.72 | 0.29 | 0.20 |
| colosseum 成功 episode t ≥ 20 | 960 | 0.40 | 0.35 / 0.37 / 0.46 | 0.28 / 0.29 / 0.33 | 0.26 / 0.28 / 0.43 | 0.79 | 0.74 |
| pisa / alhambra t < 20 | 120 | 2.92 | 0.52 / 0.53 / 0.52 | 2.56 / 2.52 / 2.52 | 0.67 / 0.73 / 0.79 | 0.15 | 0.11 |
| pisa / alhambra t ≥ 20 | 480 | 0.44 | 0.40 / 0.43 / 0.59 | 0.33 / 0.36 / 0.38 | 0.22 / 0.26 / 0.42 | 0.66 | 0.54 |
| 全部 t < 20 | 520 | 3.23 | 0.45 / 0.49 / 0.55 | 2.70 / 2.79 / 2.87 | 0.62 / 0.65 / 0.72 | 0.27 | 0.20 |
| 全部 t ≥ 20 | 2080 | 0.32 | 0.36 / 0.37 / 0.49 | 0.22 / 0.23 / 0.26 | 0.21 / 0.23 / 0.38 | 0.80 | 0.74 |
