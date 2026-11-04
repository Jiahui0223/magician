# VALIDATION_G1_G2 — Y 的固定验证集（VALSEL / S-c；自动生成）

**指标**：FM 损失越低越好；fair ES 是样本质量（越低越好，同时也是原 V3 的 checkpoint 选择指标）；Brier / false wall 针对命中与无命中。数据来自训练器内置评估（EMA），不是 Test。

| step | FM in_val | FM out_sc | fair ES in_val | fair ES out_sc | Brier out_sc | false wall out_sc | 选择分数（相对 5K） |
|---|---|---|---|---|---|---|---|
| 40000 (父) | 0.0900 | 0.0272 | 6.425 | 6.482 | 0.0291 | 0.0172 | 1.395 |
| 45000 | 0.0898 | 0.0270 | 6.327 | 6.414 | 0.0293 | 0.0165 | 1.377 |

*G2 固定状态诊断本轮没有运行（预算优先给 G3）；Y 对决策的影响以 H0 下的 G3 闭环为准。*
