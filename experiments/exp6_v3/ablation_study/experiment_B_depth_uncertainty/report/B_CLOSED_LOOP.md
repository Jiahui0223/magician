# Experiment B — 闭环 G3（**Not Run**）

预注册 Gate（`B_DECISION.md`）在 dev 集（MP3D Pool FX VALSEL、Macarons++ Pool CL Dev-3）上对 (B2, T1) × (P1, P2) 全部 FAIL，按协议 §3 不运行正式 G3。协议允许的 ≤ 2 个 exploratory 闭环也未运行：固定状态上冻结 λ = 1.0 的 P2 在两个数据集都没有改善（MP3D T1 +40%、Macarons++ B2 +20%），闭环不会给出可用的正结论；唯一的正信号（T1 在 Macarons++ 上 λ = 0.5，−0.023 pp）属于非冻结的敏感性设置，若要验证需另行预注册。

既有 G3 对照表（MAG-G / MAG-B / B0 / B2 / E2-D100 / V3-T0 / V3-T1，Task 2 / Task 3 / EXP6 的同协议闭环）见 `../../reports/V3_ABLATION_COMPARISON_TABLES.md` 表 3；MAG 组不是 Flow K=4 模型，不参与 DepthU 计算。
