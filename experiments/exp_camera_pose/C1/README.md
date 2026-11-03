# C1 —— Frontier-Aware Pose Selection（同一动作集合内重新排序）
`Score_C1 = D/maxD + λ·F/maxF`。F 由 `scripts/frontier.py` 计算（规划器 D100 视图中 FREE→UNKNOWN 进入点所在的不同粗格数，不用 GT）。λ 按预注册规则选出（B2 0.1、E2-D100 0.1、V4 0、V5 0），见 `../configs/c1_lambda.json`。
- 固定状态：选择改变 3–5%，一步收益变化 ≤ 0.03 pp，无系统提升（`../reports/fixed_state.md`）。
- pilot：4 个组合的 mean ΔAUC 为 +0.002 / −0.019 / −0.020 / −0.029，门槛全部未通过（`../reports/pilot.md`）。
- 结论：NO-GO；没有运行完整 G3。
