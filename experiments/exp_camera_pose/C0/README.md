# C0 —— Original Neighbor Baseline
官方邻居 → 合法过滤 → B2 / E2-D100 / V4 / V5 → 未来深度（K = 4）→ S10 DistP1 → 原选择规则。完全不改。
- 室外 pilot C0：复用历史 Task 3 / E2 G3 的 episode。Stage A 回归：新入口 λ 0 在 barts s0 上 100 步与历史逐位一致。
- 室内 pilot C0：同一入口在 λ 0 下新跑（VALSEL 4 个 episode），产物在 `work_dirs/exp_camera_pose/closed_loop/{B2,E2D}_C0/`。
- 固定状态 C0：历史逐候选 DistP1。指标与 EXP6 各报告的 G2 数字完全一致。
- V4 / V5 的 C0（新入口）：室外 6 个 episode 的覆盖曲线与历史 G3 逐位相同（6/6、6/6）；室内 VALSEL 为新跑。
