# EXP6 V3 Task 2 — 资源记录

| 日期 | 部分 | 资源 | 备注 |
|---|---|---|---|
| 2026-10-31 | Phase 0 审计、协议、单测（7 passed） | CPU | |
| 2026-10-31 | E1 固定状态打分（`e1_score_fixed.py`：1,676 状态 × {GT + 4 模型 × K=4} × 16 + 3 变体；稀疏 + D100 融合重放，无模型推理） | 8 × L40S（gpu117 3429774 + gpu115 3435033），17:07 → 17:43 + 7 状态补算 ≈ **4.9 GPU-h**；1.2–2 s / 状态；峰值 14.2 GB | 输出 `work_dirs/exp6_v3/task2/exp1/scores/`（JSON，约 0.3 GB） |
| 2026-10-31 | E1 / E2 G0（R0 回放 + 两个变体冒烟） | 4 × L40S，2 min | |
| 2026-10-31 | E2 R1 / R2 离线 counterfactual | CPU，各 10–15 min | |
| 2026-10-31 | E2 R3 两步候选 counterfactual（368 状态 × 34 个扩展位姿渲染） | 4 × L40S 约 10 min ≈ 0.4 GPU-h（含重放） | `work_dirs/exp6_v3/task2/exp2/r3/` |
| 2026-10-31 | G3 闭环：室外 B2 + S1c / T1 + S1c 各 30 条（192 / 197 s / 条）+ 覆盖率回放 60 条；MP3D T1 + S2S3 / B2 + S2S3 / B2 + bonus / B2 + trigger 各 10 条（472–493 s / 条） | 8 × L40S，17:51 → 19:00 ≈ **9.6 GPU-h**（室外 3.3 + 回放 1.0 + 室内 5.4）；子进程峰值 19.6 GB（室内）；新评分器每步 0.5 s（室内）/ 0.3 s（室外） | 输出 `work_dirs/exp6_v3/task2/g3/`（12 GB 含 svc） |
| 2026-10-31 | 分析与报告（`e1_analyze`、`e2_analyze`、`e2_failure_cases`、`t2_g3_analyze`） | CPU，各 1–3 min | |
| **合计** | **Task 2** | **≈ 15 GPU-h（L40S）+ CPU**；无训练；既有分配，无新 Slurm 申请，未取消任何作业 | 2026-10-31 17:00 → 19:10 |
