# EXP5 Task 3 — Full-Depth P1 Fusion Ablation（只执行 Experiment 1）

| 文件 | 内容 |
|---|---|
| `protocol.md` | 协议 v1.0，正式运行前冻结，含 Go / No-Go 标准 |
| `DEVIATIONS.md` | 偏差与工程记录 |
| `reports/T3_E1_AUDIT.md` | Stage 0 源码审计 |
| `reports/T3_E1_GEOMETRY.md` | Stage 1 几何诊断与对比图说明 |
| `reports/T3_E1_CLOSED_LOOP.md` | Stage 2 闭环结果与 Stage 3 固定状态归因（B0–B3） |
| `reports/T3_E1_B2_CLOSED_LOOP.md` | B2 闭环验证（稠密输入 / 稀疏评分） |
| `reports/T3_E1_DECISION.md` | 判定与下一步建议（含 B2 附录） |
| `scripts/p1_full.py` | FullDepth-P1：只重写 `integrate_record`；`patch()` 由环境变量 `T3_P1_MODE` 控制 |
| `scripts/t32_*_planner.py`、`t33_*_sim.py` | 规划器包装与闭环启动脚本（原样运行 Task 2 脚本） |
| `scripts/t31_geometry.py`、`t31_viz.py` | 固定状态几何诊断与对比图 |
| `scripts/t34_analyze.py`、`t35_out_cov.py` | 汇总分析；室外覆盖重算（原样运行 Task 2 的 `t25_out_cov`） |
| `analysis/` | 表格、汇总 JSON、覆盖曲线 |
| `geometry_diagnostics/figures/` | 稀疏 / 全深度 P1 对比图 |

大文件（闭环轨迹、几何诊断数据、覆盖重算）在 `work_dirs/exp5_future_view/task3_full_depth_p1/`。

## 状态
| 步骤 | 状态 |
|---|---|
| Stage 0 审计 | DONE |
| Stage 1 冒烟 + 几何诊断 | DONE：400 个固定状态；B0 逐位复现 |
| Stage 2 闭环 B0 / B1 | DONE：80 / 80 条；B0 与 Task 2 逐位相同 |
| Stage 3 固定状态归因 B2 / B3 | DONE |
| B2 闭环（协议附录 A） | DONE：40 / 40 条；t = 0 分数与固定状态 B2 逐位相同。B2 − B0：MP3D ΔAUC +0.012（4/5 胜）、ΔFinal +0.061（CI 不含 0），Macarons++ ΔAUC +0.001、ΔFinal +0.022 → 按冻结标准 No-Go（无退化；MP3D 只被 oLBM −0.040 挡住） |
| 判定 | **No-Go for direct replacement**：MP3D ΔAUC +0.009（持平），Macarons++ ΔAUC −0.038（退化） |
| E1.2 BAND 密度消融（`density_ablation/`） | DONE（固定状态）：50% / 75% 未通过冻结的闭环筛选，未跑闭环；E2 推荐 D100 |
| Experiment 2 Pilot（`e2_pilot/`） | 训练完成（E2-D100、E2-S10 各 5k 步）；G1 通过，G2 不通过；G3（探索性，用户指示）→ Conditional Go：MP3D 全深度微调有意义（+0.017 vs E2-S10，+0.043 vs B0），Macarons++ 两个微调组都不如冻结模型 |

## 变更记录
- 2026-10-28：创建并完成 Experiment 1。
- 2026-10-28：用户批准 B2 闭环验证；协议附录 A 冻结后运行，已完成（偏差 #4–#6）。
- 2026-10-28：E1.2 BAND 密度消融（任务书 Experiment 1.2），固定状态完成（偏差 #7–#9）；E2 Pilot 方案已提交。

- 2026-10-29：E2 Pilot（用户批准）：审计、协议冻结、D100 数据重建、两组并行训练、G1 / G2；G3 未运行。
