# EXP5 Task 0（RESUMED）状态表

> 计划：`TASK0_RESTART_PLAN.md`；协议：`protocol.md`；偏差：`DEVIATIONS.md`；旧草案 `TASK0_PLAN_DRAFT.md` 保留。
> 数据：`/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans`（只放数据集；10-23 从 /bsushare 迁来，原因是 2 TB 上限）；工作目录：`work_dirs/exp5_future_view/task0`。

| 阶段 | 状态 | 时间 | GPU-h | 报告 |
|---|---|---|---|---|
| T0.1 恢复 | **DONE**：决策记录标为 RESUMED，目录已建 | 2026-10-23 13:20 | 0 | `../decision_log.md` |
| T0.2 下载 | **DONE**：完整的 v1 扫描发布，90 个房屋 × 17 种类型，1,530 个 zip（1.17 TB），全部经 sha256 / CRC 校验；7 种几何类型已解压；已从 /bsushare 迁到 scratch 并逐字节校验 | 13:21–23:25 | 0 | `reports/T0.2_data_report.md` |
| T0.3 几何验证 | **DONE**（M 协议）：<br>- 相机约定和单位已验证；<br>- textured mesh 作为 GT；<br>- 室内传感器与冻结传感器逐位等价；<br>- 22 个楼层、44 条参考轨迹；<br>- P1 表面精度合格；<br>- 规划侧几何与 GT 一致 | 13:22–14:30 | ≈4 | `reports/T0.3_geometry_report.md` |
| T0.4 固定状态零样本 | **DONE**（协议 v1.0 / v1.1）：M 157 个状态、R 157 个（世界模型 70 个），全部完成，无最终失败。世界模型的 CRPS 更好（24–26 对 V2 44），V2 的 Brier 更好；一步排序相当，DistU 最好；T1-C 在所有指标上不差于 Frozen；R ≈ M | 10-23 14:12 – 10-24 18:20 | ≈109 | `reports/T0.4_report.md` |
| T0.5 室内 / 室外对照 | **DONE**（协议 v1.2）：Dev-5 的 85 个状态。室内不比室外更好：F@2 相当，精度更低，误墙率高 2–3 倍；V2 在 barts / sestino 上弱 | 10-23 14:27 – 10-24 18:20 | ≈18 | `reports/T0.5_report.md` |
| T0.6 室内闭环 | **DONE**（协议 v1.3）：22 × 4 条 episode 全部完成。AUC@100：V2K4 0.467、V2K1 0.472、参考 0.350、Frozen 0.331、T1-C 0.311；世界模型约 68% 的步数回到已访问位姿 | 10-23 15:10 – 10-24 16:41 | 98.1（另有中断 / 重复 ≤ 7） | `reports/T0.6_report.md` |
| T0.7 最终报告 | **DONE**：8 个问题的回答；事后诊断（幻觉增益，非预注册）；后续方向 A / B / C 等你决定；分析代码快照 `W/snapshots/T0.7_analysis/` | 10-24 18:20 – 18:50 | ≈0.5 | `TASK0_FINAL_REPORT.md` |
| T0.8 现有轨迹统一指标复算 | **DONE**：110 条 episode 的 C_obs 复现完全一致；单位核对没有错误（ε = 0.08 模型单位 = 6.0–12.8 cm）；Comp@5cm：V2 0.60、随机参考 0.49、世界模型 0.40–0.41 | 10-24 21:00–22:15 | ≈3.7（共享 GPU） | `reports/T0.8_metric_report.md` |
| T0.9 MAGICIAN 官方复现 | **DONE**：论文协议无法用发布代码复现；发布代码协议 15 / 15 次运行成功（3 个米制场景 150 个位姿时 Comp@5cm 0.839 / 4.67 cm） | 10-24 21:30–23:15 | ≈5.8 | `reports/T0.9_magician_reproduction.md` |
| T0.10 同协议对照 | **DONE**：66 / 66 条 episode。V2K4 ≈ MAG-B（AUC 0.467 / 0.475）；beam +0.021；V2K4-NV 最好（0.509） | 10-24 22:20 – 10-25 01:40 | ≈6.1 | `reports/T0.10_fair_comparison.md` |
| T0.11 统一分析 | **DONE**：6 个问题的回答；`TASK0_FINAL_REPORT.md` 追加 §14；快照 `W/magician_comparison/snapshots/T0.11/` | 10-25 | — | `reports/T0.11_final_comparison.md` |
| T0.12 网格动作空间对齐（你 10-25 确认的设计） | **DONE**：50 / 50 条 episode。AUC@100：MAG-B 0.254、MAG-G 0.238、V2K4-NV 0.224、V2K4 0.214、随机 0.133；V2 与 MAGICIAN 的比较 CI 都含 0，差距主要来自 TbHJ；150 步时覆盖率相同；官方规则下相机会飞到建筑外 | 10-25 18:41–20:18 | ≈7.3 | `reports/T0.12_grid_alignment.md` |
| T0.12-C 离线诊断：V2 为什么不往远处走（你 10-25 要求） | **DONE**：原因是 V2 的规划区域只有约 7.8 m，每 10 步跟着相机挪一次，第 50 步以后框内只剩约 6% 的工作；整层探针下，同一个 V2 会选择往远处走；MAGICIAN 的优势在打分覆盖整层，一步判断并不更准 | 10-25 20:54–21:44 | ≈3.3 | `reports/T0.12C_far_exploration_diagnosis.md`；总结 `reports/T0.12_summary.md` |
