# Overnight queue — STATUS（2026-11-03 04:25）

图例：COMPLETED / RUNNING / QUEUED / FAILED / BLOCKED / SKIPPED

| 阶段 | 状态 | 说明 |
|---|---|---|
| S0 Density A/B 固定状态 G1 / G2 | COMPLETED | 365/365 个状态，3195 项回归检查全部逐位一致 |
| S0 Density A/B G3（7 组新配置） | RUNNING | 04:16 时完成 213/280 |
| S0 报告（G3 / FINAL / 训练建议） | QUEUED | 等 G3 结束 |
| S1-a Refresh 固定状态 | QUEUED | `chain_stage1.sh` 等 Density A/B 的 G3 结束后自动开始 |
| S1-b Refresh G3 R2 / R1 | QUEUED | 门槛：R10 回归全部通过 |
| S1-c Refresh G3 R5 | QUEUED | — |
| S2 训练密度判定 | BLOCKED | 等 Density A/B 的 G3 分析（规则见 OVERNIGHT_PLAN §4） |
| S2 训练配方审计 | RUNNING | 后台 agent |
| S2 训练 V2 / V4 / V4C / V5 / V3-T1 | QUEUED | 依次：V2 → V4 → V4C / V5 → V3-T1 |
| Mixed 20k | SKIPPED | 本夜不启动（用户要求） |

作业：3443137（gpu115，4 × L40S，11-04 15:11 到期）运行中；3461041 是用户的 vscode-dev，排队中，没有动；3460841 已取消（冗余）。
