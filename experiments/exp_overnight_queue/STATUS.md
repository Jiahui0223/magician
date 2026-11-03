# Overnight queue — STATUS（2026-11-03 05:05）

图例：COMPLETED / RUNNING / QUEUED / FAILED / BLOCKED / SKIPPED

| 阶段 | 状态 | 说明 |
|---|---|---|
| S0 Density A/B 固定状态 G1 / G2 | COMPLETED | 365/365 个状态，3195 项回归检查全部逐位一致 |
| S0 Density A/B G3（7 组新配置） | COMPLETED | 280/280 个 episode，完整性检查 0 异常 |
| S0 报告（G3 / FINAL / 训练建议） | COMPLETED | 64b6d9a |
| S1-a Refresh 固定状态 | COMPLETED | 39/39 个 episode，R10 回归 1460/1460 逐位一致；结果见 G1_G2_RESULTS.md |
| S1-b Refresh G3 R2 / R1 | RUNNING | 05:00 开始，8 个 worker；先做 R10 复现（2 个 episode） |
| S1-c Refresh G3 R5 | QUEUED | — |
| S2 训练密度判定 | COMPLETED | 按预注册规则选 S10，证据中等；D100 分支保留 |
| S2 训练配方审计 | COMPLETED | TRAINING_PROTOCOL_DRAFT + UNIFIED_PROTOCOL（2ee6aae） |
| S2 V2-S10 | SKIPPED（复用历史 V2 Original，配方逐项一致） | 固定状态评估与 V4 训练并行 |
| S2 V4-S10 → V4C / V5 → S10 缓存 → V3-T1 | QUEUED | `chain_stage2.sh`，等 Stage 1 结束 |
| Mixed 20k | SKIPPED | 本夜不启动（用户要求） |

作业：3443137（gpu115，4 × L40S，11-04 15:11 到期）运行中；3461041 是用户的 vscode-dev，排队中，没有动；3460841 已取消（冗余）。
