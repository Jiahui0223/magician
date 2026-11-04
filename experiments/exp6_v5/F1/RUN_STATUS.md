# RUN_STATUS — EXP6 V5 / F1（只写有依据的状态）

## 此刻在跑什么（2026-11-03 23:45）
- 无。F1 已全部完成，结论见 `reports/FINAL_REPORT.md`。

## 编号进度
| 编号 | 项 | 状态 | 关键数字 |
|---|---|---|---|
| F1-P0 | 检查 Table 1 负载 | ✅ | Table 1 已于 21:00 全部完成，没有负载冲突 |
| F1-P1.0 | 审计 B0 checkpoint、继承、配置 | ✅ | sha `343f41ab…`；B 分支 382 个张量 `torch.equal`；见 `CHECKPOINT_AUDIT.md` |
| F1-P1.1 | B0 启动链回归（barts_s0） | ✅ | 执行动作与官方覆盖曲线逐位一致 |
| F1-P1.2 | 条件输出统计与 out_scale_init 规则 | ✅ | 0.02753 / 0.58191 = 0.0473（96 个视角） |
| F1-P1.3 | 短程冒烟 A / B（各 300 步，B 为 150 + 150） | ✅ | 有限值；B 第一阶段损失 0.24 → 0.17，切换到联合阶段时没有发散；300 / 300 步摘要相同 |
| F1-P1.4 | G1/G2 冒烟（B0，1 个 episode） | ✅ | 与 E2 记录逐位一致，332 s / episode |
| F1-P1.T-A | F1-A 正式训练 | ✅ | 21:29–22:18；0.35 s/步，14.3 GB / rank；EMA 0 / 1k / 2k / 3k / 4k / 5k |
| F1-P1.T-B | F1-B 正式训练 | ✅ | 21:29–22:47；第一阶段 0.49 s/步、第二阶段 0.78 s/步；两分支 5000 / 5000 步样本流摘要相同 |
| F1-P1.V3 | V3 历史整理（只读） | ✅ | `V3_HISTORY.md` |
| F1-P2 | G1/G2（Dev-3 的 B0 / B2 两层，5 个模型） | ✅ | 275 个状态，22:47–22:55；B0 与 E2 记录逐位一致；`reports/G1.md`、`reports/G2.md` |
| F1-P3.pilot | G3 Pilot（起点 0, 1） | ✅ | 4 个模型 × 6 个 episode，全部有效；`reports/G3_PILOT.md` |
| F1-P3.ext | G3 扩展（起点 2–4） | ✅ | 每个模型 15 / 15，共 60 个有效，0 次失败；预注册规则选出 A2k / B5k；`reports/G3_EXTENDED.md` |
| F1-P3.cost | 受控计时（barts_s0，每个模型独占一张卡） | ✅ | B0 0.541 → B5k 0.588 s/步；4 个重跑与记录逐位一致；`reports/RUNTIME.md` |
| F1-P3.T1 | 正式 Table 1 评估（All-14） | ✅ | 22:58–23:38；F1-CONT-v1@2k 与 F1-GQ-v1@5k 各 70 / 70；Dev-3 部分与 F1 记录 15 / 15 逐位一致；dunnottar 上 5 次 OOM 重试成功；`reports/TABLE1_FOLLOWUP.md` |
| F1-P3.report | 报告与导出 | ✅ | `reports/FINAL_REPORT.md`；`table1_export/f1_model_registry_entry.yaml` |
