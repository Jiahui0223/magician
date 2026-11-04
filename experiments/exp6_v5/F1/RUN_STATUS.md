# RUN_STATUS — EXP6 V5 / F1（只写有依据的状态）

## 此刻在跑什么（2026-11-03 21:50）
| 编号 | 任务 | 节点 / GPU | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|
| F1-P1.T-A | F1-A 训练 5k（torchrun 4 卡） | gpu115 / 0–3（作业 3443137） | `work_dirs/exp6_v5/F1/logs/train_A.log` | 21:29 | 约 22:05 |
| F1-P1.T-B | F1-B 训练 5k（torchrun 4 卡） | gpu114 / 0–3（作业 3461041） | `logs/train_B.log` | 21:29 | 约 22:30 |
| F1-P3.G3 | G3 队列（Dev-3 × 5 个起点 × A2k / A5k / B2k / B5k，共 60 个 episode；对应 checkpoint 出现后自动开始） | gpu115 每卡 2 个 worker + gpu114 每卡 1 个 | `g3/logs/queue_*.log` | 21:47（A2k） | B5k 训练完成后约 30 分钟 |

## 编号进度
| 编号 | 项 | 状态 | 关键数字 |
|---|---|---|---|
| F1-P0 | 检查 Table 1 负载 | ✅ | Table 1 已于 21:00 全部完成，没有负载冲突 |
| F1-P1.0 | 审计 B0 checkpoint、继承、配置 | ✅ | sha `343f41ab…`；B 分支 382 个张量 `torch.equal`；见 `CHECKPOINT_AUDIT.md` |
| F1-P1.1 | B0 启动链回归（barts_s0） | ✅ | 执行动作与官方覆盖曲线逐位一致 |
| F1-P1.2 | 条件输出统计与 out_scale_init 规则 | ✅ | 0.02753 / 0.58191 = 0.0473（96 个视角） |
| F1-P1.3 | 短程冒烟 A / B（各 300 步，B 为 150 + 150） | ✅ | 有限值；B 第一阶段损失 0.24 → 0.17，切换到联合阶段时没有发散；300 / 300 步摘要相同 |
| F1-P1.4 | G1/G2 冒烟（B0，1 个 episode） | ✅ | 与 E2 记录逐位一致，332 s / episode |
| F1-P1.T-A | F1-A 正式训练 | 🔄 | 0.26 s/步，14.3 GB / rank |
| F1-P1.T-B | F1-B 正式训练 | 🔄 | 第一阶段约 0.6 s/步 |
| F1-P1.V3 | V3 历史整理（只读） | ✅ | `V3_HISTORY.md` |
| F1-P2 | G1/G2（Dev-3 的 B0 / B2 两层，5 个模型） | ⏳ | 等 A5k / B5k |
| F1-P3.pilot | G3 Pilot（起点 0, 1） | 🔄 | — |
| F1-P3.ext | G3 扩展（起点 2–4） | 🔄 | — |
| F1-P3.T1 | 正式 Table 1 评估（All-14） | ⬜ | 在 Dev-3 选定 checkpoint 之后进行 |
