# RESOURCE_AND_FAILURE_LOG — 本夜资源与故障记录

## 资源
| 节点 | Slurm 作业 | GPU | CPU | 结束时间 | 本夜用途 |
|---|---|---|---|---|---|
| gpu115 | 3443137 | 4 × L40S | 24 | 2026-11-04 15:11 | D 的 Dev 队列（04:20–08:56，每卡 2 个 worker）→ 主 All-14 队列（08:58–10:00）→ dunnottar 独占 → YBEST-H3 补齐 → D4（11:55 起） |
| gpu114 | 3461041 | 4 × L40S | 60 | 2026-11-08 18:30 | Y 训练（04:19–08:22，torchrun 4 卡）→ Y45K / Y50K 的 Dev-5（08:22–08:56）→ 主 All-14 队列 → 组合模型 Dev-5 → D4 |

**为什么没有新的 sbatch**：账户 QOS 最多同时 8 张 GPU，已被这两个交互分配占满，另有一个 4 卡作业 3465477 在排队。因此全部任务在已有分配内运行。

## 事件与恢复（完整记录见 `work_dirs/exp6_v3/Y/EVENTS.jsonl`、`work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead/EVENTS.jsonl`）
| 时间（MST） | 事件 | 处理 |
|---|---|---|
| 04:12 | D 冒烟：`d_config.yaml` 有 YAML 格式错误（值中有冒号） | 加引号、校验后重跑；没有影响任何结果 |
| 04:15 | Y 续训冒烟的第二段被跳过（最后一步的完整验证太耗时） | 改为直接检查 Y 写出的 checkpoint |
| 04:19 | Y 控制器误判训练进程不存在，导致重复启动；重复的那一份 OOM 退出（EXIT 1），没有写出 checkpoint | 存活的那一份从未改动的父状态正常开始；控制器已修正（不依赖路径写法的进程匹配，加 5 分钟宽限）；见 `Y/reports/DEVIATIONS.md` |
| 05:40 | gpu114 被训练占用约 36 GB / GPU | Y45K 的评估推迟到 50K 停止之后 |
| 08:22 | Y 在 50K 按 control.json 有序停止 | 45K / 50K 自动注册并开始评估 |
| 08:58 | 按预注册规则选出 Y50K（Y）与 D1（D）；注册 YBEST-H3 与 YD 组合 | 主 All-14 队列在 16 个 worker 上运行 |
| 10:00 | 主队列完成；dunnottar（gpu115，独占）与第二波（gpu114）自动接力 | — |
| 10:10 | 依据 YBEST-H3 的 Dev 结果（AUC@100 +0.010，5/0），把 YBEST-H3 的 All-14 排在 D4 之前 | — |
| 11:55 | gpu115 空闲，加 8 个 worker 跑 D4 | — |
| 12:31 | D4 的 Dev-5 + pisa 完成（30 / 30）；所有 t1_queue worker 退出，GPU 空闲 | 重新生成 Table 1 与全部报告 |

- **episode 失败**：本夜所有 D / Y / YBEST / YD 的 episode **0 次失败尝试、0 次放弃**。
- **训练异常**：无 NaN、无 OOM。
