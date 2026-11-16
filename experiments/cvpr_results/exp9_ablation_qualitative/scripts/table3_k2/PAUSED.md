# T3K2（K = 2 噪声图 2–3 的重复实验）暂停记录（2026-11-16 16:00，用户决定）

**状态**
- 论文需要的 Unseen-6 部分**已全部完成**：两个 id 各 30/30，官方重放逐位一致。
- Table 3 v4（`fig/table3/table3_hypothesis_planning_v4.*`）已生成并提交（489eccf），**不受暂停影响**。用户只使用 Unseen-6，All-14 版本不进论文。

**暂停时的进度（其余 8 个场景）**

| id | 完成条数（共 70 条，含 Unseen-6 的 30 条） | 未完成 |
|---|---|---|
| V3-T0-K2D1-N2 | 30/70 | 其余 8 个场景全部未完成 |
| V3-T0-K2D2-N2 | 50/70 | 20 条 |

**停止方式**
- 按 PID 停掉 t3k2 队列任务及其 t3k2_task.sh / t1_queue.sh 子进程，以及 sim / planner / 重放进程（gpu114 和 gpu115），两台机器的 GPU 均已释放。
- gpu114 上的 g114x 临时 worker 已退出；gpu115 上共享队列的 worker 未动。
- 被中断的 20 条 episode 的 claim 已移到 `work_dirs/cvpr_results/table1/g3/claims_paused_t3k2_20261116/`（移走，未删除）。它们的半成品输出留在原处，由 t1_queue 的重试逻辑处理。

**如需恢复（只影响 All-14 记录）**
1. 按 `t3k2_task.sh` 的用法重新入队剩余场景（t1_queue 会跳过已有 canonical 记录的条目）。
2. 全部 140 条完成后运行 `bash experiments/cvpr_results/exp9_ablation_qualitative/scripts/table3_k2/t3k2_collect.sh`。

**未提交**：共享登记文件里 T3K2 的新增行混有 EXP9 另一会话尚未提交的改动，暂未提交，涉及：
- `scripts/cvpr_lib.py`
- `table1/MODEL_REGISTRY.yaml`
- `table1/scripts/t1_queue.sh`
- `scripts/build_manifest.py`
- `manifest/episodes.csv`
- `manifest/provenance.json`
