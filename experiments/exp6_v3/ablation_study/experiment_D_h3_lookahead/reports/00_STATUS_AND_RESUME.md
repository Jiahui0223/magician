# 00_STATUS_AND_RESUME — ablation D（2026-11-04 12:00）

| 变体 | 状态 |
|---|---|
| d0ref | 回归通过（与 V3-H3 逐位一致） |
| D1 | All-14 完整（70 / 70） |
| D2 | Dev-5 + pisa（30 / 70），PARTIAL_BUDGET |
| D3 | Dev-5 + pisa（30 / 70），PARTIAL_BUDGET |
| D4 | 运行中，截至 11:55 为 16 / 30（Dev + pisa） |

- **D4 的运行情况**：gpu114 与 gpu115 各 8 个 worker，任务文件 `experiments/cvpr_results/table1/configs/jobs_wave2.txt`。
- **状态文件**：`work_dirs/exp6_v3/ablation_study/experiment_D_h3_lookahead/STATUS.md`，控制器在 gpu115 上每 60 秒更新一次。
- **续跑或补跑**：`bash experiments/cvpr_results/table1/scripts/t1_queue.sh <GPU> <jobfile>`。已完成的 episode 会自动跳过；失败的会自动存档，并最多重试 3 次。
- **D2 / D3 补齐 All-14**：生成缺失的任务行（`<model>:<scene>:<start>`），dunnottar 需要每卡只跑 1 个 worker。
