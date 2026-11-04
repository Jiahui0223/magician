# STATUS — EXP6 V3 ablation D（2026-11-04T19:34:06Z，controller 自动更新）

- **阶段**：COMPLETED
- **作业**：3443137 (gpu115) + 3461041 (gpu114)；本节点存活的 worker：0；任务文件 `experiments/cvpr_results/table1/configs/jobs_main_all14.txt` 剩余 0 个任务
- **失败尝试**：0；放弃：0（队列会自动重试最多 3 次）；自动重启次数：0
- **续跑命令**：`bash experiments/cvpr_results/table1/scripts/t1_queue.sh <GPU> experiments/cvpr_results/table1/configs/jobs_main_all14.txt`

| 模型 | Train-6 | Dev-5 | Test-3 | All-14 |
|---|---|---|---|---|
| V3-D1-H3-NORISK | 30/30 | 25/25 | 15/15 | 70/70 |
| V3-D2-DEPTH3-B3 | 5/30 | 25/25 | 0/15 | 30/70 |
| V3-D3-DEPTH3-B5 | 5/30 | 25/25 | 0/15 | 30/70 |
| V3-D4-DEPTH4-B3 | 5/30 | 25/25 | 0/15 | 30/70 |
| V3-Y45K-H0 | 0/30 | 25/25 | 0/15 | 25/70 |
| V3-Y50K-H0 | 30/30 | 25/25 | 15/15 | 70/70 |
| V3-YBEST-H3 | 30/30 | 25/25 | 15/15 | 70/70 |
| V3-YD-COMBINED-Y50K-D1 | 0/30 | 25/25 | 0/15 | 25/70 |
