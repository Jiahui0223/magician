## 时间线、命令与事件（手工记录，时间为 gpu114 的 `date`）

| 时间 | 事件 | 命令 / 位置 |
|---|---|---|
| 04:28 | 读取 `cvpr_overnight_experiment_plan.md`，开始预检查（git、Slurm、GPU、现有记录、冻结代码） | — |
| 04:33:38 | 回归 / 冒烟第一轮（4 条，liberty 起点 0，每卡 1 条） | `setsid nohup bash work_dirs/cvpr_results/overnight_20261110/jobs/regression.sh` |
| 04:39:16 | 回归第二轮（封装终版：加入 `--noise`） | `setsid nohup bash work_dirs/cvpr_results/overnight_20261110/jobs/regression2.sh` |
| 04:43:42 | **Table 3 第一阶段启动**：V3-T0-K4D1 / K1D2 / K1D1 共 210 条，10 个 worker（GPU 0：dunnottar 链 1 个；GPU 1–3：各 3 个） | `setsid nohup bash experiments/cvpr_results/scripts/table3/run_table3.sh`（pid 34623） |
| 04:51:06 | 只停掉启动脚本的父进程（`kill 34623`），使第二阶段（K = 1 噪声复核，P2）不会自动接在第一阶段后面；10 个第一阶段 worker 不受影响。改由 `jobs/phase1_monitor.sh` 在第一阶段 worker 全部退出后写入 `T3_PHASE1_DONE` | `logs/table3_chain.log` |
| 04:51 | **Random Walk 基线启动**：GPU 0 上 2 个 worker（只有模拟器，各 1.5 GB） | `setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh 0 work_dirs/cvpr_results/overnight_20261110/jobs/random.txt`（× 2） |
| 04:51–05:04 | MACARONS 冒烟（barts 起点 0，GPU 3）：770 s（与 3 个闭环 worker 共享 GPU；T7a 独占时约 400 s），nvidia-smi 显存 22.9 GB（PyTorch allocated 峰值 5.3 GB） | `m_t7_macarons.py --scene barts --start 0 --seed 0 --out .../regression/MACARONS/macarons` |
| 05:05 | **Fig 4 候选池扫描**：GPU 1（bannerman、redeemer、pantheon）、GPU 2（alhambra、neuschwanstein、colosseum），每个场景 158–294 s | `setsid nohup bash work_dirs/cvpr_results/overnight_20261110/jobs/fig4_scan.sh GPU SCENE…` |
| 05:07 | Fig 4 背面检查（magician env，GPU 3，每场景 37–49 s） | `setsid nohup bash work_dirs/cvpr_results/overnight_20261110/jobs/fig4_backface.sh 3` |
| 05:18 | Fig 4 选例与出图；`run_all.sh` 首次完整运行，质量关口 23 PASS / 1 PENDING | `bash experiments/cvpr_results/scripts/run_all.sh` |

## 运行中发现的事实

- **K = 1 与 K = 4 第 0 个样本同种子但非逐位相同**：liberty 起点 0、t = 0，9 个候选的一步价值相对差最大 2.5%（K = 4 四个样本之间的离散度 12–75%）。原因是 bf16 自动混合精度下批大小不同（每视角 1 个 vs 4 个样本）。K = 1 行是真实的单样本推理，未做补偿。
- **实际吞吐由模拟器决定**：约 0.9 条 / 分钟（10 个 worker），与 K、深度关系不大；小场景（liberty）单条约 2–3 min，大场景在 3 worker / GPU 下 8–10 min。
- **MACARONS 的显存**：nvidia-smi 22.9 GB / 进程，因此每张 L40S 最多 2 个 MACARONS worker。
- **pantheon 起点 2 的相机在网格内部**（Fig 4 背面检查的副产物）：该起点 6 个检查步的背面像素占比 11–70%；neuschwanstein 起点 2 为 90–98%（与此前记录一致），起点 0 的 t = 30 为 15%。这些是 MAGICIAN 官方起点，所有方法都一样面对；本轮只把它们排除在 Fig 4 的例子之外，**没有**从任何表格中剔除。
- 没有提交 sbatch 作业：`sbatch --test-only` 显示新的 L40 作业要么 2026-11-11 02:58 才能启动，要么需要抢占用户 osg01 的 6 个作业。
