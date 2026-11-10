# RUN_LOG — CVPR overnight 2026-11-10（生成于 2026-11-10 05:23:33，git b8d65febc72b）

## 环境

| 项 | 值 |
|---|---|
| 节点 / Slurm | gpu114，job **3489889**（vscode-dev 交互分配，4 × NVIDIA L40S 46 GB，60 CPU，500 GB RAM，2026-11-09 20:35 → 11-14 20:35） |
| 运行方式 | 分配内的 `setsid nohup` 进程（没有提交新的 sbatch 作业：所有 L40 均已分配，新作业只能靠抢占他人作业启动，见 `OVERNIGHT_AUDIT.md` §5） |
| 环境 | 模拟器 / 官方重放 / Random / MACARONS：`/bsuscratch/jiahuizhang/envs/magician`；规划器 / Fig 4 前向 / 汇总：`/bsuscratch/jiahuizhang/envs/seen2scene` |
| 队列 | `experiments/cvpr_results/table1/scripts/t1_queue.sh GPU JOBFILE`：闭环 → 官方覆盖重放（不逐位相同即失败）→ 规范记录；noclobber claim；失败释放重试 ≤ 3 次；已完成的跳过（可断点续跑） |

## 各队列的执行统计（来自 `work_dirs/cvpr_results/table1/g3/logs/queue_gpu114_*.log`，仅 2026-11-10）

| 方法 | 成功 episode | 单条墙钟（均值 / 最大，秒，含官方重放） | 首条 → 末条完成时间 | 非零退出 | 失败重试目录 | 放弃 |
|---|---|---|---|---|---|---|
| V3-T0-K4D1 | 21 | 478 / 757 | 04:49:43 → 05:21:04 | 0 | 0 | 0 |
| V3-T0-K1D2 | 19 | 572 / 910 | 04:53:53 → 05:21:04 | 0 | 0 | 0 |
| V3-T0-K1D1 | 0（未启动） | — | — | — | — | — |
| RANDOM | 26 | 143 / 181 | 04:53:47 → 05:22:55 | 0 | 0 | 0 |
| MACARONS | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D2-N1 | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D2-N2 | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D2-N3 | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D1-N1 | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D1-N2 | 0（未启动） | — | — | — | — | — |
| V3-T0-K1D1-N3 | 0（未启动） | — | — | — | — | — |

### 回归 / 冒烟（`work_dirs/cvpr_results/overnight_20261110/logs/regression.log`）

```
REGRESSION START gpu114 job 3489889 2026-11-10 04:33:38
exit 0 T0_K1D1 gpu3 117s 2026-11-10 04:35:35
exit 0 T1_K4D1 gpu1 135s 2026-11-10 04:35:53
exit 0 T0_K1D2 gpu2 182s 2026-11-10 04:36:40
exit 0 T0_K4D2 gpu0 342s 2026-11-10 04:39:20
REGRESSION DONE 2026-11-10 04:39:20
```

### 回归第二轮（终版封装）（`work_dirs/cvpr_results/overnight_20261110/logs/regression2.log`）

```
REGRESSION2 START gpu114 job 3489889 2026-11-10 04:39:16
exit 0 T0_K1D1_N1 gpu2 106s 2026-11-10 04:41:02
exit 0 T1_K4D1 gpu1 124s 2026-11-10 04:41:20
exit 0 T0_K1D2_N3 gpu3 177s 2026-11-10 04:42:13
exit 0 T0_K4D2 gpu0 344s 2026-11-10 04:45:00
REGRESSION2 DONE 2026-11-10 04:45:00
```

### Fig 4 扫描与背面检查（`work_dirs/cvpr_results/overnight_20261110/logs/fig4_scan.log`）

```
exit 0 fig4_scan alhambra gpu2 158s 2026-11-10 05:07:44
exit 0 fig4_backface bannerman gpu3 37s 2026-11-10 05:08:04
exit 0 fig4_backface redeemer gpu3 48s 2026-11-10 05:08:52
exit 0 fig4_backface colosseum gpu3 45s 2026-11-10 05:09:37
exit 0 fig4_scan bannerman gpu1 289s 2026-11-10 05:09:55
exit 0 fig4_backface pantheon gpu3 47s 2026-11-10 05:10:24
exit 0 fig4_backface alhambra gpu3 45s 2026-11-10 05:11:09
exit 0 fig4_backface neuschwanstein gpu3 49s 2026-11-10 05:11:58
exit 0 fig4_scan neuschwanstein gpu2 257s 2026-11-10 05:12:01
exit 0 fig4_scan redeemer gpu1 168s 2026-11-10 05:12:43
exit 0 fig4_scan colosseum gpu2 280s 2026-11-10 05:16:41
exit 0 fig4_scan pantheon gpu1 294s 2026-11-10 05:17:37
```

### 阶段标记（`work_dirs/cvpr_results/overnight_20261110/logs/table3_chain.log`）

```
T3_RUN_START gpu114 slurm_job 3489889 pid 34623 2026-11-10 04:43:42
T3_LAUNCHER_PARENT_STOPPED (phase 2 will be started manually after the MACARONS baseline; phase-1 workers keep running) 2026-11-10 04:51:06
RANDOM_RUN_START gpu114 slurm_job 3489889 2 workers on GPU 0 2026-11-10 04:51:44
FIG4_SCAN_START gpu114 slurm_job 3489889 GPUs 1,2 2026-11-10 05:05:09
```

### 回归比对结果（`logs/regression_check.txt`）

```
regression T0_K4D2 vs Table 1 V3-T0-D1: executed_keys_equal=True official_coverage_equal=True K=4 depth=2 ckpt=7ed0ddd72bb9 kd_sha=9cb47eec7b22 -> PASS
regression T1_K4D1 vs Table 1 V3-H1: executed_keys_equal=True official_coverage_equal=True K=4 depth=1 ckpt=639c9d3ae3f3 kd_sha=9cb47eec7b22 -> PASS
regression2 T0_K4D2 vs Table 1 V3-T0-D1: executed_keys_equal=True official_coverage_equal=True K=4 depth=2 ckpt=7ed0ddd72bb9 kd_sha=3c61da691a5e -> PASS
regression2 T1_K4D1 vs Table 1 V3-H1: executed_keys_equal=True official_coverage_equal=True K=4 depth=1 ckpt=639c9d3ae3f3 kd_sha=3c61da691a5e -> PASS
regression2 T0_K1D1_N1: K=1 noise_offset=1 | t=0 per-view value closest to K4 sample 1 (max rel diff 0.021; to the other samples [0.65, 0.7, 0.89]) -> PASS
regression2 T0_K1D2_N3: K=1 noise_offset=3 | t=0 per-view value closest to K4 sample 3 (max rel diff 0.013; to the other samples [2.21, 8.13, 1.84]) -> PASS
REGRESSION PASS
```

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
