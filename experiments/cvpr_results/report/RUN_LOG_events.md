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
| 05:24–05:29 | **事故（我的操作造成）**：为了压低 MACARONS 的显存占用，在 GPU 2 上用 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` 重跑 barts 起点 0 做测试，进程反而涨到 38.9 GB，把 GPU 2 占满。05:25–05:29 该卡上 Table 3 的 53 次 episode 启动因 CUDA out of memory 失败（全部在 gpu2，涉及 24 条 episode，其中 11 条累计 3 次），队列自动释放，没有任何 episode 被放弃。05:29:05 终止测试，worker 恢复。53 个失败目录**移到** `work_dirs/cvpr_results/table1/g3/failed/_incident_20261110_0525_gpu2_oom/`（保留，未删除），使队列的 3 次重试额度不被这次基础设施错误占用。损失：GPU 2 约 4 分钟的吞吐。没有任何已完成记录受影响（规范记录只在官方重放通过后写入）。教训：MACARONS 每张卡只放 1 个进程，不与闭环规划器混跑做显存试验 | `logs/table3_chain.log` 的 INCIDENT 行；`table1/g3/logs/queue_gpu114_2.log` |
| 05:31 | **第二阶段启动脚本进入等待**：第一阶段结束后自动开始 MACARONS（每卡 1 个 worker；dunnottar 在 GPU 0 独占）+ K = 1 噪声复核（N1 深度 2 → N1 深度 1 → N2 → N3） | `setsid nohup bash experiments/cvpr_results/scripts/table3/run_phase2.sh`（pid 56398） |
| 05:36–05:40 | 14 场景 × 5 官方起点的背面审计（magician env，GPU 3，约 4 min）：4 个起点位姿在网格内部 | `scripts/audit_start_backface.py` → `manifest/start_pose_backface.csv` |
| 05:39:41 | V3-T0-K1D2 bridge 起点 2 在 GPU 3 上显存不足一次（当时 GPU 3 上还有 Fig 4 背面检查），队列释放后重试成功 | `table1/g3/failed/V3-T0-K1D2_V2M_bridge_s2_r0_try1` |
| 05:44 | 暂停 Fig 4 背面检查（避免再与闭环抢显存），并把它扩展为同时检查当前视角与 P_t 已观测的视角（选例规则的预先声明修订，见 `fig4_plot.py` 文档；修订前的选择存为 `fig/fig4/selection_v0.json`） | `logs/table3_chain.log` |
| 06:14:49 | Random Walk 70 / 70 | 单条均值 142 s |
| 06:38–07:11 | Table 3 第一阶段完成：K4D1 06:38、K1D2 06:40、K1D1 07:11（`T3_PHASE1_DONE` 07:11:21） | 单条均值 420 / 514 / 297 s |
| 07:11:33 | 第二阶段自动开始：MACARONS（每卡 1 个 worker）+ K = 1 噪声复核 | `run_phase2.sh` |
| 09:18:52 | MACARONS 70 / 70 | 单条均值 431 s，0 次失败 |
| 09:50–14:19 | K = 1 噪声复核 6 组全部 70 / 70（N1 深度 2 09:50、N1 深度 1 10:23、N2 深度 2 11:36、N2 深度 1 12:23、N3 深度 2 13:35、N3 深度 1 14:19；`PHASE2_DONE` 14:19:10） | 0 次失败 |
| 14:19–15:10 | 会话中断（用户更换模型）；期间没有任务在跑，没有数据丢失 | — |
| 15:10–15:14 | 恢复 Fig 4 扩展背面检查（4 卡并行，6 个场景各 97–181 s） | `logs/fig4_scan.log` |
| 15:15–15:25 | 完整汇总：manifest（1,050 条 COMPLETE 记录）→ 三张表 → Fig 3 / 4 / 5 → K = 1 配对核对 → 25 项质量关口全部 PASS → 报告 → sha256 | `bash experiments/cvpr_results/scripts/run_all.sh` |
| 15:20 | 修正两处检查 / 显示规则（不改任何数据）：K = 1 配对检查把 t = 0 新覆盖值全为 0 的并列状态（16 个）误判为不匹配，改为允许并列并直接核对握手里的噪声偏移（560 / 560）；表格加粗改为按三位小数后的值、并列都加粗（Table 3 的 AUC@20 0.4195 与 0.4205 都显示 0.420） | `scripts/table3/k1_pairing_check.py`、`scripts/make_tables.py` |

## 运行中发现的事实

- **K = 1 与 K = 4 第 0 个样本同种子但非逐位相同**：liberty 起点 0、t = 0，9 个候选的一步价值相对差最大 2.5%（K = 4 四个样本之间的离散度 12–75%）。原因是 bf16 自动混合精度下批大小不同（每视角 1 个 vs 4 个样本）。K = 1 行是真实的单样本推理，未做补偿。
- **实际吞吐由模拟器决定**：约 0.9 条 / 分钟（10 个 worker），与 K、深度关系不大；小场景（liberty）单条约 2–3 min，大场景在 3 worker / GPU 下 8–10 min。
- **MACARONS 的显存**：默认分配器下 nvidia-smi 22.9 GB / 进程（PyTorch allocated 峰值只有 5.3 GB）；加 `max_split_size_mb:256` 后反而到 38.9 GB。因此每张 L40S 只放 1 个 MACARONS worker。
- **pantheon 起点 2 的相机在网格内部**（Fig 4 背面检查的副产物）：该起点 6 个检查步的背面像素占比 11–70%；neuschwanstein 起点 2 为 90–98%（与此前记录一致），起点 0 的 t = 30 为 15%。这些是 MAGICIAN 官方起点，所有方法都一样面对；本轮只把它们排除在 Fig 4 的例子之外，**没有**从任何表格中剔除。
- 没有提交 sbatch 作业：`sbatch --test-only` 显示新的 L40 作业要么 2026-11-11 02:58 才能启动，要么需要抢占用户 osg01 的 6 个作业。
