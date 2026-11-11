# EXP7 作业记录

| 资源 / 作业 | 节点 | GPU | 用途 | 启动 | 状态 |
|---|---|---|---|---|---|
| 分配 3489889（`vscode-dev`，交互式，原有） | gpu114 | 4 × L40S | Pool A：位姿表生成；Round-1 smoke；MP3D-150 的 8 个 worker（每卡一个 A 槽、一个 B 槽），由 `scripts/pool_a_launch.sh` 启动 | 2026-11-10 17:12:25 | 运行中 |
| sbatch 3494056（`exp7_poolB`，`scripts/pool_b.sbatch`） | gpu115 | 3 × L40 | Pool B：MP3D-150 的 6 个 worker（UAVScenes 尚无可上 GPU 的单元，按任务书转跑 MP3D） | 2026-11-10 17:13:21 | 运行中（时限 14 h） |
| UAVScenes 可行性（CPU） | gpu114 CPU | 0 | 下载 HF zip 中央目录与 AMvalley 的最小资产，检查 mesh / 位姿 / 标定 / 渲染 | 16:5x–17:12 | 完成 → `UAVSCENES_FEASIBILITY.md` |
| 分配 3491984 | gpu107 | 1 × P100 | 另一个会话的分配，EXP7 不使用 | — | 未触碰 |

日志与产物位置：
- worker 日志：`work_dirs/cvpr_results/exp7/logs/worker_<pool>_<host>_g<gpu>_<joblist>.log`，每个单元每次尝试记一行 `exit rc ... wall`。
- 单条 episode 日志：`logs/episodes/<METHOD>_<tag>.log`，含逐步覆盖值以及 `/usr/bin/time -v` 记录的峰值内存。
- 任务清单：`jobs/mp3d150_slotA.txt`（78 行：OURS 44 条，之后 OURS-H0 34 条）和 `jobs/mp3d150_slotB.txt`（68 行：MAG-B 34 条，之后 RANDOM 34 条）。
- 认领记录：`jobs/claims/<METHOD>_<tag>/owner`。
- Smoke：`work_dirs/cvpr_results/exp7/smoke/`。
- 原始 episode：`episodes/mp3d150/raw/<METHOD>/<group>/<tag>/`，含 `svc/`、`episode.json`、`planner.jsonl`、`planner_info.json`、`planner.err`。
- 规范记录：`episodes/mp3d150/<METHOD>/<tag>.json`，由 `scripts/collect_mp3d150.py` 生成。

已发生的偏差：
- 17:20 前后重新运行了一次 `build_mp3d150_manifest.py`，用来写入 MAGICIAN 权重的 sha。它顺带重写了两份任务清单。两次运行打印的状态计数完全相同（176 行，PENDING / REUSED 数一致），而清单只由状态和固定排序决定，所以内容应当相同。但重写之前没有留存 md5，这一点无法用哈希直接核对。重写后的 md5：A 为 a8d0d103…，B 为 dc0f80f6…。各 worker 随后按顺序认领，没有出现重复认领：claims 目录用 mkdir 创建，同一单元只能成功一次。
- 此后构建脚本只有带 `--write-jobs` 参数才会写任务清单。
