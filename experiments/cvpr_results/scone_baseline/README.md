# EXP9-SCONE：SCONE 基线（Macarons++ 室外 + MP3D-150 室内）

任务书：`TASK_BOOK.md`（逐字）。当前状态：Phase A（审计）+ Phase B（集成）+ Phase C（smoke）完成后停下，等待批准 Phase D。

| 文件 | 内容 |
|---|---|
| `AUDIT_AND_PLAN.md` | Phase A：官方代码 / 权重 / 入口、MAGICIAN 论文 SCONE 的来历、复用的协议、计划 |
| `PROTOCOL_ADAPTATION.md` | 原版 SCONE 与适配逐项对照、公平性说明 |
| `SMOKE_TEST.md` | Phase C 结果、Phase D 入队命令与成本估计 |
| `scripts/scone_common.py` | 官方权重加载（sha256 + 与 `pretrained_macarons.pth` 逐位核对）、SCONE 贪心 NBV（官方调用）、占据场诊断 |
| `scripts/scone_sim_out.py` | 室外运行器 = 冻结 `m_t7_macarons.py` 逐行拷贝，差异 D1–D4 见文件头 |
| `scripts/scone_out_cov.py` | 室外官方覆盖重放（冻结 `t25_out_cov.py` 原样运行，须逐位一致） |
| `scripts/scone_sim_in.py` | 室内仿真侧（冻结 `t23_in_sim.main()` 原样运行，只替换 child） |
| `scripts/scone_planner_grid.py` | 室内 planner child（MAG-B child 的设置/观测/LeakGuard 原样复用，只换决策） |
| `scripts/scone_validate_in.py` | 室内单条验证（EXP7 `collect_mp3d150.validate` 原样 + 起点 / 权重 / guard 检查） |
| `scripts/scone_smoke_check.py` | smoke 检查（评估侧）：覆盖重放、合法性、oracle 候选打分相关性、占据场 sanity |
| `scripts/scone_queue_tasks.py` | 共享 GPU 队列任务生成（`smoke` / `full`，默认 dry run，`--write` 写入） |
| `scripts/inspect_weights.py` | Phase A 权重核对（CPU） |

外部资源（只读）：
- SCONE 官方仓库 clone：`/bsuscratch/jiahuizhang/projects/occupancy/SCONE_official/SCONE`（`24f1765`）；论文 PDF：`SCONE_official/paper/`。
- 官方权重：`SCONE_official/weights_gdrive/occupancy/pretrained_scone_occ.pth`、`.../coverage_gain/pretrained_scone_vis.pth`（MACARONS Google Drive，2026-11-14 下载，已设只读）。
- 上游代码参考拷贝：`work_dirs/cvpr_results/scone_baseline/reference/`（MACARONS `b23c180` scene tester / utils，MAGICIAN `e4e08fe` SCONE tester 与配置）。

输出：`work_dirs/cvpr_results/scone_baseline/{smoke,outdoor,indoor}/`。不写入 Table 1 / EXP7 / compare_full 的任何已有文件。

## 复现命令

```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
PY="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
$PY $R/experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py smoke          # dry run
$PY $R/experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py full           # dry run（114 个任务）
$PY $R/experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py full --write   # Phase D 入队（需批准）
```

## 变更记录

- 2026-11-14：建目录，保存任务书。Phase A：clone SCONE 官方仓库、下载论文和官方权重，核对 sha256 与容器一致性；发现 MAGICIAN `e4e08fe` 的 SCONE tester（30% 随机、碰撞判定写反、默认加载 trained_macarons.pth）。Phase B：写入全部脚本（CPU 编译、导入、权重加载测试通过）。Phase C：两条 smoke 任务进入共享 GPU 队列（`scone_0smoke_a_out_colosseum_s0`、`scone_0smoke_b_in_oLBMNvg9in8_L1_S36`，排在 FisherRF 室内任务之后）。
- 2026-11-14 17:03：Phase C 两条 smoke 完成（rc 0），G1–G9 全部通过（`SMOKE_TEST.md`）。按任务书停在 Phase D 之前，等待批准。
