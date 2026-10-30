# EXP5 Task 0 扩展：MAGICIAN 室内复现与同协议对照（T0.8–T0.12）

> 依据：你 2026-10-24 的扩展任务书。T0.1–T0.7 的结果、轨迹、模型和协议一律只读，本目录与 `W/magician_comparison/` 物理隔离。
> 协议：`protocol.md` v1.0（T0.8）、v1.1（T0.9）、v1.2（T0.10）、v1.3（T0.12），每个阶段都在正式运行前冻结，指纹记在 `FROZEN_CONFIGS.json`。
> 偏差：`DEVIATIONS.md`（只追加）；正式运行清单：`RUN_MANIFEST.json`。

## 状态

| 阶段 | 状态 | 报告 |
|---|---|---|
| T0.8 现有 22 个楼层轨迹的统一指标复算 | **DONE**（10-24 22:15）：110 条 episode 的覆盖率复现完全一致；单位测试通过；Comp@5cm：V2K1 0.607、V2K4 0.600、随机参考 0.489、Frozen 0.414、T1-C 0.401 | `../reports/T0.8_metric_report.md` |
| T0.9 MAGICIAN 官方 MP3D 复现 | **DONE**（10-24 23:15）：论文协议无法用发布代码复现；发布代码协议 15 / 15 次运行成功，3 个米制场景 150 个位姿时 Comp@5cm 0.839、平均距离 4.67 cm | `../reports/T0.9_magician_reproduction.md` |
| T0.10 同协议对照 | **DONE**（10-25 01:40）：66 / 66 条 episode。V2K4 与 MAG-B 相当（AUC 0.467 对 0.475）；beam 的贡献很小（+0.021）；V2K4-NV 最好（AUC 0.509） | `../reports/T0.10_fair_comparison.md` |
| T0.11 统一分析 | **DONE** | `../reports/T0.11_final_comparison.md`；`../TASK0_FINAL_REPORT.md` §14 |
| T0.12 网格动作空间对齐（你 10-25 确认的设计：1.0 m 网格、5 × 8 朝向、官方合法性规则、150 步、5 层 × 2 起点） | **DONE**（10-25 20:18）：50 / 50 条 episode。MAG-B 的 AUC@100 为 0.254，高于 V2K4-NV 0.224 和 V2K4 0.214，但 CI 都含 0，优势集中在一个房屋；150 步时三者覆盖率相同；相机会飞到建筑外（描述性诊断） | `../reports/T0.12_grid_alignment.md`；`../TASK0_FINAL_REPORT.md` §15 |
| T0.12-C 离线诊断：V2 为什么不往远处走（你 10-25 要求） | **DONE**（10-25）：主要原因是 V2 的规划区域只有 7.8 m，第 50 步以后区域内只剩约 6% 的工作。把区域换成整层（探针）后，同一个 V2 在 TbHJ 上选中真实一步最优的比例从 27% 升到 47–51%。MAGICIAN 一步并不更准，它的优势在于打分覆盖整层 | `../reports/T0.12C_far_exploration_diagnosis.md` |

## 目录
- **experiments/…/magician_comparison/**：
  - `protocol.md`、`FROZEN_CONFIGS.json`、`RUN_MANIFEST.json`、`DEVIATIONS.md`；
  - `configs/`：`t08_metrics/`、`t09_official/`（官方配置的副本，只改了 README 规定的项）、`t10_shared_protocol/`；
  - `adapters/`：`magician_core.py`（与官方循环逐位等价）、`station_env.py`、`magician_planner.py`、`v2nv_planner.py`；T0.12 新增 `grid_env.py`、`magician_planner_grid.py`；
  - `scripts/`：`mc_common.py`、`t08_*`、`t09_*`、`t10_sim.py`、`t10_beam_analysis.py`、`t11_viz.py`；T0.12 新增 `t12_sim.py`、`t12_analyze.py`、`t12_viz.py`；
  - `tests/`：`t08_units_test.py`、`t10_native_equiv.py`、`t12_grid_tests.py`。
- **work_dirs/…/magician_comparison/**：
  - `external/MAGICIAN`：官方 `e697009`，用 `--recursive` 克隆；
  - `weights/official_drive`：官方 Drive 权重，附 sha256；
  - `manifests/`、`metric_replay/`、`official_reproduction/`、`shared_protocol/`、`grid_protocol/`（T0.12）、`analysis/`、`figures/`、`logs/`、`snapshots/`。
- **数据**（不进仓库、不上传）：
  - MP3D：`/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans`（= `/bsuscratch/jiahuizhang/data/…`）；
  - `mp3d_magician`：`/bsuhome/jiahuizhang/scratch/data/mp3d_magician/`。

## 环境
- 现有 magician 环境（Python 3.9、torch 2.6.0+cu118、PyTorch3D 0.7.8；RaDe-GS 扩展来自同一子模块提交 `5f5cd3b`），没有安装任何新包。
- V2 的规划器仍用 seen2scene 环境。

## 可视化（10-25 追加；`scripts/t11_viz.py`，描述性，只读取已完成的结果）
- `W/magician_comparison/figures/indoor/`：
  - `dataset_levels.png`：22 层测试集的平面图和站点图；
  - `auc_area.png`：8 组的房屋聚类平均覆盖曲线和 AUC，以及与 MAGICIAN Beam 的差；
  - `auc_heatmap.png`：22 条 episode × 8 组的 AUC 热力图；
  - `paired_dots.png`：三类比较和消融的逐条 episode 差值；
  - `behaviour.png`：回访、停滞、站点数、Comp；
  - `cover_<episode>.png`：5 条代表性 episode 的覆盖范围图。
- `W/magician_comparison/anim/anim_<episode>.{mp4,gif}`：5 条代表性 episode 的逐步动图（101 帧，T1-C / V2K4 / MAG-G / MAG-B / V2K4-NV 并排；俯视覆盖、轨迹、深度视图、覆盖率曲线、到目前为止的 AUC）。
- 逐点覆盖重放在 `W/magician_comparison/viz/covrep/`（25 条），每条重放曲线都与存档完全一致。
- 问答（数据位置、划分、Cov@t 和"步"的定义）：`../reports/T0_FAQ_dataset_metrics.md`。
- T0.12（网格）：`W/magician_comparison/figures/grid/`：
  - `grid_curves.png`、`grid_paired.png`、`grid_vs_station.png`、`cover_<episode>.png`；
  - `anim/anim_grid_<episode>.{mp4,gif}`：5 个首起点，151 帧，叠加了网格点。
