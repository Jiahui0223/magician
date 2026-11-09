# CVPR Table 1 — Macarons++ Active Mapping Benchmark

这是论文 Table 1 的统一入口，也是今后所有室外主动建图实验的标准结果来源。协议见 `PROTOCOL.md`（v1.0），机器可读版本见 `FROZEN_CONFIG.yaml`。

## 目录
| 路径 | 内容 |
|---|---|
| `SCENE_REGISTRY.yaml`、`DATASET_AUDIT.md` | 15 个场景、Train / Dev / Test / Other 划分及依据、资产、起点、能否运行 |
| `MODEL_REGISTRY.yaml`、`CHECKPOINT_AUDIT.md` | 模型登记：checkpoint、sha256、训练历史、推理 / 评分密度、启动链 |
| `EVALUATION_AUDIT.md`、`audit/` | 历史 episode 清单与核对 |
| `TABLE1*.md`、`TABLE1_FULL.csv`、`TABLE1_LATEX.tex` | **由 `scripts/aggregate.py` 自动生成，不要手改** |
| `results/` | 各层级 CSV / JSON；`*_mono.*` 为辅口径（单调覆盖） |
| `figures/` | 由 `scripts/plot.py` 生成的 PNG 和 PDF |
| `reports/` | 基线复现、缺失 episode、计算成本、结果分析、最终报告 |
| `work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json` | **逐 episode 结果的唯一来源**（不进 git） |
| `work_dirs/cvpr_results/table1/g3/` | 新跑的闭环原始输出、日志、认领文件、失败归档 |

## 重新生成全部结果
```bash
cd experiments/cvpr_results/table1/scripts
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
$PY validate.py                       # 完整性检查 → reports/MISSING_EPISODES.md
$PY aggregate.py                      # 主口径（MAGICIAN 官方覆盖）→ TABLE1*、results/*
$PY aggregate.py --protocol mono      # 辅口径（单调覆盖）→ results/*_mono.*
$PY plot.py                           # figures/*.png、*.pdf
$PY baseline_report.py                # reports/BASELINE_REPRODUCTION.md
```

## 加入新模型（不需要改写 Table 1）
1. 在 `MODEL_REGISTRY.yaml` 中登记：id、label、status、sha256、训练历史、推理 / 评分密度、启动链。
2. 在 `scripts/t1_queue.sh` 的 `run_sim` 中加一行启动命令。只能用该模型自己的历史启动链，不要修改规划器或评价器。
3. 用当前代码重跑一个已有历史结果的 episode，确认逐位一致（`scripts/replay_check.sh <GPU> <MODEL>`）。若是全新模型，没有历史结果可比，则记录为首次运行。
4. 生成任务列表（`<MODEL>:<scene>:<start>`，14 个可运行场景 × 5 个起点），运行 `t1_queue.sh <GPU> <jobfile>`。每个 episode 完成后会自动做覆盖重放核对，并写入统一记录。
5. 重新运行上面的生成命令，表格、图表和与 MAG-B 的配对比较会自动更新。

**历史模型的结果冻结，不得覆盖**：`import_episodes.py` 发现已有记录的曲线不同时会 STOP。

更改评价协议必须新建 protocol 版本（PROTOCOL.md §8），不能静默覆盖。

## 变更记录
| 日期 | 变更 |
|---|---|
| 2026-11-03 | protocol v1.0 冻结。完成场景、模型与历史 episode 审计，复用 390 个历史 episode，8 个主模型回归逐位一致 |
| 2026-11-03 | 补跑 520 个 episode（双节点 8 × L40），dunnottar 的双图任务改为独占运行。910 / 910 有效，生成全部表格、图表与报告（`reports/FINAL_REPORT.md`） |
| 2026-11-03 | 按“加入新模型”流程加入 EXP6 V5 / F1 的两个 Dev-3 选定 checkpoint：F1-CONT-v1@2k 与 F1-GQ-v1@5k（role f1，不进主图）。各 70 / 70，由本流水线运行（dunnottar 上 5 次 OOM，自动重试成功）；Dev-3 部分与 F1 记录 15 / 15 逐位一致。已有 13 个模型的结果没有改动 |
| 2026-11-04 | 加入 EXP6 V3 / H 的 V3-H1 / V3-H2 / V3-H3（role h；冻结 V3-T1 + 新评分 / 规划；H0 = V3 行）。各 70 / 70，由本流水线运行，0 次失败。已有模型的结果没有改动 |
| 2026-11-04 | 加入 EXP6 V3 夜间 Y / D 的行（role y / d / yd）：V3-D1-H3-NORISK、V3-YBEST-H3、V3-Y50K-H0 各 70 / 70；V3-D2-DEPTH3-B3、V3-D3-DEPTH3-B5、V3-D4-DEPTH4-B3 各 30 / 70（Dev-5 + pisa，partial）；V3-Y45K-H0 与 V3-YD-COMBINED-Y50K-D1 当时为 Dev-5（25 / 70，partial）。均由本流水线运行，已有模型的结果没有改动 |
| 2026-11-04 | 补齐 V3-YD-COMBINED-Y50K-D1 的 Train-6 / Test-3（45 条），现为 70 / 70，用于 EXP6 V3 Experiment E 的 2 × 2（权重 × 风险项）。dunnottar 的 5 条因与诊断进程同卡而 OOM 并用尽 3 次重试；锁文件删除后每卡一条重新排队，failed 目录全部保留，s2 从 t = 35 逐位续跑（`logs/requeue.log`）。已有模型的结果没有改动 |
| 2026-11-04 | 新增两个独立目录：`../protocol_audit/`（MAGICIAN 评估与训练协议审计）和 `../paper_tables/`（论文风格的表与图，只读本目录的 `results/` 与规范 episode，`refresh.sh` 一键刷新）。本目录的文件与数值没有改动；审计对已有审计文档的 4 处更正记录在 `../protocol_audit/MAGICIAN_TRAINING_AUDIT.md` §D |
| 2026-11-04 | 加入 EXP6 V3 Experiment E3 的行 **V3-DV2-KSV**（role e）：V3-T1 40K + D1 + 可开关的已观测表面可见性约束（`experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/E3_dv2_closed_loop/`，协议 `configs/dv2_v2.yaml`）。Dev-5 预登记门槛通过后才登记；70 / 70 由本流水线运行（0 失败、0 重试），其中 30 条与实验目录里的运行逐位一致。`MODEL_REGISTRY.yaml` 追加一个条目、`scripts/t1_queue.sh` 追加一个分支；已有模型的条目、规范记录和汇总行逐行未变（per_model / per_split / per_scene / per_episode 核对）。All-15 仍为不可用（sofia_church PENDING） |
| 2026-11-08 | 按“加入新模型”流程加入 **V3-T0-D1**（role d）：V3-T0（EXP6 V3 task0_vae，与 V3-T1 同数据 / 配方 / seed，仅 `xattn.enabled = false`；sha `7ed0ddd7…`）+ 冻结 D1（与 V3-D1-H3-NORISK 同 d_policy / d_config，规划器握手里的三个 sha 逐位相同）。动机：XB Task A 证明 T1 的交叉注意力在推理时惰性。首次运行、无历史可比；70 / 70 由本流水线在 gpu115 4 卡运行（`scripts/run_t0d1.sh`，0 失败 0 重试，19.6 GPU-h）。All-14：AUC@20 0.420 / Cov@20 0.646 / AUC@50 0.633 / AUC@100 0.763 / Final 0.911，对 V3-D1-H3-NORISK 五项差均在场景级 95% CI 内。`MODEL_REGISTRY.yaml` 追加一个条目、`t1_queue.sh` 追加一个分支；paper_tables 的 supp / curve / improvement 列表加入该行；已有模型记录与汇总未变 |
