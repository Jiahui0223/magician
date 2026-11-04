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
