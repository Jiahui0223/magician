# EXP6 V5 / F1 — Geometry Query Isolation & Early-Coverage Evaluation

**问题**：从 B0（V2-Mix EMA@17.5k）出发，只把 Ray32/Conv1D 几何条件换成 V5 的多尺度 Geometry-Memory Query，在同样的室外 S10 数据、5k 步预算与 B0 评估协议下，G1 深度 / 命中、G2 选点、G3 的 AUC@20 / AUC@100 / Final 是否提升？

**对照**：F1-A 用同一份 B0 原样继续训练 5k 步。

- **任务书**：`../EXP6_V5_F1_AI_TASK.md`。
- **协议**：`PROTOCOL.md`（f1-v1，训练前冻结）。
- **进度**：`RUN_STATUS.md`。
- **结论**：`reports/FINAL_REPORT.md`。

## 文件
| 路径 | 内容 |
|---|---|
| `PROTOCOL.md`、`configs/*.yaml` | 冻结的训练与评估协议 |
| `CONFIG_ALIGNMENT.md`、`CHECKPOINT_AUDIT.md` | 配置对照；父模型 / 继承 / checkpoint 审计 |
| `V3_HISTORY.md` | V3-T0 / T1 的历史结论，只读整理 |
| `DEVIATIONS.md` | 运行偏差 |
| `scripts/train_f1.py`、`scripts/slurm/train_arm.sh` | 两个分支的训练（torchrun 4 卡，可续训） |
| `scripts/smoke_compare.py` | 起步时的条件输出统计，以及 out_scale_init 规则 |
| `scripts/eval_g12.py`、`scripts/analyze_g12.py` | G1/G2 固定状态诊断 |
| `scripts/run_g3.sh`、`g3_cov.py`、`g3_import.py` | G3 闭环队列、覆盖率重放、记录导入 |
| `scripts/aggregate.py`、`plot.py`、`validate.py` | 汇总、作图、完整性检查（结果全部自动生成） |
| `results/`、`figures/`、`reports/` | 生成的表格、图表与报告 |
| `table1_export/f1_model_registry_entry.yaml` | 提交给 CVPR Table 1 的注册条目 |
| `work_dirs/exp6_v5/F1/` | 权重、日志、闭环原始输出（不进 git） |

## 复现
```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
# 训练（每个分支 4 卡；同一命令可续训）
bash experiments/exp6_v5/F1/scripts/slurm/train_arm.sh A
bash experiments/exp6_v5/F1/scripts/slurm/train_arm.sh B
# G1 / G2（每卡一个进程，按 part 切分；需要 LD_PRELOAD=libtbb 等环境，见 train_arm.sh）
$PY experiments/exp6_v5/F1/scripts/eval_g12.py --stratum B0 --part 0/4
# G3（每个 worker 绑定一张卡，可续跑）
bash experiments/exp6_v5/F1/scripts/run_g3.sh 0 experiments/exp6_v5/F1/configs/jobs_g3_dev3.txt
# 汇总
$PY experiments/exp6_v5/F1/scripts/validate.py
$PY experiments/exp6_v5/F1/scripts/aggregate.py
$PY experiments/exp6_v5/F1/scripts/analyze_g12.py
$PY experiments/exp6_v5/F1/scripts/plot.py
```
