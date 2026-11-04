# EXP6 V3 / H — Coverage-Aware Decision & Short-Horizon Planning

**问题**：冻结 V3-T1（3D VAE + Scene X-Attn + 2D rectified flow），只改评分与规划。比较以下四种策略：
- **H0**：原 DistP1 Greedy，等于 Table 1 的 V3 行；
- **H1**：覆盖对齐的几何价值；
- **H2**：在 H1 基础上对 K = 4 个样本做风险校正，mean − 0.25·std；
- **H3**：两步并集规划。

比较在 Macarons++ All-14（Train-6 / Dev-5 / Test-3 × 5 个起点 × 100 步）上进行，与 CVPR Table 1 的流水线完全统一。

- **任务书**：`../EXP6_V3_H_PLANNER_AI_TASK.md`
- **协议**：`FROZEN_PROTOCOL.md` 与 `FROZEN_CONFIG.yaml`（H-v1，正式运行前冻结）
- **结论**：`reports/FINAL_REPORT.md`

## 文件
| 路径 | 内容 |
|---|---|
| `MODEL_AUDIT.md`、`SCORING_AUDIT.md`、`DATASET_AUDIT.md`、`CVPR_TABLE1_ALIGNMENT.md` | 审计：模型、冻结评分路径（含代码行号）、场景、Table 1 接入方式 |
| `scripts/h_policy.py` | 四种策略：在规划器进程内替换 `score_state` 的打分 |
| `scripts/h_out_planner.py`、`scripts/sim_out_h.py` | 规划器子进程与模拟器启动器（冻结 M 协议） |
| `scripts/build_pose_tables.py` | H3 第二层用的设置网格位姿表，并与服务记录逐位核对 |
| `scripts/g0_regression.py` | G0 回归报告 |
| `scripts/eval_fixed.py`、`scripts/run_g2.sh` | G2 固定状态诊断 |
| `scripts/aggregate_local.py`、`scripts/plot.py` | 本地分析：只读 Table 1 规范记录 |
| `results/`、`reports/`、`figures/` | 自动生成的结果、报告和图 |
| `work_dirs/exp6_v3/H/` | 冒烟运行、位姿表、G2 记录、日志（不进 git） |
| `work_dirs/cvpr_results/table1/g3/closed_loop/outdoor/V3-H*/` | 正式闭环的完整轨迹、规划器步日志（含逐步分数与 H3 信息）、服务记录 |

## 复现
```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
# 位姿表（magician 环境，一次性）
CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python experiments/exp6_v3/H/scripts/build_pose_tables.py
# 闭环（Table 1 队列；任务行 V3-H1|V3-H2|V3-H3:<scene>:<start>）
bash experiments/cvpr_results/table1/scripts/t1_queue.sh <GPU> experiments/cvpr_results/table1/configs/jobs_h.txt
# 单条 episode
V3_CKPT= H_POLICY= env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python experiments/exp6_v3/H/scripts/sim_out_h.py \
  --v3-ckpt work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy h3 --group V2M --scene barts --start 0 --root <DIR>
# G2 与汇总
bash experiments/exp6_v3/H/scripts/run_g2.sh 0 B0 0/1
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
$PY experiments/exp6_v3/H/scripts/eval_fixed.py --analyze
$PY experiments/exp6_v3/H/scripts/aggregate_local.py
$PY experiments/exp6_v3/H/scripts/plot.py
$PY experiments/exp6_v3/H/scripts/g0_regression.py
```
