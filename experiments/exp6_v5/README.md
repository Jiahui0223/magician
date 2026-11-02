# EXP6 V5 —— Geometry-Aware Ray-to-Scene Memory + Multi-Surface Belief（Outdoor-only Warm-start）

任务来源：`EXP6_V5_ARCHITECTURE_DESIGN.md`（用户确认的结构与数学语义）、`EXP6_V5_AI_IMPLEMENTATION_TASK.md`（工程任务与验收）。
一句话：从 V4 Outdoor-only EMA@40k 继承 3D U-Net 与 2D Rectified Flow，把固定 32 点射线采样 + Conv1d + Ray X-Attn
换成 Camera Ray Query + 几何引导的 Local / Coarse 场景记忆交叉注意力，每条射线得到 z_r 与可按深度查询的多表面
Surface Belief b_r(t)，Belief Summary 与 z_r 经 CondAdapter 进入原 Flow，仍生成 K = 4 未来 Depth / HIT；规划器冻结。

| 文件 | 内容 |
|---|---|
| `STATUS.md` | 状态板（每次运行开始 / 结束更新） |
| `AUDIT_V5.md` | 训练前审计（V4 checkpoint、形状、数据隔离、坐标、标签可行性、规划器、资源） |
| `ARCHITECTURE.md` | 实现架构与设计对应 |
| `FROZEN_PROTOCOL.md` | 冻结协议 v1.0（组、训练、λ、最终权重规则、评估、停止条件） |
| `CHECKPOINT_LINEAGE.md` | checkpoint 谱系与 sha |
| `configs/` | `model_v5.json`、`train_v5_outdoor.json`、`train_v4_continue.json` |
| `scripts/` | 标签（`build_multisurface_labels.py`、`run_labels*.sh`、`label_quality.py`）、审计（`audit_v4_shapes.py`）、训练（`train_v5.py`、`train_arm.sh`）、G0（`g0_*.sh`、`g0_*_check.py`）、规划器接入（`v5_core.py`、`v5_{in,out}_planner.py`、`sim_{in,out}_v5.py`、`run_queue_v5.sh`）、评估（`eval_g1_g2.py`、`eval_belief_gt.py`、`eval_belief_metrics.py`、`eval_g3.py`） |
| `tests/` | G0 单测（几何、检索 mask、多表面目标、前向 / 反向 / 无 grid_sample、Teacher 迁移、规划器与数据隔离） |
| `reports/` | G0、标签质量、训练、G1 / G2、G3、对比表、资源、偏差、最终报告 |
| `figures/` | 标签与评估图 |

共享模型代码：`seen2scene/futureview/v5_model.py`（旧 `model.py` 不改）。大文件：`work_dirs/exp6_v5/`（labels、runs、checkpoints、eval、eval_fixed、logs），不进 git。

运行（gpu 节点，seen2scene env）：
```bash
# 单测
for t in experiments/exp6_v5/tests/test_*.py; do PYTHONPATH=$PWD LD_PRELOAD=$S2E/lib/libtbb.so $S2E/bin/python $t; done
# 正式训练（4 卡，可恢复）
bash experiments/exp6_v5/scripts/train_arm.sh v5     # 或 v4c
# G3 worker（每 GPU 一个；job 文件 work_dirs/exp6_v5/g3_{jobs,cov}_{V5,V4C}.txt）
bash experiments/exp6_v5/scripts/run_queue_v5.sh 0 work_dirs/exp6_v5/g3_jobs_V5.txt
```

## 变更记录
- 2026-11-01：建立目录；审计、标签（K 16 / 存 8）、V5 模型、训练器、规划器接入、评估脚本、G0；协议 v1.0 冻结（λ_surf 0.37、λ_bridge0 0.096、固定 EMA@12000）。
