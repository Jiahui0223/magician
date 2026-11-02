# EXP6 V5.2 —— Camera-Conditioned Ray Geometry Belief for Direct NBV Scoring（Pilot）

任务书：`EXP6_V5_2_ARCHITECTURE_AND_PILOT.md`（本目录；原放在 `experiments/exp6_v5.2/`，按任务书的目录名移到 `exp6_v5_2/`）。
一句话：复用 V5 的 D100 P1 + 3D U-Net + Ray-to-Memory Query + Multi-Surface Belief（V5 EMA@12000 warm-start），新增条件首中 Hazard，
直接用首中 PMF 计算候选相机的预计新增可见表面（+ 小权重的遮挡表面 / 未探索空间潜力）选 NBV；不用 2D Flow、不生成深度图、不调用 DistP1。

| 文件 | 内容 |
|---|---|
| `STATUS.md` | 状态板 |
| `AUDIT.md` | 源码 / 权重审计 |
| `ARCHITECTURE.md` | 实现架构与语义 |
| `FROZEN_PROTOCOL.md` | 冻结协议 v1.0（训练、Score、评估、smoke 与全量 G3 放行规则） |
| `CHECKPOINT_LINEAGE.md` | 谱系与 sha |
| `configs/` | `model_v52.json`、`train_v52.json`、`score_v52.json`（训练后标定 floor 冻结） |
| `scripts/` | 训练 `train_v5_2.py` / `train_arm.sh`；G0 `g0_*.sh`；规划器 `v52_core.py`、`v52_{in,out}_planner.py`、`sim_{in,out}_v52.py`、`run_queue_v52.sh`；评估 `eval_g1_g2.py`、`calibrate_floors.py`、`eval_belief_metrics_v52.py`、`analyze_g1_g2.py`、`eval_g3_smoke.py`；后续链 `post_v52_*.sh` |
| `tests/` | G0 单测（首中分布、可见性与遮挡、体素去重与距离、空间潜力、无 GT 泄漏 / 无 Flow） |
| `reports/` | G0、G1、G2、G3 smoke、最终报告 |

共享代码：`seen2scene/futureview/v52_model.py`、`v52_score.py`。大文件：`work_dirs/exp6_v5_2/`。
