# EXP6 V3.1 — V3-T0 架构（Simple 3D VAE + 原 Ray Encoder + 原 2D Flow）的 Outdoor-only 从零重训

任务书与审计 / 协议文件见 `../exp6_v4/`（`AUDIT_BASELINES.md`、`CHECKPOINT_LINEAGE.md`、`FROZEN_PROTOCOL.md`）；本目录的 `FROZEN_PROTOCOL.md` 只列 V3.1 专属要点。原 V3 Task 0（Mixed from scratch）保持不动，作为历史参照。

| 文件 | 内容 |
|---|---|
| `configs/model_v3_1.json` | = `../exp6_v3/configs/model_v3.json`，`xattn.enabled = false`（T0 的有效配置） |
| `configs/outdoor_train.json` | V3 `train_v3.json` 配方（lr 1e-4、λ_rec 1、β 0 → 0.02、EMA 0.999、40k 步、64 视角 / 8 状态）去掉室内域；`accum = 2`；EMA 保存步；固定 EMA@40k；无早停 |
| `scripts/trainer_outdoor.py`、`scripts/train.py` | 训练器 = `seen2scene/v3/trainer.py` 去掉室内域（差异见文件头：单域采样、整步全局射线数归一化、EMA 保存步、无早停） |
| `scripts/train_v31.sh`、`scripts/g0_v31.sh` | 正式启动、G0 冒烟（overfit / profile / resume / 1k short） |
| `tests/test_v31_model.py` | G0-8：架构 = T0（参数量与 seed-0 初始 sha 与 T0 相同）、μ / logvar / KL / 重建有限且可反传 |
| `report/`、`reports/` | 同 V4 的报告集（`V3_1_*`） |

G3 / G1 / G2 用 `../exp6_v4/scripts/run_queue_v4.sh`（组 `V31`，V3 规划器路径 `../exp6_v3/shared/v3_{in,out}_planner.py`，`V3_CKPT = work_dirs/exp6_v3_1/checkpoints/final.pt`）。输出：`work_dirs/exp6_v3_1/`（训练）、`work_dirs/exp6_v4/eval/`（评估，两组同根）。
