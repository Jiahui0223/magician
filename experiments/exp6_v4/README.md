# EXP6 V4 — 3D U-Net + Ray-Aligned Cross-Attention + 原 2D Rectified Flow，Outdoor-only 从零训练

任务书：`../exp6_v3/EXP6_V4_V3_1_AI_HANDOFF.md` → `../exp6_v3/EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md`。本目录是 V4；V3.1（V3-T0 架构的 Outdoor-only 重训）在 `../exp6_v3.1/`。两者共用本目录的审计与协议文件。

| 文件 | 内容 |
|---|---|
| `AUDIT_BASELINES.md` | Phase 0 审计：V2 Original / V2-Mix(B0, B2) / E2-D100 / V3 T0-T1 的训练来源、结构、划分、P1 口径、G3 协议、V2 Original 的 G3 数值 |
| `CHECKPOINT_LINEAGE.md` | 全部 checkpoint 的路径、sha256、初始化来源、训练阶段标签 |
| `FROZEN_PROTOCOL.md` | 第一阶段冻结协议 v1.0（数据、预算、batch、优化、checkpoint 规则、G0、G1/G2/G3、第二阶段草案、U0） |
| `STATUS.md` | 状态板（每次运行启动 / 结束更新） |
| `configs/model_v4.json` | V2 `model_v1.json` + `ray_xattn`（dim 64、4 头、Fourier 4 频、W_o std 0.005） |
| `configs/outdoor_train.json` | V2 Original 配方 + 40k 预算 + D100 室外分片 + EMA 保存步 + 固定 EMA@40k |
| `configs/indoor_finetune_DRAFT.json`、`configs/nwin_ablation_DRAFT.json` | 第二阶段草案（**不启动**） |
| `scripts/train_v4.py` | 训练器（= `task1/scripts/t12_train.py` 的配方；损失 / 采样 / 评估函数原样导入；差异见文件头） |
| `scripts/train_v4.sh`、`scripts/g0_v4.sh`、`scripts/g0_replay.sh`、`scripts/g0_replay_check.py` | 正式启动、G0 冒烟、规划器回放核对 |
| `scripts/select_final.py` | 冻结规则发布 `checkpoints/final.pt`（固定 EMA@40k） |
| `scripts/{v4_in_planner,v4_out_planner,sim_in_v4,sim_out_v4,out_cov_v4}.py`、`run_queue_v4.sh`、`post_arm_v4.sh` | G3：冻结的 Task 2 模拟器 + V2 规划器路径（B2 口径）加载 V4 checkpoint；V31 组走 `../exp6_v3/shared` 的 V3 路径 |
| `tests/` | G0 单测：`test_v4_model.py`（G0-4/5/6/7/11a）、`test_split_and_leakage.py`（G0-2/9）、`test_loss_normalisation.py`（G0-3） |
| `report/` | `G0_VALIDATION.md`、`OUTDOOR_TRAINING.md`、`G1_G2_EVAL.md`、`G3_OUTDOOR.md`、`G3_INDOOR_ZEROSHOT.md`、`FINAL_REPORT.md` |
| `reports/` | `V4_COMPARISON_TABLES.md`、`V4_DEVIATIONS.md`、`V4_RESOURCES.md` |

**模型改动（仓库级）**：`seen2scene/futureview/model.py` 新增 `RayXAttn` 与 `V2Model.ray_xattn`（配置键 `ray_xattn.enabled`，默认关闭）；`RayEncoder.forward` 拆成 `point_features()` + `head()`（同一算子同一顺序，关闭时与 V2 逐位相同，`tests/futureview` 12 passed + `test_g0_11a`）。

**输出**：`work_dirs/exp6_v4/{runs/main, logs, checkpoints, eval}`（不入库）。
