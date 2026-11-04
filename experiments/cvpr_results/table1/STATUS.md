# STATUS — CVPR Table 1（Macarons++ Active Mapping）

## 此刻在跑什么
- 无。所有任务已在 2026-11-03 21:00 前完成。

## T1 编号进度
| 编号 | 项 | 状态 | 关键数字 |
|---|---|---|---|
| T1.0 | 场景审计（`SCENE_REGISTRY.yaml` / `DATASET_AUDIT.md`） | ✅ | 15 个场景；sofia_church 不可运行 → PENDING |
| T1.1 | 模型 / checkpoint 审计（`MODEL_REGISTRY.yaml` / `CHECKPOINT_AUDIT.md`） | ✅ | 13 个模型 READY，sha 已核对 |
| T1.2 | 历史 episode 审计 + 导入（`EVALUATION_AUDIT.md`） | ✅ | 390 个，逐位重放一致 |
| T1.3 | 协议冻结（`PROTOCOL.md` v1.0 / `FROZEN_CONFIG.yaml`） | ✅ | 偏差记录 4 条（调度 2 条、OOM 重试 1 条、标签修正 1 条） |
| T1.4 | 回归重放（barts_s0） | ✅ | 8 / 8 主模型逐位一致（17:39） |
| T1.5a | 补跑：MAG-B（gpu115） | ✅ | 40 个，约 30 min / 个 |
| T1.5b | 补跑：主模型 V2O / B0 / B2 / E2D / V3 / V4 / V5 | ✅ | 280 个 |
| T1.5c | 补跑：可选模型 MAG-G / E2S / B1 / V3.1 / V4C | ✅ | 200 个 |
| T1.5d | 加入 gpu114（作业 3461041，4 × L40） | ✅ | 每卡 3 个 worker（19:05 起） |
| T1.5e | ❌ dunnottar 双图任务 OOM → 独占阶段（每卡 1 个） | ✅ | 34 个，20:38–21:00，全部成功 |
| T1.6 | `validate.py` | ✅ | 910 / 910 有效，bad 0 |
| T1.7 | `aggregate.py`（官方 + 单调）、`plot.py`（56 个文件） | ✅ | — |
| T1.8 | 报告：BASELINE_REPRODUCTION / COMPUTE_COST / RESULTS_ANALYSIS（+ mono）/ MISSING_EPISODES / FINAL_REPORT | ✅ | — |
| T1.9 | sofia_church → All-15（需要 protocol v1.1） | ⬜ | 等待用户决定 |
