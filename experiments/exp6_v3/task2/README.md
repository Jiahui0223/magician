# EXP6 V3 Task 2 — Coverage-Aware NBV Planning

任务书：用户 2026-10-31 17:00 指令（V4 暂停；不训练；不改 Task0 / Task1 / Ablation A / B / C 的冻结结果）。协议：`protocol.md` v1.0（冻结 sha `configs/FROZEN_SHA256_E1.json`）。状态板：`STATUS.md`。总报告：`reports/`。

| 目录 | 内容 |
|---|---|
| `common/t2_scorers.py` | 评分器族（S0 原 DistP1 逐位复现、S1a–d 表面足迹面积、S2 覆盖率尺度软新颖性、S3 稠密几何一致性、组合、GT 全局变体）；只用预测 / 真实深度与真实观测历史，不读 GT 场景 |
| `exp1_coverage_scoring/scripts/e1_score_fixed.py` | 固定状态（Pool CL 1,461 + Pool FX 365）上 GT 深度与 Experiment A 缓存的 K = 4 样本经全部评分器 → `work_dirs/exp6_v3/task2/exp1/scores/`；每状态断言 S0 逐位 |
| `exp1_coverage_scoring/scripts/e1_analyze.py` | E1-GT / E1-Pred 指标、dev 集 G2 Screening、校准、失败案例、图 → `results/`、`figures/`、`report/` |
| `exp1_coverage_scoring/scripts/t2_scorer_patch.py` + `t2_{in,out}_planner.py` + `t2_{in,out}_sim.py` | G3：在冻结的 B2 / V3 规划器子进程中用 `T2_SCORER` 替换 `score_state` 的 DistP1（其余逐位不变） |
| `exp2_long_range/` | R0 回放、R1 评分区域 counterfactual、R2 frontier-aware 规划器、R3 候选空间 counterfactual、R4 组合（MP3D 优先） |
| `tests/` | `test_scorers.py`（合成几何单测）、G3 回放 / 冒烟 |

大文件（分数缓存、闭环输出）在 `work_dirs/exp6_v3/task2/`，不提交。
