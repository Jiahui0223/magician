# Experiment G — Generative vs Deterministic

任务书原文：`TASK_BOOK.md`。问题：其余全部相同时，生成式（flow）是否优于确定性占据预测？关键新结果是确定性 U-Net **D0**；G1（Table 3 `V3-T0-K1D2`）与 G4（Table 1 `V3-T0-D1`）复用存档，G0（V3-T0 权重 + 零噪声解码）只作诊断。

冻结协议：`reports/G0_PROTOCOL.md`（G-v1，sha256 `4aa44592ce8e3bf5b33243d7ba8273a91f7102746c7169871be5be20d8d0c5be`，2026-11-14 21:54，在任何 D0 训练步之前）。

## 当前状态（2026-11-14）
| 阶段 | 状态 | 文件 |
|---|---|---|
| G-A 审计 + 冻结协议 | ✅ | `reports/G0_PROTOCOL.md`、`logs/EVENTS.md` |
| G-A 实现 | ✅ 只在本目录 `scripts/`；`seen2scene/` 与其他实验文件未改 | `scripts/d0_head.py`、`train_g.py` 等 |
| G-A 校验 (a)–(d) | 见 `logs/EVENTS.md` | `work_dirs/exp6_v3/experiment_G/checks/` |
| G-B D0 训练 40K | 见 STATUS | `work_dirs/exp6_v3/experiment_G/d0/` |
| G-C 固定状态 | ⏳ 已准备（`post_train.sh fixed`） | `scripts/g_fixed.py`、`g_fixed_worker.sh`、`g_analyze_fixed.py` |
| G-D 闭环 + Table 3 | ⏳ 已准备（`post_train.sh closed`） | `scripts/g_episode.sh`、`sim_out_g.py`、`g_out_planner.py`、`g_out_cov.py`、`g_collect.py` |

## 目录
| 路径 | 内容 |
|---|---|
| `scripts/d0_head.py` | D0 / G0 的全部函数：`det_forward`、`det_terms`（替换 `fm_terms`）、`det_per_view_loss`、`det_sample`（替换 `v3.model.sample`）、`DetPredictor`、`ZeroNoisePredictor`、`load(mode)` |
| `scripts/train_g.py`、`run_train.sh` | 训练入口（`--head det` 只在模块层面替换 4 个函数；`--head flow` 不替换任何东西）与 4 卡启动器（环境同 `shared/train_arm.sh`） |
| `scripts/test_d0_unit.py` | 单元测试（b） |
| `scripts/check_train_vs_t0.py` | 与存档 V3-T0 运行对比（a: flag-off 逐位；d: D0 第 1 步 rec / kl 逐位等） |
| `scripts/g_out_planner.py`、`sim_out_g.py`、`g_out_cov.py`、`g_episode.sh` | 闭环：Table 3 的 K = 1 + D1 路径 + `--g-mode flow|g0|d0`；官方覆盖回放；单 episode 队列任务（幂等、原子认领、≤ 3 次） |
| `scripts/g_smoke.sh`、`g_smoke_check.py` | 闭环 smoke / 回归（G1R 必须逐位复现 V3-T0-K1D2） |
| `scripts/g_fixed.py`、`g_fixed_worker.sh`、`g_analyze_fixed.py` | 755 状态固定状态评估（CT / U 回放链，四行同一进程）与汇总 |
| `scripts/g_select_final.py`、`post_train.sh`、`make_queue_tasks.py`、`g_collect.py` | 训练后：冻结规则选模 → 检查 → 入队 → Table 3（csv / md / tex，配对 CI vs G4 与 G1，分组） |
| `work_dirs/exp6_v3/experiment_G/` | 大文件：`d0/{runs/main,logs,checkpoints}`、`checks/`、`smoke/`、`fixed/`、`closed_loop/<ARM>/<scene>_s<k>/` |

## 变更日志
- 2026-11-14 21:45 审计；21:54 冻结 G-v1；实现与校验见 `logs/EVENTS.md`。

## G2 后续（2026-11-15 起；冻结协议 `reports/G2_PROTOCOL.md`，sha256 `83dd7436…`）
| 项 | 脚本 | 结果 |
|---|---|---|
| 1–2 概率质量 + 歧义分层 | `g2_prob.py`（VALSEL-in / S-c-out，与 eval.jsonl 逐位回归）、`g2_pool.py`（755 状态 F-pool）、`g2_analyze.py` | `results/g2/G2_ITEMS12.md`、`item1_summary.csv`、`item2_strata.csv`、`g2_items12.json`；`figures/g2/{reliability,spread_skill}.*` |
| 3 闭环时间分析 | `g2_temporal.py` | `results/g2/temporal/`、`figures/g2/temporal_diff.*` |
| 4 定性图 | `g2_figs.py`（预先声明的 GT-only 选图规则） | `figures/g2/qualitative_*`、`results/g2/qualitative_selection.json` |
| 5 室内 MP3D-150 | `g_in_planner.py`、`sim_in_g.py`、`g_in_episode.sh`、`g2_in_smoke_check.py`、`g2_indoor_collect.py`；入队 `g2_queue.py` | `results/g2/indoor/` |
- 大文件：`work_dirs/exp6_v3/experiment_G/g2/{prob,pool,indoor,indoor_smoke,test}/`。

## G3 后续（2026-11-15 起；冻结协议 `reports/G3_PROTOCOL.md`，sha256 `34f5002b…`）
Ours（G4）vs 把 K = 4 个生成样本合成一张图（M-raw：平均 (b, y) 后解码；M-depth：≥ 2/4 投票 + 命中深度平均），再经 K = 1 交给同一个 D1。
| 项 | 脚本 | 结果 |
|---|---|---|
| 合并钩子 + 闭环 | `m_combine.py`、`g3_out_planner.py`、`sim_out_g3.py`、`g3_episode.sh`、`g3_in_planner.py`、`sim_in_g3.py`、`g3_in_episode.sh`、`g3_queue.py` | `work_dirs/exp6_v3/experiment_G/g3/{closed_loop,indoor}/` |
| 校验 | `test_m_combine.py`、`g3_smoke_check.py` | `work_dirs/.../g3/{checks,smoke}/` |
| 固定状态 | `g3_fixed.py`、`g3_analyze_fixed.py` | `results/g3/fixed_table.*` |
| 最终表 + 时间曲线 | `g3_collect.py` | `results/g3/final_table_{outdoor,indoor}.{md,csv,tex}`、`G3_TABLES.md`、`figures/g3/temporal_vs_g4.*` |
| 定性图 | `g3_figs.py` | `figures/g3/averaging_*` |
