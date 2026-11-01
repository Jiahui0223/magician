# EXP6 V4 Ablation Study — Experiment A：DistP1 评分用 S10 稀疏 P1（A0）还是 D100 稠密 P1（A1）

任务书：用户 2026-11-01 粘贴的 "EXP6 V4 Ablation Study — Experiment A"。只回答一个问题：**从零用 D100 训练的 V4，DistP1 的历史表面排除依据应当用与模型输入一致的 D100，还是沿用 S10？** 模型、权重、候选、K = 4、噪声、n_win、距离权重、合法性全部不变；唯一变量 = 评分用 `band_dil`。

| 文件 | 内容 |
|---|---|
| `configs/experiment_A.json` | 两组定义（A0 = V4 G3 原运行，`T3_P1_MODE=b2`；A1 = `T3_P1_MODE=full`）、checkpoint sha、不变项、判定规则 |
| `scripts/a_g0_smoke.sh`、`a_g0_check.py` | G0 闭环：A0 3 步冒烟逐位复现存档 V4 episode；A1 冒烟运行 |
| `tests/test_single_variable.py` | G0 单变量证明（GPU）：同一批已执行记录 → B2 融合与 Full 融合的输入体 / 结构 / 域 / 投影体 / 粗网格逐位相同，只有 `band_dil` 不同 |
| `scripts/a_fixed.py`、`a_run_fixed.sh`、`a_fixed_analyze.py` | 固定状态：同一 V4 预测用两种 band_dil 评分（VALSEL 90 + Dev-3 275 态）；A0 与存档 V4 行逐位核对；被选增益 pp / Top-1 / 平均分 / 被判已知比例 / 选择不同比例 |
| `scripts/a_run_queue.sh`、`a_post.sh`、`a_analyze_g3.py` | A1 的 G3（10 MP3D + 30 Macarons++，同起点 / 候选 / 种子）与 A1 − A0 配对分析（历史组参照） |
| `report/` | `A_IMPLEMENTATION_AUDIT.md`、`A_FIXED_STATE.md`、`A_G3_RESULTS.md`、`A_FINAL_REPORT.md` |
| `results/`、`figures/` | `a_fixed_summary.json`、`a_g3.{json,md}`、覆盖率曲线 |

仓库级改动：`exp6_v4/scripts/sim_{in,out}_v4.py` 读环境变量 `V4_P1_MODE`（默认 `b2`，V4 原 G3 不变），`out_cov_v4.py` 读 `V4_EVAL_ROOT`。大文件在 `work_dirs/exp6_v4/ablation_study/experiment_A/`（不入库）。
