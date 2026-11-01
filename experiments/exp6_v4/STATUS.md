# EXP6 V4 / V3.1 — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-11-01 16:10（P6 Experiment A 运行中：固定状态 A0/A1 + A1 的 40 条 G3 在 gpu115 新作业 3443137 上）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 状态 |
|---|---|---|---|---|---|---|
| P3-V4 | **V4 Outdoor-only 正式训练**（40k 步，64 视角 / 8 状态 / 步，lr 2e-4，固定 EMA@40k；`train_v4.sh`，torchrun 4 卡） | gpu115 GPU 0–3 | 3435033（步 .9 `v4-train`） | `work_dirs/exp6_v4/logs/train.log`、`runs/main/{log,eval}.jsonl` | 20:47 | ✅ 11-01 01:07 EXIT 0，40,000 步；均值 0.472 s/步、峰值 18.4 GB/卡，墙钟 4.34 h ≈ 17.3 GPU-h；S-c ES 7.23 (5k) → 3.74 (40k)；final.pt = EMA@40k sha `1b31afff…`；报告 `report/OUTDOOR_TRAINING.md` |
| P3-V3.1 | **V3.1 Outdoor-only 正式训练**（40k 步，64 视角 / 8 状态 / 步，accum 2，lr 1e-4，固定 EMA@40k；`train_v31.sh`） | gpu117 GPU 0–3 | 3429774（本地 setsid，manifest job None） | `work_dirs/exp6_v3_1/logs/train.log`、`runs/main/{log,eval}.jsonl` | 20:47 | ✅ 11-01 10:10 EXIT 0，40,000 步；均值 1.29 s/步、峰值 23.4 GB/卡，墙钟 14.39 h（跨 11-01 02:00 夏令时回拨，按实际经过时间）≈ 57.6 GPU-h；S-c ES 8.88 (5k) → 6.19 (40k)，active units 16/16（无后验坍缩，KL/unit 0.38）；final.pt = EMA@40k sha `13df0a45…`；报告 `../exp6_v3.1/report/OUTDOOR_TRAINING.md` |
| P4-V4 链 | `post_arm_v4.sh v4`：等 `EXIT 0` → `select_final.py v4`（固定 EMA@40k → `checkpoints/final.pt`）→ 2 步冒烟 → V4 的 G3（10 MP3D + 30 Macarons++，4 worker）→ 室外覆盖率 | gpu115 | 3435033（步 .10 `v4-post`） | `work_dirs/exp6_v4/logs/post_v4.log` | 01:07 | ✅ 01:58 完成：final.pt（EMA@40k）→ 冒烟 OK → 10 室内 + 30 室外 + 30 覆盖率，0 失败（≈ 3.3 GPU-h）；V4 单组临时表 `reports/g3_v4_interim_v4.md`（正式表等 V3.1） |
| P4-V3.1 链 | `post_arm_v4.sh v31`：等 `EXIT 0` → final.pt（EMA@40k）→ 冒烟 → V3.1 的 G3（4 worker）→ 覆盖率 | gpu117 | 3429774（到期 14:50） | `work_dirs/exp6_v4/logs/post_v31.log` | 10:10 | ✅ 10:37 完成：final.pt（EMA@40k）→ 冒烟 OK → 10 室内 + 30 室外 + 30 覆盖率（两节点 8 worker，0 失败，≈ 3.4 GPU-h） |
| P4-V3.1 辅助链 | `post_v31_helper.sh`：等冒烟 OK → 加入 V31 的 G3 队列（4 worker，claim 共享）→ 等 40 条闭环齐 → 覆盖率 → 等 `POST V31 DONE` → `run_g12_arm.sh v31` → `analyze_g12_v4`（合并两组）→ `analyze_g3_v4` → `training_summary v31` → `make_tables_v4` | gpu115 | **3435072**（10:00 启动，24 h） | `work_dirs/exp6_v4/logs/post_v31_helper.log` | 10:03 | ✅ 10:51 全部完成：V3.1 G1/G2 365 态（0.9 GPU-h）→ `reports/g12_v4_s_final_v4.md`、`reports/g3_v4.md`、`reports/V4_COMPARISON_TABLES.md`、`../exp6_v3.1/report/OUTDOOR_TRAINING.md` |
| P4-G12 | 改为按组分跑：`run_g12_arm.sh v4`（V4 + B2 回放，tag `s_final_v4`）现在在 gpu115 跑；`run_g12_arm.sh v31` 等 V3.1 final.pt 后跑；`analyze_g12_v4.py --tag s_final_v4 --tag31 s_final_v31` 合并（同状态 / 候选 / 噪声） | gpu115 | 3435033（步 .11 取消，新步 `v4-g12`） | `work_dirs/exp6_v4/logs/run_g12_v4.log` | 02:01 | ✅ V4 部分 02:15 完成（365 状态，B2 回放 0 差异，≈ 0.9 GPU-h）；V3.1 部分待 final.pt |
| G0-V4 | V4 冒烟链：overfit（GPU 0）‖ 单卡 resume A/B（GPU 1）→ 4 卡 1k short | gpu115 | 3435033（ssh） | `work_dirs/exp6_v4/logs/{g0,smoke_overfit,resume_*,smoke_short}.log` | 20:19 | ✅ 20:44 全部通过（overfit drop 0.927 / sign 0.976；resume 精确；1k short 1.77 → 0.128，0.360 s/步，10.6 GB） |
| G0-replay | 冻结规划器 2 步回放（T0 / V2M × 室内 / 室外） | gpu117 GPU 1–2 | 3429774 | `work_dirs/exp6_v4/eval/g0_replay/` | 20:28 | ✅ 20:29 4/4 逐位 PASS |
| G0-V3.1 | V3.1 冒烟链：overfit → 4 卡 profile → 4 卡 resume A/B → 4 卡 1k short | gpu117 | 3429774 | `work_dirs/exp6_v3_1/logs/` | 20:29 | ✅ 20:42 全部通过（overfit drop 0.899；profile 0.396 s/步；resume PASS；1k short 3.36 → 0.359，0.397 s/步，4.1 GB） |

**GPU / 作业**（2026-10-31 20:00 `squeue`）
| 作业 | 节点 | 到期 | 用途 |
|---|---|---|---|
| 3429774 `vscode-dev` | gpu117 4 × L40S | 11-01 ≈ 15:49 | V3.1 G0 + 正式训练（更长的分配给更慢的 V3.1） |
| 3435033 `exp6-v3-aux` | gpu115 4 × L40S | 已于 11-01 10:00 结束 | V4 G0 + 正式训练 + V4 的 G3 + V4 的 G1/G2（全部完成） |
| 3435072 `exp6-v3-aux2` | gpu115 4 × L40S（24 h） | 11-02 ≈ 10:00 | 10:00 启动：V3.1 的 G3 辅助、G1/G2、统一分析 |

## P0 审计 — ✅（2026-10-31 19:55–20:10）
| 编号 | 内容 | 状态 |
|---|---|---|
| P0.1 | V2 Original / V2-Mix / E2-D100 / V3 T0-T1 的训练脚本、配置、manifest、checkpoint sha256 | ✅ `AUDIT_BASELINES.md`、`CHECKPOINT_LINEAGE.md` |
| P0.2 | 划分：Train-6 vs Dev-3 / Test-3；MP3D train / VALSEL / 5 评估楼层 | ✅（单测断言） |
| P0.3 | V2 Original 的 G3 数值（Task 2 V2O，同协议） | ✅ MP3D 0.315 / 0.441 / 0.179；Macarons++ 0.724 / 0.898 / 0.578 |
| P0.4 | 冻结协议草案 v1.0 | ✅ `FROZEN_PROTOCOL.md`（G0 后冻结） |

## P1 代码 — ✅（20:10–20:30）
| 编号 | 内容 | 状态 |
|---|---|---|
| P1.1 | `seen2scene/futureview/model.py`：`RayXAttn` + 开关；V2 关闭时逐位相同 | ✅ `tests/futureview` 12 passed |
| P1.2 | `exp6_v4/scripts/train_v4.py`、`exp6_v3.1/scripts/trainer_outdoor.py`、配置、启动脚本 | ✅ |
| P1.3 | G0 单测 10 项（`exp6_v4/tests`、`exp6_v3.1/tests`） | ✅ 10 passed |
| P1.4 | G3 脚本（V4 规划器子进程 / 模拟器 / 队列 / post 链）、FT 与 n_win 草案配置 | ✅（G3 待权重） |

## P2 G0 — ✅（2026-10-31 20:45；`report/G0_VALIDATION.md`）
| 门 | 内容 | V4 | V3.1 |
|---|---|---|---|
| G0-1 | 基线审计 | ✅ | ✅ |
| G0-2 | 数据划分 | ✅ 单测 | ✅ 单测 |
| G0-3 | DDP / 累积归一化 | ✅ 单测 + 4 卡 1k short | ✅ 单测 + 4 卡 1k short |
| G0-4 | V4 形状 | ✅ | — |
| G0-5 | X-Attn 梯度 | ✅ 单测 + overfit 第 10 步 | — |
| G0-6 | 相机条件 | ✅ | — |
| G0-7 | 可见性 mask | ✅ | — |
| G0-8 | V3.1 VAE | — | ✅ 单测 |
| G0-9 | 零泄漏 | ✅ | ✅ |
| G0-10 | resume | ✅ 单卡 A/B 精确 | ✅ 4 卡 A/B |
| G0-11 | 基线回放 | ✅ 开关关闭逐位；✅ 规划器回放 4/4 | ✅ 规划器回放 4/4 |
| G0-12 | 吞吐 1k | ✅ 0.360 s/步、10.6 GB（小场景子集） | ✅ 0.397 s/步、4.1 GB（小场景子集） |

## P3 正式训练 — ✅（2026-10-31 20:47 启动，V4 11-01 01:07 / V3.1 10:10 结束；协议 §9 冻结于 20:46；两组均无 resume、无偏差）
## P4 冻结 checkpoint + G1/G2/G3 双域 — ✅（11-01 10:51；`report/G3_OUTDOOR.md`、`G3_INDOOR_ZEROSHOT.md`、`G1_G2_EVAL.md`；表 `reports/V4_COMPARISON_TABLES.md`）
## P5 报告 / Go-No-Go — ✅（11-01 11:10；`report/FINAL_REPORT.md` §5：MP3D 微调 Conditional Go、U0 Go（若要对 X-Attn 下结论）、延长 60k No-Go、n_win 独立决定；**全部等待用户决定，未启动**）

## P6 Ablation Experiment A — DistP1 评分 S10（A0）vs D100（A1）— 🔄（用户任务书 2026-11-01 ~15:50；目录 `ablation_study/experiment_A_distp1_dense_vs_sparse/`）
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 状态 |
|---|---|---|---|---|---|
| P6.1 | 实现审计（调用链；唯一变量 = `caster.band_dil`）；`sim_{in,out}_v4.py` 加 `V4_P1_MODE`（默认 b2） | CPU | — | `report/A_IMPLEMENTATION_AUDIT.md` | ✅ |
| P6.2 | G0：A0 3 步冒烟逐位复现存档 V4（两域）；A1 冒烟运行；单变量 GPU 测试（输入体 / 结构 / 粗网格逐位相同，band_dil 稀疏 903k vs 稠密 1,282k 体素） | gpu115 GPU 0–1 | 3443137（本地） | `work_dirs/exp6_v4/ablation_study/experiment_A/smoke/g0_check.json` | ✅ 16:00 PASS |
| P6.3 | 固定状态 A0 / A1（VALSEL 90 + Dev-3 275；同一 V4 预测两种 band_dil；A0 与存档 V4 行逐位核对） | gpu115 GPU 0–3（与 P6.4 共用） | 3443137 | `.../experiment_A/logs/a_fixed_driver.log` | 🔄 16:01 起 |
| P6.4 | A1 G3：10 MP3D + 30 Macarons++（`T3_P1_MODE=full`）→ 覆盖率 → `a_analyze_g3.py` | gpu115 GPU 0–3 | 3443137 | `.../experiment_A/logs/a_post.log` | 🔄 16:01 起 |
| P6.5 | 报告 `A_FIXED_STATE.md`、`A_G3_RESULTS.md`、`A_FINAL_REPORT.md`（GO / NO-GO / INCONCLUSIVE） | CPU | — | | ⏳ |

**GPU / 作业（16:00）**：3429774（gpu117）与 3435072 已到期；现用 3443137 `vscode-dev` gpu115 4 × L40S（到 11-04 ≈ 15:10）。
