# EXP6 V4 / V3.1 — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 20:50（G0 全部通过；冻结 + 正式训练启动中）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 状态 |
|---|---|---|---|---|---|---|
| G0-V4 | V4 冒烟链：overfit（GPU 0）‖ 单卡 resume A/B（GPU 1）→ 4 卡 1k short | gpu115 | 3435033（ssh） | `work_dirs/exp6_v4/logs/{g0,smoke_overfit,resume_*,smoke_short}.log` | 20:19 | ✅ 20:44 全部通过（overfit drop 0.927 / sign 0.976；resume 精确；1k short 1.77 → 0.128，0.360 s/步，10.6 GB） |
| G0-replay | 冻结规划器 2 步回放（T0 / V2M × 室内 / 室外） | gpu117 GPU 1–2 | 3429774 | `work_dirs/exp6_v4/eval/g0_replay/` | 20:28 | ✅ 20:29 4/4 逐位 PASS |
| G0-V3.1 | V3.1 冒烟链：overfit → 4 卡 profile → 4 卡 resume A/B → 4 卡 1k short | gpu117 | 3429774 | `work_dirs/exp6_v3_1/logs/` | 20:29 | ✅ 20:42 全部通过（overfit drop 0.899；profile 0.396 s/步；resume PASS；1k short 3.36 → 0.359，0.397 s/步，4.1 GB） |

**GPU / 作业**（2026-10-31 20:00 `squeue`）
| 作业 | 节点 | 到期 | 用途 |
|---|---|---|---|
| 3429774 `vscode-dev` | gpu117 4 × L40S | 11-01 ≈ 15:49 | V3.1 G0 + 正式训练（更长的分配给更慢的 V3.1） |
| 3435033 `exp6-v3-aux` | gpu115 4 × L40S | 11-01 ≈ 11:09 | V4 G0 + 正式训练 + V4 的 G3 |
| 3435072 `exp6-v3-aux2` | 排队（24 h） | — | 任一作业结束后启动；用于 resume / G3 |

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

## P3 正式训练 — ⬜
## P4 冻结 checkpoint + G1/G2/G3 双域 — ⬜
## P5 报告 / Go-No-Go — ⬜（第二阶段 MP3D 微调、U0、n_win 闭环均不自动启动）
