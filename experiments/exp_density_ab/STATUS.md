# EXP Density A/B — STATUS（宏观状态板）

图例：✅ 完成 · 🔄 运行中 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 此刻在跑什么
| 项 | 进程 | 节点 / GPU | 日志 | 开始 | 预计 |
|---|---|---|---|---|---|
| P2-T1/T2 固定状态 12 组 | 7 个 da_fixed 进程（valsel × 3、dev3 B0 × 2、dev3 B2 × 2） | gpu115 / 0–3（交互作业 3443137） | `work_dirs/exp_density_ab/logs/fixed_*.log` | 2026-11-03 00:42 | 约 30–40 分钟 |
| P3-B1 / B2 G3（7 组新配置，280 个 episode） | `run_g3_local.sh`：8 个 da_g3_queue2 worker（每 GPU 2 个，同时最多 1 个 MP3D），第一批完成后自动接第二批 | gpu115 / 0–3 | `work_dirs/exp_density_ab/g3/logs/queue2_*` | 2026-11-03 01:16（01:04 起由旧调度先跑） | 约 4–6 小时 |
| 额外节点 | sbatch 3460841（4 × L40，第一批 → 第二批共用队列） | PENDING（Priority） | `logs/sbatch_3460841.out` | — | 取决于排队 |

## P1 Task 0 — 核验
| id | 项 | 状态 | 结果 |
|---|---|---|---|
| P1-T0.1 | 三个 checkpoint 的路径、SHA、EMA、步数、严格加载 | ✅ | 三个都是 `ema`，步数分别为 17500、5000、5000；24,822,114 参数；严格加载 0 缺失 / 0 多余 |
| P1-T0.2 | 两组 E2 初始化是否同一份 B2 | ✅ | 配置 init sha 都等于 B2；两组的 ema_000000 与 B2 逐张量相等 |
| P1-T0.3 | 两组 E2 对齐（配置、元数据、日志、采样结构） | ✅ | 配置只差 4 个说明字段；views、lr 逐步相同；417 对分片的结构数组逐位相同 |
| P1-T0.4 | S10 / D100 监督集合差异 | ✅ | 见 CHECKPOINT_AUDIT §4 |
| P1-T0.5a | 固定状态评估器回归（smoke） | ✅ | dev3 7 个状态 63 项检查、valsel 10 个状态 80 项检查，全部逐位相等 |
| P1-T0.5b | 闭环 G0 复现（5 组历史配置 × 1 个室内 + 1 个室外 episode） | ✅ | 10/10 个 episode 的执行动作、覆盖曲线、规划器每步分数与历史逐位相同 → 5 组历史 G3 可复用 |
| P1-T0.5c | 由历史 episode 文件重算 G3 AUC / Final | ✅ | 5 组到 3 位小数与报告一致 |
| P1-T0.6 | S10 / D100 两种输入下严格加载 | ✅ | 与 T0.1 同一模型；两种输入的前向在 smoke 中都已运行 |
| P1-T0.7 | B0、B1、B2 是否共享同一权重 | ✅ | 三组都使用 V2M 组 checkpoint（sha 343f41ab…），T3_CKPT 未设置；P1 模式分别为 sparse、full、b2 |

## P2 Task 1 + Task 2 — 固定状态 12 组（VALSEL 90 + Dev-3 275）
| id | 项 | 状态 | 结果 |
|---|---|---|---|
| P2-S | smoke（1 个室内 + 1 个室外 episode） | ✅ | 回归检查全部通过；峰值显存 12.1 GB（室内）、3.2 GB（室外） |
| P2-R | 完整运行，39 个 episode | ✅ 01:10 | 365/365 个状态、3195 项回归检查全部逐位一致；VALSEL 分片 1/3 OOM 后已重跑（偏差 §9） |
| P2-A | 分析（G1 / G2 / 评分掩码） | ✅ | `results/fixed_summary.json`、`results/tables_fixed.md` |

## P3 Task 3 — 闭环 G3
| id | 项 | 状态 | 结果 |
|---|---|---|---|
| P3-B1 | 第一批 4 组新配置（160 个 episode） | 🔄 | 1 个 MP3D episode OOM → 重跑；调度改为每 GPU 最多 1 个 MP3D（偏差 §9） |
| P3-B2 | 第二批 3 组新配置（120 个 episode） | ⏳ 第一批全部认领后开始 | — |
| P3-A | G3 分析 | ⬜ | — |

## P4 Task 4 — 配对补训
| id | 项 | 状态 | 结果 |
|---|---|---|---|
| P4 | 判定 | ✅ NOT REQUIRED（依据 P1-T0.2 / T0.3，见 FROZEN_PROTOCOL §5） | 本轮不训练 |
