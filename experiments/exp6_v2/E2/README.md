# E2 —— E2-D100：V2-Mix 在 D100 数据上微调 5k 步

E2-D100 在 V2-Mix EMA@17.5k（即 B2 的权重）上，用 D100（稠密 P1）重建的 MP3D + Macarons++ 分片混合微调 5k 步。规划方式与 B2 相同：D100 输入，S10 DistP1 评分。
同时保留对照臂 **E2-S10**：同一配方，但用稀疏 S10 分片，规划用 B0 方式。
配置：`configs/e2.json`。冻结协议：`experiments/exp5_future_view/task3_full_depth_p1/e2_pilot/protocol.md`。

## 1. 架构
与 B2 完全相同的 V2 结构和共享代码（见 `../B2/README.md` §1），参数 24,822,114。微调只改变权重，不改结构。checkpoint 内嵌的 model_cfg 已核对与 `task1/configs/model_v1.json` 一致。

## 2. 训练（以 `e2_pilot/configs/e2_d100_v1.json`、manifest 和日志为准）
| 项 | E2-D100 | E2-S10（对照） |
|---|---|---|
| 初始化 | V2-Mix EMA@17.5k（sha 343f41ab…，= B2 权重） | 同左 |
| 数据 | MP3D train（`e2/w2_d100`，D100 P1 重建）+ Train-6 去 S-c（`e2/out_views_d100`），各 32 views | 同配方，S10 稀疏分片（`e2_s10_v1.json`） |
| 优化 | 5k 步，lr 2e-5，warmup 500，余弦降到 0，AdamW (0.9, 0.99)，wd 0.01，bf16，EMA 0.999，seed 0，64 views / 8 states | 同左 |
| 选择 | **按协议固定为 EMA@5k，不做 checkpoint 选择**（`e2_pilot/protocol.md:50`；2026-11-02 更正：此前误写为“ES 比值最小”） | 同左（固定 EMA@5k；若按 ES 规则会选 1k，未采用） |
| 脚本 | `e2_pilot/scripts/e2_paths.py`、`e2_build.py`（数据）、`e2_train.py`；启动见 `scripts/train.sh` | 同左（`e2_train.py s10`） |

张量级核对（`results/smoke_*.json` 的 `lineage`）：E2-D100 第 0 步权重 = V2-Mix EMA@17.5k；EMA@5k 与初始化不同。

## 3. Checkpoint（`work_dirs/checkpoints/exp6_v2/E2/`，全部是指向原文件的相对软链接）
| 新路径 | 原路径（`work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/…`） | step | sha256 | 用途 |
|---|---|---|---|---|
| `e2d100_ema5000.pt` | `d100/ema_005000.pt` | 5000 | 61439612cb6132af… | **E2 模型**（EXP6 各版本中的 "E2-D100" 基线） |
| `e2d100_raw5000.pt` | `d100/ckpt_last.pt` | 5000 | f647eb5f7a7c… | 原始训练状态（模型 + EMA + AdamW + RNG） |
| `intermediate/e2d100_ema{000000…004000}.pt` | `d100/ema_*.pt` | 0–4000 | 见 MANIFEST | 中间 EMA（step 0 = 初始化） |
| `control_e2s10/e2s10_ema5000.pt` | `s10/ema_005000.pt` | 5000 | 578855bb14995626… | E2-S10 对照 |
| `control_e2s10/e2s10_raw5000.pt` | `s10/ckpt_last.pt` | 5000 | 2cb3f1357f02… | 对照原始训练状态 |
| `control_e2s10/intermediate/e2s10_ema{001000…004000}.pt` | `s10/ema_*.pt` | 1000–4000 | 见 MANIFEST | 对照中间 EMA；S10 的 step 0 与 D100 的 step 0 字节相同，只登记一次 |

E2 的初始化权重就是 `../B2/v2mix_ema17500.pt`，不在 E2 目录重复登记。

## 4. 入口（`scripts/`）
| 脚本 | 作用 |
|---|---|
| `verify.sh` | 核对 E2-D100 和 E2-S10 的 checkpoint（链接 / SHA / step / model_cfg / 严格加载 / 参数） |
| `smoke.sh [dev3\|valsel] [GPU]` | 最小推理 smoke + 新旧路径对比 + 与历史行对比（模型键 D100 = E2-D100，S10 = 对照） |
| `g12_fixed.sh` | 历史 G1 / G2 命令（共享，见 B2） |
| `g3.sh [OUT] [N]` | E2-D100 的 G3 闭环（`configs/g3_jobs_*.txt`）；`CONTROL=1` 时加跑 E2-S10（`configs/control_e2s10_g3_jobs_*.txt`）；默认输出 `work_dirs/exp6_v2/g3` |
| `train.sh [d100\|s10]` | 打印数据构建和训练的原命令；只有 `RUN=1` 时才执行训练，输出到新目录。本次未运行 |

## 5. 历史已报告结果（`results/historical/` 是指向原报告的软链接）
- **G1**：相对 B2，unknown 与 truly-new 的误差下降，通过。
- **G2**：Dev-3 排序差于 B2，未通过；MP3D 更好。见 `E2_G12_REPORT.md`、`g12_s5000.md`。
- **G3**（探索性，`E2_G3_REPORT.md`）：

  | 数据集（G3 范围） | E2-D100 AUC / Final | B2 | E2-S10 |
  |---|---|---|---|
  | MP3D VALSEL（10 ep × 150） | 0.364 / 0.553 | 0.334 / 0.530 | 0.347 / 0.519 |
  | Macarons++ Dev-3 + Test-3（6 scenes，30 episodes × 100 steps） | 0.711 / 0.880 | 0.730 / 0.906 | 0.719 / 0.880 |

  结论 Conditional Go：室内有效，室外微调不如冻结模型。
