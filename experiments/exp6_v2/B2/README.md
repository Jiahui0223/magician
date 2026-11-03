# B2 —— 冻结 V2-Mix，D100 输入 / S10 DistP1 评分

**B2 没有自己的训练**：权重就是 EXP5 Task 2 按冻结规则选出的 V2-Mix EMA@17.5k。B2 与 B0 用的是同一份权重，区别只在规划器的 P1 模式：B2 的模型输入用稠密 D100 地图，DistP1 的新颖性仍用稀疏 S10 的 band_dil。
配置：`configs/b2.json`（本文件的机器可读版本）。

## 1. 架构（共享 V2 代码，未复制）
3D U-Net 编码器（`Encoder3D` 32/64/96）→ 32 点射线编码器 + Conv1D（`RayEncoder`）→ 2D rectified-flow U-Net（`UNet2D`）生成 K = 4 个未来深度样本（10 步）→ DistP1 打分 → 选点。
- **参数**：24,822,114。
- **模型代码**：`seen2scene/futureview/model.py`（V2Model）、`predictor.py`（V2Predictor）、`rays.py`、`geometry.py`、`scoring.py`。
- **规划核心**：`experiments/exp5_future_view/task1/scripts/v2_plan_core.py`。
- **模型配置**：`experiments/exp5_future_view/task1/configs/model_v1.json`。已核对它与 checkpoint 内嵌的 model_cfg 完全一致。

## 2. 规划设置（必须保持，整理中未改）
| 项 | 值 | 实现 |
|---|---|---|
| P1 模式 | `b2`（环境变量 `T3_P1_MODE=b2`） | `task3_full_depth_p1/scripts/p1_full.py`：模型输入换成 FullDepth-P1（D100），只把 `caster.band_dil` 换回稀疏 S10 |
| Flow 采样 | K = 4，10 步，噪声按状态键 sha 播种 | `predictor.V2Predictor`、`v2_plan_core.score_state` |
| 评分 | DistP1（`new_set_from_depth`），z0 = 场景尺度 / 2 | `task1/configs/scoring_v1.json` |
| 室内窗口 / 刷新 | n_win = 5，R10 | `task2/configs/eval_indoor_v1.json`、`eval_outdoor_v1.json`（组 `V2M`） |
| 模拟器 | MP3D 150 步，Macarons++ 100 步 | `task3 t33_in_sim.py / t33_out_sim.py` → `task2 t23_in_sim.py / t24_out_sim.py`，规划子进程为 `t32_*_planner.py` |

## 3. 训练来源（以代码 / 配置 / manifest 为准）
| 阶段 | 初始化 | 数据 | 步数 | 选择 | 脚本 / 配置 | 权重（新路径） |
|---|---|---|---|---|---|---|
| V2 Original（EXP5 Task 1） | 随机 seed 0 | Macarons++ Train-6 去 S-c，稀疏 P1（T1.1 分片） | 60k | EMA@60k（固定） | `task1/scripts/t12_train.py` + `task1/configs/train_main.json` | `lineage/v2orig_ema60000.pt` |
| V2-Mix（EXP5 Task 2） | V2 Original EMA@60k | MP3D train（t21 分片，n_win 5）+ Train-6 去 S-c（稀疏），各 32 views | 20k | ES 比值最小 → **EMA@17.5k** | `task2/scripts/t22_train_mix.py` + `task2/configs/train_mix_v1.json`；`t22_select.py` | **`v2mix_ema17500.pt`** |

张量级核对（`results/smoke_*.json` 的 `lineage`）：V2-Mix 第 0 步权重 = V2 Original EMA@60k。

## 4. Checkpoint（`work_dirs/checkpoints/exp6_v2/B2/`，全部是指向原文件的相对软链接）
| 新路径 | 原路径 | step | sha256 | 用途 |
|---|---|---|---|---|
| `v2mix_ema17500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt`（= `task2/training/mix_v1/ema_017500.pt` 硬链接） | 17500 | 343f41ab8d566c52… | **B2 模型**（= Task 2 组 V2M；B0 / B2 规划器） |
| `v2mix_ema20000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-2_V2Mix_last/ema_020000.pt` | 20000 | 5c68720ea6c64f1a… | V2-Mix 末 EMA（未选） |
| `v2mix_raw20000.pt` | `work_dirs/exp5_future_view/task2/training/mix_v1/ckpt_last.pt` | 20000 | 9805f081da3a… | V2-Mix 原始训练状态（模型 + EMA + AdamW + RNG） |
| `intermediate/v2mix_ema{000000…015000}.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_*.pt` | 0–15000 | 见 MANIFEST | V2-Mix 每个评估点的 EMA（step 0 = V2 Original 权重） |
| `lineage/v2orig_ema60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt` | 60000 | a379c0fb9d7d8026… | V2 Original（V2-Mix 的初始化） |
| `lineage/v2orig_raw60000.pt` | 同目录 `ckpt_last.pt` | 60000 | 7e827ecc2ce5… | V2 Original 原始训练状态 |

所有链接均满足：链接与原文件是同一文件（samefile），SHA256 与 Task 2 manifest 一致，严格加载（strict=True），参数 24,822,114。明细见 `work_dirs/checkpoints/exp6_v2/MANIFEST.json`。

## 5. 入口（`scripts/`）
| 脚本 | 作用 |
|---|---|
| `verify.sh` | 加载 `configs/b2.json`，核对 checkpoint（链接 / SHA / step / model_cfg / 严格加载 / 参数），CPU 上运行 → `results/verify_config_checkpoint.json` |
| `smoke.sh [dev3\|valsel] [GPU]` | 最小推理 smoke：一个固定状态 episode，分别从旧路径和新路径加载，比较未来深度、DistP1、排序，并与历史行比较 → `results/smoke_<set>.json` |
| `g12_fixed.sh` | 历史 G1 / G2 命令（`e2_fixed.py` 同时计算 B0 / B2 / E2-S10 / E2-D100，再跑 `e2_analyze.py`）；只补写缺失文件，历史结果存在时不做任何事 |
| `g3.sh [OUT] [N]` | G3 闭环：MP3D 10 × 150、Macarons++ 30 × 100（`configs/g3_jobs_{in,out}.txt` = 历史 `jobs_b2_*.txt`），worker 为 `../shared/g3_queue.sh`；默认输出 `work_dirs/exp6_v2/g3`，不会覆盖历史结果 |
| `train_lineage.sh` | 打印两阶段训练的原命令；只有 `RUN=1` 时才执行，输出到新目录。本次未运行 |

## 6. 历史结果（`results/historical/` 是指向原报告的软链接）
- **G3**（Task 3 E1 B2 闭环，`T3_E1_B2_CLOSED_LOOP.md`）：

  | 数据集 | AUC | Final | 相对 B0 |
  |---|---|---|---|
  | MP3D | 0.334 | 0.530 | ΔFinal +0.061，CI 不含 0 |
  | Macarons++ | 0.730 | 0.906 | ΔAUC +0.001 |

- **G2**（固定状态，E2 pilot 的行，被 EXP6 各版本引用）：
  - Dev-3 275 态：Top-1 0.581，增益 1.275 pp；
  - MP3D VALSEL 90 态：Top-1 0.533，增益 0.882 pp。
- **Task 2 V2-Mix 总报告**：`TASK2_FINAL_REPORT.md`。

## 7. 注意
- `task1/configs/closed_loop_v1.json` 中 `baselines` 的 "B2" 行，以及该文件的 "E2" 臂（E2_V2G_Nat_K4），是 EXP5 Task 1 / EXP4 的旧命名，**与本目录的 B2 / E2 无关**。
- 历史启动脚本（`work_dirs/exp5_future_view/task3_full_depth_p1/{run_queue2.sh, chain_b2.sh, jobs_b2_*.txt}`）在 work_dirs 中，不在版本库里。本目录的 `scripts/g3.sh` + `../shared/g3_queue.sh` 和 `configs/g3_jobs_*.txt` 是它们的版本化形式（参数相同）。
