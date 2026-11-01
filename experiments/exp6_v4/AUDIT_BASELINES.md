# EXP6 V4 / V3.1 — Phase 0 审计：历史基线、数据划分与评估协议（2026-10-31）

> 本文件只记录从源码、配置、run manifest 和 checkpoint 实测得到的事实；每一条都给出出处。未核实的内容标为"未核实"。审计对象：`plan.md`（研究路线，与本轮无冲突条目）、V2 Original、V2-Mix（B0 / B2）、E2-D100、EXP6 V3 T0 / T1、Macarons++ 与 MP3D 划分、射线 / 特征形状、P1 输入与评分、G3 协议。checkpoint 的完整 sha256 与继承关系见 `CHECKPOINT_LINEAGE.md`；本轮冻结的训练 / 评估协议见 `FROZEN_PROTOCOL.md`。

## 1. 历史模型的训练来源（G0-1）

| 模型 | 初始化 | 训练数据（域） | 步数 / 全局 batch | 优化 | 选择规则 → 最终权重 | 来源文件 |
|---|---|---|---|---|---|---|
| **V2 Original**（EMA@60k） | 随机（seed 0） | **Outdoor-only**：Macarons++ Train-6 去 S-c，**稀疏 10 % P1**（T1.1 分片 `work_dirs/exp5_future_view/task1/data/views`）：138 条轨迹、98,887 个候选视角、13,800 个状态；S-c 42 条（31,027 视角）只做开发指标 | 60,000 步；每步 64 视角 / 8 状态（4 rank × 16 / 2，每 rank 持有全部轨迹） | AdamW(0.9, 0.99) wd 0.01，lr 2e-4，预热 1k，余弦到 0；bf16；grad clip 1；EMA 0.999；stale p 0.5 / max 9 | 协议固定 `final = EMA@60000`（`train_main.json: final_checkpoint`），无挑选 | `task1/scripts/t12_train.py`、`task1/configs/{train_main,model_v1,data_v1}.json`、`work_dirs/exp5_future_view/task1/training/main/run_none.json`（gpu115 4 × L40S，2026-10-22 21:54 → 10-23 02:35，0.25 s/步，14.4 GB/卡） |
| **V2-Mix**（EMA@17.5k；B0 与 B2 共用） | **V2 Original EMA@60k**（`t22_train_mix.py` 第 88–91 行按 sha256 断言后 `load_state_dict(ck0["ema"])`；优化器重新初始化） | **Mixed 微调**：MP3D train 房屋 114 个楼层 228 条轨迹（t21 分片，n_win = 5）+ Macarons++ Train-6 去 S-c（T1.1 稀疏分片），每步 32 + 32 视角；每 rank 持有 1/4 轨迹（`[rank::world]`） | 20,000 步；64 视角 / 8 状态 | lr 2e-5，预热 500，余弦；其余同 V2 | `score = ES_in/ES_in(0) + ES_out/ES_out(0)` 最小 → **step 17,500**（`training/mix_v1/selection.json`；评估步 0, 2.5k … 20k） | `task2/scripts/t22_train_mix.py`、`task2/configs/train_mix_v1.json`、`training/mix_v1/run_none.json`（gpu114，2026-10-27 15:50 → 17:12，7.5k 处 resume） |
| **E2-D100**（EMA@5k） | **V2-Mix EMA@17.5k**（`e2_d100_v1.json: init_checkpoint` + sha256 断言） | **Mixed 微调**：同 V2-Mix 的轨迹列表，但分片用 **D100 P1**（`e2/w2_d100/mp3d/views` + `e2/out_views_d100`）重建；采样索引与 S10 臂相同 | 5,000 步；64 视角 / 8 状态 | lr 2e-5，预热 500，余弦 | 同 V2-Mix 规则 → step 5,000（评估步 0–5k 每 1k） | `task3_full_depth_p1/e2_pilot/scripts/e2_train.py`（= 冻结的 t22 训练器 + 路径重定向）、`e2/training/d100/run_none.json`（gpu117，2026-10-29 03:38 → 04:32） |
| **V3-T0**（final = EMA@40k） | 随机（seed 0），公共模块 sha `6a9d6071a225…` | **Mixed from scratch**：MP3D train 228 条 + Macarons++ Train-6 去 S-c 138 条，D100 分片 + 4× TSDF 缓存；每步 32 + 32 视角来自 1 + 1 状态 / rank；每 rank 持 1/4 轨迹 | 40,000 步；64 视角 / 8 状态 | AdamW lr 1e-4，预热 1k，余弦；λ_rec 1.0；β_KL 0 → 0.02（5k 步）；EMA 0.999；bf16 | `ES_in/ES_in(5k) + ES_out/ES_out(5k)` 最小（VALSEL + S-c）→ **40,000**（分数 1.3803） | `seen2scene/v3/trainer.py`、`exp6_v3/configs/{model_v3,train_v3}.json`、`work_dirs/exp6_v3/task0_vae/runs/main/run_none.json`（gpu117，10-30 06:25 → 22:18，63.7 GPU-h，1.34 s/步，峰值 33 GB/卡） |
| **V3-T1**（final = EMA@40k） | 同 T0（公共模块同 sha），Cross-Attn 独立初始化、W_o 零 | 同 T0 | 同 T0 | 同 T0 | 同规则 → **40,000**（1.3946） | `work_dirs/exp6_v3/task1_xattn/runs/main/run_none.json`（gpu115 作业 3435033，10-30 11:18 → 10-31 03:26，65 GPU-h） |

结论：
- V2 Original 的 60k 步**确实是 Outdoor-only、从零初始化**，但输入 P1 是**稀疏 10 %**（B0 口径），预算 60k，且没有 X-Attn 开关；它不是 V4 的严格 U0 对照（数据密度、预算不匹配），只能作历史参照。X-Attn 的因果结论**尚待 U0**（V2 架构 + D100 + 40k + Outdoor-only），本轮不启动（任务书 1.2）。
- B0 与 B2 **共用 V2-Mix EMA@17.5k**：B0 = 稀疏 P1 输入 + 稀疏 DistP1；B2 = D100 P1 输入 + 稀疏 band_dil 评分（`task3_full_depth_p1/scripts/p1_full.py: patch(mode="b2")`，由 `T3_P1_MODE=b2` 触发；`exp6_v3/shared/sim_{in,out}_v3.py` 对 V3 组也设置该变量）。
- E2-D100 是 **V2 Original → V2-Mix → E2** 两级 Mixed 微调的产物，评估用 B2 口径；**不是** Outdoor-only。
- V3 T0 / T1 是 **Mixed from scratch**（同种子、同数据流）；Flow 条件 = VAE 的 `mu`（`seen2scene/v3/model.py: view_cond` 用 `h["mu"]`），重参数化 z 只进训练期 decoder。

## 2. V3 结构事实（V3.1 复制的对象）
| 项 | 实测 |
|---|---|
| 输入 | 4× 聚合 Partial TSDF `[sdf/TRUNC, band 比例, free 比例]` + `valid`（`seen2scene/v3/partial_tsdf.py`；细体素 VS = 0.011，填充到 8 的倍数） |
| VAE | `VAEEncoder(in 4, ch [32, 64], latent 16)`：stem 32 @4× → Res3D → stride-2 64 @8× → 2 × Res3D → 1×1 → μ / logvar（16 通道，每格 8 个细体素）；`VAEDecoder` 只在训练期重建 `[sdf_hat, 3 类 logits]` |
| 射线条件 | `LatentRayEncoder`（V2 的 Conv1d 栈，32 个采样点，每点 `[μ (16), valid8 · inside]`）→ 64 通道；+ 11 通道确定性投影 + 1 有效列 = **76 通道** |
| Flow | `FlowUNet` = V2 `UNet2D`（64 / 128 / 192 / 256，瓶颈自注意力）；T1 在瓶颈自注意力后加 `CrossAttnBlock`（Q = 瓶颈 token + Fourier PE，K/V = 有观测支持的潜变量格 ≤ 512 / 视角，W_o 零初始化） |
| 损失 | `L_FM`（全局有效射线均值）+ 1.0 × `L_recon`（类别平衡 CE + BAND 上 sdf L1）+ β(step) × KL（每有效潜变量单元） |
| 参数 | T0 23,539,622（推理 22,954,434；decoder 585,188 仅训练）；T1 +191,364 |
| 推理 | K = 4、10 步 Euler、确定性解码、原 DistP1、Greedy-NV（Task 2 规划器不变） |

## 3. V2 结构事实（V4 的主干）
| 项 | 实测（`seen2scene/futureview/model.py`、`rays.py`、`geometry.py`、`task1/configs/model_v1.json`） |
|---|---|
| 传感器 / 射线 | MAGICIAN 456 × 256、fov 60°；评分网格 64 × 114 = **7,296 条射线**（块 4），训练时亚像素抖动 [−2, 2) |
| 编码器输入 | 8× 粗网格 `[EMPTY, BAND, 域内比例]` + z 通道（4 通道）；`Encoder3D(ch [32, 64, 96])` → **F0 [1, 32, X', Y', 32]（8×）、F1 [1, 64, X'/2, Y'/2, 16]（16×）** |
| 射线编码器 | 每条射线 **N = 32** 个点 t_j = (j + 0.5)/32 · t_cap，三线性采样 F0 / F1 + 域内标志 → `[R, 97, 32]` → Conv1d ×3 + fc → 64 通道 |
| 条件图 | 11 投影通道 + 64 射线上下文 + 1 有效列 = 76 通道，`[V, 76, 64, 128]`（114 列零填充到 128） |
| Flow | `UNet2D(64/128/192/256, blocks 2, 第 3 层与瓶颈自注意力, 4 头)`，x = (b, y) 2 通道；训练 velocity MSE 只在 `M_unknown ∧ b ≠ 0` 射线上，分母 = 2 × 全局有效射线数 |
| 推理 | 10 步 Euler；K = 4 共享噪声；解码规则 `b > 0 ∧ y < 1`；bf16 自动混合，投影 / 解码 / DistP1 float32 |
| 参数 | encoder 2,412,000 / rays 76,224 / unet 22,333,890 = 24,822,114 |

## 4. 数据划分（G0-2，测试 `tests/test_split_and_leakage.py` 通过）
| 集合 | 场景 / 房屋 | 来源 |
|---|---|---|
| Macarons++ **Train-6** | bannerman, dunnottar, fushimi, liberty, pisa, redeemer；每场景 30 条轨迹，其中 S-c 保留轨迹 8/6/8/6/6/8 = 42 条（`heldout = True`），训练 138 条 | `task1/frozen_values.json: split`（规则 protocol v1.0 §2.5，seed key `exp5-V2|holdout|{scene}`）；分片标志 `heldout` 实测一致 |
| **Dev-3** | barts, pantheon, colosseum | `task2/configs/eval_outdoor_v1.json` |
| **Test-3** | alhambra, bridge, neuschwanstein | 同上 |
| 交集 | Train-6 ∩ (Dev-3 ∪ Test-3) = ∅；D100 分片根目录下无 MP3D 文件 | 测试断言 |
| MP3D train | 官方 train 61 房屋（其中 114 个楼层、228 条轨迹用于 V2-Mix / E2 / V3） | `task2/scripts/t2_common.py: TRAIN_HOUSES`，`work_dirs/.../task2/mp3d/manifests/traj_jobs.json` |
| MP3D VALSEL | 官方 val 11 房屋中去掉评估房屋的 6 个（9 条轨迹） | `t2_common.VALSEL_HOUSES` |
| MP3D 评估（zero-shot G3） | 5 楼层 × 2 起点：Z6MFQCViBuw L0 {6, 17}、oLBMNvg9in8 L1 {36, 78}、QUCTc6BB5sX L1 {34, 135}、2azQ1b91cZZ L0 {60, 179}、TbHJrupSAjP L1 {112, 114} | `t2_common.EVAL_LEVELS`；`purpose_of()` 的 GT 守卫；MP3D test 房屋 sealed |
| 非盲说明 | Test-3 与这 5 个楼层已被 Task 2 / Task 3 / E2 / V3 / Experiment C / V3-Task 2 反复使用，**不能视为全新盲测**；本轮全部闭环结论按"非盲、冻结协议下的对照"报告 |

## 5. 训练样本与泄漏（G0-9）
- 分片每行 = 一个 (轨迹, t, 候选 a) 视角：`R, T, jitter`（候选位姿）、`cls, nseg, tk, dlo, dhi, lu`（候选射线对**历史 P1（状态 t）**的投影统计）、`b, y, cat`（**GT 未来深度 t\* 的标签**，仅作监督）、`coarse[t]`（状态 t 的 P1 粗网格，由记录 0..t 构建）、`tstar`（仅评估，训练键集合 `VIEW_KEYS` 不含）。
- 采样器：编码器输入 = `coarse[t − Δ]`，Δ ∈ [0, 9]（p 0.5），投影通道 / 标签属于 t → 输入永远不晚于 t；测试断言 `coarse` 的已观测体素数随 t 单调不减（无未来信息的必要条件）。
- 规划器侧：`T3_P1_MODE=b2` 的 D100 融合只用已执行观测；候选评分读预测深度；GT 只在模拟器侧（覆盖率、合法性）；LeakGuard 保留。

## 6. P1 输入 vs 评分（G0-1 补充）
| 组 | 模型输入 P1 | DistP1 评分地图 | 代码 |
|---|---|---|---|
| V2 Original（Task 2 V2O）、B0 | 稀疏 10 % | 稀疏 band_dil | `t23_v2_in_planner.py` / `t24_v2_out_planner.py`（`T3_P1_MODE` 未设 = sparse） |
| B2、E2-D100、V3-T0 / T1、**V3.1、V4** | **D100**（全部有效像素） | 稀疏 band_dil（训练时的评分口径） | `p1_full.patch("b2")` |
| DistP1 | 唯一新体素 ∉ 膨胀稀疏 BAND，权重 min(1, (z/z0)²)，K = 4 均值；解码只在 UNKNOWN 段、稠密 t_known 之前放置命中 | `seen2scene/futureview/scoring.py` |

## 7. G3 / MAGICIAN 对齐（G0-11）
- MP3D：`task2/scripts/t23_in_sim.py`（T0.12 网格 1.0 m、5 仰角 × 8 方位、官方邻居与单段 GT 合法性、未访问优先、C_obs 覆盖率）；10 条 episode × 150 步；MAG-G / MAG-B 在同一动作协议下的闭环来自 Task 2（不重跑）。
- Macarons++：`task2/scripts/t24_out_sim.py`（冻结 M 协议 EpisodeSim、官方设置网格、GT 段测试、hover）；6 场景 × 5 起点 × 100 步；单调覆盖率 `t25_out_cov.py`。
- V2 Original 的闭环数字来自 Task 2 的 V2O 组（**同一 episode 集合、同一模拟器**；校验：Task 2 的 V2M 组按同样读法重算 = `reports/g3.json` 的 B0（AUC 0.729 / Final 0.884 / Cov@20 0.622，逐位），`g3.json: B0_equals_task2 = {indoor 10, outdoor 30}`）：
  - MP3D：AUC@150 **0.315**、Final **0.441**、Cov@20 **0.179**（逐楼层 AUC/Final/Cov@20：2azQ 0.183/0.267/0.086、QUCT 0.179/0.261/0.086、TbHJ 0.368/0.449/0.266、Z6MF 0.207/0.363/0.090、oLBM 0.636/0.865/0.366）
  - Macarons++：AUC@100 **0.724**、Final **0.898**、Cov@20 **0.578**（alhambra 0.648/0.879/0.479、barts 0.835/0.999/0.720、bridge 0.806/0.989/0.637、colosseum 0.740/0.865/0.645、neuschwanstein 0.691/0.874/0.530、pantheon 0.625/0.785/0.454）
  - 规划口径：稀疏 P1 输入 + 稀疏评分（B0 口径），n_win = 5，K = 4 —— 与 V4 / V3.1 的 B2 口径不同，表中单独标注。
- 其余历史值（`exp6_v3/reports/g3.json` 复核，与任务书 §8.2 一致）：MP3D MAG-G 0.278/0.377/0.169、MAG-B 0.292/0.377/0.165、B0 0.322/0.469/0.195、B2 0.334/0.530/0.197、E2-D100 0.364/0.553/0.195、T0 0.311/0.463/0.183、T1 0.331/0.508/0.184；Macarons++ MAG-G 0.704/0.905/0.507、MAG-B 0.685/0.891/0.504、B0 0.729/0.884/0.622、B2 0.730/0.906/0.572、E2-D100 0.711/0.880/0.570、T0 0.702/0.872/0.568、T1 0.712/0.883/0.566。

## 8. 资源（审计时刻 2026-10-31 20:00 `squeue`）
| 作业 | 节点 | GPU | 剩余 | 状态 |
|---|---|---|---|---|
| 3429774 `vscode-dev` | gpu117 | 4 × L40S 46 GB | 19 h 49 min（≈ 11-01 15:49） | RUNNING，4 卡空闲 |
| 3435033 `exp6-v3-aux` | gpu115 | 4 × L40S | 15 h 09 min（≈ 11-01 11:09） | RUNNING，4 卡空闲 |
| 3435072 `exp6-v3-aux2` | — | 4 × L40S，24 h | — | PENDING（QOSMaxGRESPerUser：两个运行作业之一结束后才会启动） |

## 9. `plan.md`
`plan.md` 是 2026-10 之前的总路线（EXP 编号对照、VHD、一致性、GeoControlNet），没有关于 V4 / V3.1 / Outdoor-only 的条目；本轮任务书 `exp6_v3/EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md` 是唯一的需求来源，与 `plan.md` 无冲突。
