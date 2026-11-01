# EXP6 V4 / V3.1 第一阶段冻结协议 v1.0（Outdoor-only 预训练 + 双域 G1 / G2 / G3）

> 草案 2026-10-31 20:10；**已于 2026-10-31 20:46:36 冻结（§9）**（冻结时间与配置 / 代码 sha256 写在文末 §9，之后的任何改动只能记入 `reports/V4_DEVIATIONS.md` / `exp6_v3.1/reports/V3_1_DEVIATIONS.md`）。依据：任务书 `exp6_v3/EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md` §6–§8 与 `AUDIT_BASELINES.md`。

## 1. 数据（两组相同）
| 项 | 定义 |
|---|---|
| 训练集 | Macarons++ **Train-6 去 S-c**：138 条轨迹（`task1/frozen_values.json` 的 `train_trajs`），E2 **D100** 视角分片 `work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100`（98,887 个候选视角、13,800 个状态）；V3.1 另读同一分片对应的 4× TSDF 缓存 `work_dirs/exp6_v3/shared_cache/tsdf4/outdoor`。**MP3D 训练样本 = 0**（manifest 断言 `domains = ["outdoor"]`、`mp3d_training_samples = 0`） |
| 监控集 | S-c：Train-6 六个场景的 42 条保留轨迹（31,027 视角）；只记录 FM loss / 采样指标 / 重建指标，**不用于选择** |
| 禁止 | Dev-3（barts, pantheon, colosseum）、Test-3（alhambra, bridge, neuschwanstein）、任何 MP3D 房屋在训练与选择中出现（`tests/test_split_and_leakage.py`） |
| 为什么用 D100 | B2 / E2-D100 / V3-T0 / T1 的输入 P1 都是 D100 + 稀疏评分；V4 与 V3.1 用同一密度才能和它们在同一规划口径（`T3_P1_MODE=b2`）下比较。V2 Original 为稀疏输入，表中单独标注 |
| 采样 | V2 / E2 / V3 规则不变：状态权重 t ≤ 20 × 2；每状态 ≤ 8 个候选；陈旧度 Δ ∈ [1, 9]（p 0.5）；RNG `[seed, rank]`，resume 时 `[seed, rank, step]`（V2 / V3 的既定规则） |
| 标签 | `lmask = UNKNOWN ∧ b ≠ 0`；x1 = (b, y)；训练不读 `tstar` |

## 2. 模型
- **V4** = `seen2scene.futureview.model.V2Model(model_v4.json)`：V2 的 3D U-Net（F0 32 @8×、F1 64 @16×）、原 32 点射线采样 + Conv1d 射线编码器、原 2D Rectified-Flow U-Net、原损失 / 采样器 / 解码；**唯一新增** `RayXAttn`（`ray_xattn.enabled = true`）：每条射线的 query = [V2 射线上下文 64 + Fourier(方向) 24 + Fourier(相机位置 / 网格范围) 24] → W_q；token = 32 个采样点的 [F0, F1, 域内标志 (97) + s_j + Fourier(2 s_j − 1) + t_j / ext_max (10)] → LayerNorm → W_k / W_v；4 头 × 16 维；mask = 采样点在网格内（UNKNOWN 点是合法 token；无网格内点的射线输出 0）；W_o（64 → 64）normal(std 0.005)、零偏置；输出 **residual 加到 V2 射线上下文**，条件图仍为 76 通道，Flow 接口不变。参数 25,430（总 24,847,544）。无 [R, R] 注意力。关闭开关时与 V2 逐位相同（测试 `test_g0_11a_v2_bitwise_replay_with_flag_off`）。
- **V3.1** = `seen2scene.v3.model.V3Model(model_v3_1.json)` = V3-T0 的有效配置（`xattn.enabled = false`）：Simple 3D VAE（[32, 64] → μ / logvar 16）、原 LatentRayEncoder、原 2D Flow；Flow 条件 = μ；训练期 decoder 重建 + KL。参数 23,539,622（与 T0 相同；seed 0 的初始公共模块 sha 与 T0 相同）。

## 3. 训练（两组同预算、同全局 batch）
| 项 | V4（V2 Original 配方） | V3.1（V3 配方） |
|---|---|---|
| 预算 | **40,000 步**（任务书建议的共同第一阶段预算；V2 Original 的 60k 是历史参照） | 40,000 步 |
| 全局 batch | 64 视角 / 8 状态 / 步（4 rank × 16 视角 / 2 状态） | 64 视角 / 8 状态 / 步（4 rank × 2 个 micro-batch × 8 视角 / 1 状态；`accum = 2` 只为显存，L_FM 按**整步全局有效射线数**归一化，L_rec / KL 按 world × accum，梯度 = 64 视角批的梯度；训练器核对数据端与前向端射线计数一致，否则 STOP） |
| 优化 | AdamW(0.9, 0.99) wd 0.01，lr **2e-4**，预热 1k，余弦到 0，grad clip 1.0，bf16，EMA 0.999，seed 0 | AdamW(0.9, 0.99) wd 0.01，lr **1e-4**，预热 1k，余弦到 0，grad clip 1.0，bf16，EMA 0.999，seed 0 |
| 损失 | 原 V2 velocity MSE（`v2_data.fm_sum`，分母 2 × 全局有效射线数） | 原 V3：L_FM + 1.0 L_recon + β KL（β 0 → 0.02，5k 步） |
| 初始化 | 随机（scratch），不加载任何权重 | 随机（scratch） |
| 评估 / 保存 | FM loss 每 2,500 步（train_sub 1,024 + S-c 2,048 视角），采样指标每 5,000 步；EMA 保存于 {0, 1k, 5k, 10k, 15k, 20k, 25k, 30k, 35k, 40k}；`ckpt_last.pt`（model / EMA / optimizer / step / RNG）每 1,000 步 | 每 5,000 步（含第 0 步）S-c 2,048 + out_train 1,024：FM、采样、重建；EMA 同上；`ckpt_last.pt` 每 1,000 步 |
| **checkpoint 规则（冻结）** | **固定 EMA@40,000 → `checkpoints/final.pt`**；不看 Dev-3 / Test-3 / MP3D 的任何指标；S-c 指标只作监控 | 同左 |
| 停止 | 损失非有限 → STOP 并报告；无早停 | 同左 |
| 延长 | 40k 未收敛也以 40k 为主结果；延长到 60k 须另立阶段、事先声明、两组同预算 | 同左 |
| 资源 | gpu115（作业 3435033）4 × L40S | gpu117（作业 3429774）4 × L40S；若分配到期则从 `ckpt_last.pt` resume（记入偏差） |
| 记录 | run manifest：git SHA / dirty、配置 sha256、数据文件 sha256 与 manifest sha、参数量、host / GPU / 作业、每 rank batch、每步 s、显存、有效射线数（`rays` / `rays_outdoor`） | 同左 + KL / 重建 / 后验统计 |
| 不同之处 | 两组继承各自历史配方（lr 2e-4 vs 1e-4、损失不同）→ 跨架构比较是**匹配预算的性能对照**，不是严格同损失对照 | |

## 4. G0（正式训练前；结果 `report/G0_VALIDATION.md`）
G0-1 基线审计（`AUDIT_BASELINES.md`、`CHECKPOINT_LINEAGE.md`）；G0-2 划分；G0-3 DDP / 累积归一化；G0-4 V4 形状；G0-5 X-Attn 梯度（单测 + tiny overfit 第 10 步梯度检查）；G0-6 相机条件；G0-7 可见性 mask；G0-8 V3.1 VAE；G0-9 零泄漏；G0-10 checkpoint / resume；G0-11 基线回放（V2 开关关闭逐位；冻结规划器 2 步闭环与存档 episode 逐位）；G0-12 吞吐（4 卡 1,000 步冒烟：s/步、显存、无 OOM）。全部关键项 PASS 后冻结并启动。

## 5. 评估（权重冻结后；两组相同；GT 只在评估侧）
| 项 | 定义 |
|---|---|
| G3 室外 | `task2/scripts/t24_out_sim.py` 冻结协议：Dev-3 + Test-3 6 场景 × 5 起点 × 100 步；planner child = V3 路径（V3.1：`exp6_v3/shared/v3_out_planner.py` + `V3_CKPT`）/ V2 路径（V4：`t24_v2_out_planner.py` + `p1_full.patch("b2")`，checkpoint 覆盖）；`T3_P1_MODE=b2`；DistP1、K = 4、噪声种子、候选、动作约束不变；单调覆盖率 `t25_out_cov.py`；指标 AUC@100 / Final@100 / Cov@20 |
| G3 室内 zero-shot | `t23_in_sim.py` 冻结协议：5 楼层 × 2 起点 × 150 步，n_win = 5，R10 刷新，B2 口径；指标 AUC@150 / Final@150 / Cov@20 / 停滞 episode（t ∈ [30, 120] 内 ≥ 30 步覆盖增长 < 0.01）；**不更新任何权重** |
| G1 / G2 | `exp6_v3/shared/eval_fixed.py` 的固定状态（MP3D VALSEL 90、Dev-3 275；B2 回放逐位核对）+ Experiment A 的评估池；G1 = 未知射线深度 MAE / HIT Brier / ES / K = 4 离散度；G2 = 被选动作的真实一步覆盖增益 pp 与 Top-1（regret 只在附录） |
| 统计 | 场景 / 楼层均值 → 场景级 bootstrap 95 % CI（10k）；配对差按 (场景, 起点)；胜 / 负场景数；Task 3 §4 "有意义 / 退化"判定（ΔAUC ≥ 0.010 且 ΔFinal ≥ 0.010 且胜 ≥ 4 且无场景 < −0.030；退化 = ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010） |
| 对照组 | MAG-G、MAG-B、V2 Original（Task 2 V2O，稀疏口径，单独标注）、B0、B2、E2-D100、V3-T0、V3-T1（历史，不重跑）+ V3.1、V4；表中标注每组的训练来源（Outdoor-only scratch / Mixed FT / Mixed scratch）与规划口径 |
| 禁止 | 用 Test-3 / MP3D G3 挑 checkpoint；beam、frontier、S1 / S2 / S3、n_win ≠ 5（主表）、2D Flow 改造、位姿预测器 |

## 6. 第二阶段（只准备，不启动）
`configs/indoor_finetune_DRAFT.json`（V4-FT / V3.1-FT 从各自 Outdoor-only EMA@40k 初始化，MP3D train 228 条，5k 步，lr 2e-5 预热 500，评估双域 G3）与 `configs/nwin_ablation_DRAFT.json`（冻结模型、规划器 only，n_win 7 / 9，先固定状态后闭环）在第一阶段报告后由用户决定。

## 7. U0
V2 Original 不是 V4 的严格对照（稀疏输入、60k、无开关）。X-Attn 的因果结论需要 U0 = `model_v4.json` 去掉 `ray_xattn`（`enabled = false`）、同数据 / 预算 / 种子 / 配方；本轮不训练 U0，结论中标注"待 U0"。

## 8. 偏差与止损
- 训练中不改架构、种子、损失、数据、评估协议；bug 需同时记录、两组一致处理。
- 损失非有限 / OOM / 分配到期 → resume（偏差记录）；若两次 resume 后仍不能完成，报告为未完成，不以中间步数冒充 40k。
- 未经用户确认不启动 MP3D 微调、U0、n_win 闭环、Flow 改造。

## 9. 冻结记录
（冻结 2026-10-31 20:46:36，G0 全部通过后；代码 git SHA `f57be9ac8ec89d36b9e2da96bd85e8dfafe83770`（提交 `[V4][setup]`，作用域内无未提交改动：True）。配置 / 代码 sha256 记录：`configs/FROZEN_SHA256.json`（V4，15 个文件）与 `../exp6_v3.1/configs/FROZEN_SHA256.json`（V3.1，14 个文件）。关键哈希：`model_v4.json` `d2537315a6c628e7…`，`outdoor_train.json`（V4）`a3d0bd22edd4fb8c…`，`model_v3_1.json` `680e872a25199c7f…`，`outdoor_train.json`（V3.1）`ff6ea9145df779a0…`，`seen2scene/futureview/model.py` `9a30386b167e1c21…`。自此之后的改动只能记入 `reports/V4_DEVIATIONS.md`。）
