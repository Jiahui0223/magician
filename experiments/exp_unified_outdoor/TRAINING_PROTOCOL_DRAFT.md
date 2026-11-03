# 统一 Outdoor-only 训练基准 —— 训练协议草案（只准备、未训练）

> 2026-11-03 起草。状态：**草案，等待用户确认**。本目录只新建文件；未修改 / 删除任何历史文件；未启动训练、未使用 GPU。
> 所有"事实"均给出 `文件:行`；标 **【未验证】** 的是推断或外推，需要 GPU 冒烟或真实运行确认。
> 观测密度（S10 / D100）由另一实验决定，所以两套配置都已准备。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| V2 Original（task1 main）能否直接当 V2-S10？ | **可以复用（10k / 20k / 40k / 60k）**：数据、划分、采样器、batch、LR 计划、步数、种子、精度都相同，且该运行从未 resume（§4）。**缺 EMA@48k**（只有 45k / 50k），也没有保留中间的完整训练状态。只有在需要 V2 的 48k 匹配点时才要重跑（0→60k 约 4.7 h，4 卡） |
| 剩余配方差异 | 只剩架构本身要求的例外：V3 的 VAE 重建 / KL 损失与 4× TSDF 输入、V3 的 accum 2（显存）、V5 的 L_surf / L_bridge / Bridge 阶段 / 参数组 LR、V4C / V5 沿用历史续训 LR。另有一个待用户决定的项：**V3 的 LR 用 1e-4（历史值）还是 2e-4（统一值）**——没有证据表明 2e-4 不稳定，也从未测试过 2e-4（§3 D1） |
| Resume | 四个历史训练器都保存 model / EMA / optimizer / step；LR、KL β、V5 阶段与 λ 都是 step 的纯函数，不需要保存 scheduler 状态。缺的是**各 rank 的 RNG**：DDP resume 时会按 (seed, rank, step0) 重新播种，结果可复现，但和不中断的运行不一样。新文件 `scripts/uo_common.py` 的 `ExactResume` 在每个 checkpoint 步保存各 rank 的 RNG 并在 resume 时精确恢复；CPU 单元测试通过，**DDP 实机测试未做【未验证】** |
| 60k 墙钟估计（4 × L40S） | V2 ≈ 4.7–4.8 h；V4 ≈ 7.6–7.9 h；V3-T1 ≈ 21.8 h【未验证】；V4C-12k ≈ 1.8 h；V5-12k ≈ 2.6 h（§7） |
| 阻塞项 | (1) V3-T1-S10 需要先构建 S10 TSDF 缓存（脚本已写；CPU 上 3 个状态逐位核对通过；GPU 约 ≤ 4.6 GPU-h，或 48 核 CPU 约 5 h【未验证】）；(2) V4C / V5 必须等对应 V4 写出 `ema_048000.pt`；(3) V3-T1 统一采样器下单个 micro-batch 最多 12 视角 / 3 状态，比历史的 8 视角多，**正式运行前需要做 4 卡 profile 冒烟确认显存**【未验证】；(4) V3 LR 待用户决定 |

---

## 1. 统一配方（所有臂共用）

| 项 | 统一值 | 依据 |
|---|---|---|
| 训练数据 | Macarons++ Train-6 去 S-c：138 条轨迹 / 98,887 视角 / 13,800 状态；S-c 的 42 条轨迹只做监控 | 划分 `v2_data.load_split`（`task1/scripts/v2_data.py:176-191`，冻结划分 `task1/frozen_values.json`）；数量见 `work_dirs/exp5_future_view/task1/training/main/run_none.json`（train_trajs 138、train_views 98887、train_states 13800）；dry check 两种密度都是 138 / 42、0 缺失 |
| 密度 S10 | `work_dirs/exp5_future_view/task1/data/views/<scene>/<traj>.npz`（T1.1，V2 Original 用的分片） | `v2_data.py:34` 的默认根目录 |
| 密度 D100 | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100/<scene>/<traj>.npz` | `exp6_v4/configs/outdoor_train.json:8` |
| S10 与 D100 的关系 | 180 条轨迹的 t / a / R / T / jitter / k_M / dims / heldout / tstar **逐位相同**；cls / nseg / b / y / tk / dlo / dhi / lu / cat / coarse 不同（这些依赖 P1，所以**训练标签掩码也随密度变化**） | dry check `s10_vs_d100_and_labels`（本次运行）；单条轨迹逐键对比（pisa/cg-A-0000） |
| 步数 | 60,000 步（V4C / V5：从 V4@48k 再训 12,000 步，总计也是 60k） | 用户决定 |
| 全局 batch | 每步 64 视角 / 8 状态 = 4 rank × 16 视角 / 2 状态 | `t12_train.py:227-229`、`train_v4.py:157-158`、`train_v5.py:283-284`；V3：`trainer_outdoor.py:128-130` + accum |
| 采样规则 | 状态权重 t ≤ 20 × 2；每状态最多 8 个候选；陈旧度 Δ ∈ [1, min(9, t)]，p = 0.5；每个 rank 的流为 `np.random.default_rng([seed, rank])` | `v2_data.py:64-78`；`task1/configs/data_v1.json:18-20`；`t12_train.py:196` |
| 损失（共同部分） | Rectified-Flow 速度 MSE，只在 UNKNOWN 且标签非零的射线上计算，除以 2 × **全局**有效射线数 | `v2_data.py:141-150`；`t12_train.py:234-241`；V3 `seen2scene/v3/losses.py:18-27` + `trainer_outdoor.py:183-199` |
| 优化 | AdamW β (0.9, 0.99)、wd 0.01、梯度裁剪 1.0；峰值 LR 2e-4，线性预热 1k，余弦退火到 0 | `train_main.json:7-8`；`t12_train.py:38-42, 195, 261` |
| 精度 | bf16 autocast（编码器 / 射线 / U-Net），TF32 开 | `v2_data.py:131, 146`；`t12_train.py:166-167` |
| EMA / 种子 | EMA 0.999（每步 lerp）；seed 0，torch 种子 seed + 1000·rank | `t12_train.py:184-190, 263-267` |
| 保存 | 完整状态 `ckpt_last.pt` 不超过每 2k 步一次（V2 2,000；V4 / V3 / V5 1,000）；EMA 每 2,000 步保存一次（含 10k / 20k / 40k / 48k / 60k）；另外每个 checkpoint 步都写各 rank 的 RNG 快照 `rng_states/` | 本草案 configs；§5 |
| 结果权重 | 固定取最后一步的 EMA（@60k；V4C / V5 取 @12k）；S-c 只做监控，不参与选择 | `exp6_v4/FROZEN_PROTOCOL.md:28`（固定 EMA 规则） |

---

## 2. 各架构：历史配方 → 统一配方

### 2.1 V2 Original（训练器 `experiments/exp5_future_view/task1/scripts/t12_train.py`，原样调用）

| 项 | 历史（task1 main） | 统一 S10（`configs/v2_s10.json`） | 统一 D100（`configs/v2_d100.json`） |
|---|---|---|---|
| 数据 / 密度 | S10 分片（`v2_data.py:34`；运行记录了 180 个分片的 sha256） | 同历史 | D100 分片（`uo_common.patch_v2` 替换 ViewData 的默认根目录） |
| 步数 / LR | 60k；2e-4，预热 1k，余弦到 0（`train_main.json:7`） | 同 | 同 |
| batch | 64 / 8；4 rank × 16（`run_none.json` world_size 4；`log.jsonl` 中每 rank views = 16） | 同 | 同 |
| 损失 | FM（`v2_data.py:141-150`） | 同 | 同 |
| 初始化 / EMA | 从零训练（`t12_train.py:184-185`）；EMA 0.999 | 同 | 同 |
| 保存 | ckpt_every 5000 → 每 5k 一个 `ema_<step>.pt` + `ckpt_last.pt`（`train_main.json:10`；`t12_train.py:290-293`） | **ckpt_every 2000**（每 2k 一个 EMA 和完整状态） | 同左 |
| 选择 | 固定 EMA@60000（`train_main.json:13`） | 同 | 同 |
| 改动的键 | — | version、protocol、data_root（新增）、ckpt_every | 同左 |

### 2.2 V4 = V2 + RayXAttn（训练器 `experiments/exp6_v4/scripts/train_v4.py`，原样调用）

| 项 | 历史（`exp6_v4/configs/outdoor_train.json`） | 统一 S10 / D100（`configs/v4_{s10,d100}.json`） |
|---|---|---|
| 数据 | D100（`outdoor_train.json:8`；`train_v4.py:78, 93-94`） | 读 cfg 的 `data_root`（S10 或 D100；训练器本来就读这个键） |
| 步数 | **40k**（`outdoor_train.json:12`） | **60k** |
| LR / batch / 损失 / EMA / 种子 | 2e-4 / 1k / 余弦；64 / 8；FM；0.999；0（`outdoor_train.json:12-13`；`train_v4.py:154-170`） | 同 |
| 模型 | `model_v4.json`（ray_xattn：dim 64、4 头、W_o std 0.005） | 同一个文件，未改 |
| 保存 | ckpt_every 1000；EMA 存于 {0, 1k, 5k, …, 40k}（`outdoor_train.json:15-16`；`train_v4.py:144-150, 223-226`） | ckpt_every 1000（不变）；**EMA 每 2k**（0…60k，含 48k = V4C / V5 的父节点） |
| 选择 | 固定 EMA@40000（`outdoor_train.json:19`） | 固定 EMA@60000 |
| 改动的键 | — | version、protocol、data_doc、steps、ema_save_steps、checkpoint_policy、final_checkpoint（S10 另加 data_root） |

### 2.3 V3-T1（VAE + Cross-Attention）

历史 V3-T1 是**室内 + 室外混合**训练；V3.1 是**只用室外数据、但用 T0 架构**（没有 X-Attn）。统一方案采用 V3.1 的室外训练循环（`experiments/exp6_v3.1/scripts/trainer_outdoor.py`，原样调用），但模型换成 T1。

| 项 | 历史 V3-T1（`exp6_v3/configs/train_v3.json` + `seen2scene/v3/trainer.py`） | 历史 V3.1（`exp6_v3.1/configs/outdoor_train.json`） | 统一 S10 / D100（`configs/v3t1_{s10,d100}.json`） |
|---|---|---|---|
| 数据 | MP3D + Macarons++，每步 32 + 32 视角（`exp6_v3/protocol.md:29`） | 只用室外 D100 分片 + D100 4× TSDF 缓存（`trainer_outdoor.py:53-64`） | 只用室外；分片 = `data_root`，缓存 = `tsdf_cache_root`（D100：`work_dirs/exp6_v3/shared_cache/tsdf4`；S10：`work_dirs/exp_unified_outdoor/cache/tsdf4_s10`，**需新建**） |
| 模型 | `model_v3.json` 文件里 `xattn.enabled=false`，运行时按 `--arm t1` 打开（`trainer.py:105-108`）；23,730,986 参数 | T0（关闭，`trainer_outdoor.py:101` 有断言）；23,539,622 参数 | `model_v3.json` 不改，`patch_v3t1` 在构造前把 enabled 设为 True（和 `trainer.py:105-108` 做法相同）；dry check：23,730,986 参数，`model.xa` 为真，X-Attn 记忆张量形状为 [8, 512, 66] 等 |
| 步数 / LR | 40k；**1e-4**，预热 1k，余弦（`train_v3.json:6-8`） | 40k；1e-4（`outdoor_train.json:12-14`） | **60k；2e-4（待决定，§3 D1）** |
| batch | 每 rank 8 室内 + 8 室外，1 + 1 个状态，accum 1（`trainer.py:204-210`） | 每 rank 2 个 micro-batch × 8 视角 / 1 状态，accum 2（`outdoor_train.json:25`；`trainer_outdoor.py:127-130, 148-149`） | accum 2 不变；**采样器统一为 "v2_pool"**（§3 D3） |
| 损失 | L_FM + 1.0·L_recon + β·KL，β 在 5k 步内从 0 升到 0.02（`train_v3.json:25-30`；`losses.py:30-71`） | 同 | 同（架构必需） |
| 保存 | ckpt_every 1000；每次评估（5k）保存 EMA | 同 + ema_save_steps（`outdoor_train.json:51-62`；`trainer_outdoor.py:151-166, 252-258`） | ckpt_every 1000；EMA 每 2k，外加每 5k 的评估 |
| 选择 | 按 ES 分数选择 + 早停规则（`train_v3.json:47-61`）；实际选中 EMA@40k，未触发早停（`task1_xattn/report/T1_TRAINING.md:3`） | 固定 EMA@40k | 固定 EMA@60k |
| 改动的键 | — | — | version、note、model、train_set（新增）、data_root（新增）、tsdf_cache_root（新增）、steps、lr、ema_save_steps、checkpoint_policy、selection；`unified.v3_arm=t1`、`unified.sampler=v2_pool` |

### 2.4 V4-Continue-12k 与 V5-Warm-12k（训练器 `experiments/exp6_v5/scripts/train_v5.py --arm v4c|v5`，原样调用）

| 项 | 历史（`exp6_v5/FROZEN_PROTOCOL.md:16-25`；`train_v4_continue.json` / `train_v5_outdoor.json`） | 统一 S10 / D100（`configs/v4c12k_from48k_*.json`、`configs/v5warm12k_from48k_*.json`） |
|---|---|---|
| 父节点 | V4 EMA@40k（`work_dirs/exp6_v4/checkpoints/final.pt`，sha 1b31afff…；在 `train_v5.py:51-52` 硬编码，213-214 行断言） | **新的统一 V4 同密度 EMA@48k**：`work_dirs/exp_unified_outdoor/runs/v4_<d>/ema_048000.pt`；`patch_v5` 替换 V4CK / V4SHA，首次启动时把 sha256 写入 `<out>/PARENT.json`，之后每次启动都核对 |
| 数据 | D100（`train_v5.py:49` 硬编码 OUTD） | `data_root`（替换 OUTD）；V5 的多表面标签 `work_dirs/exp6_v5/labels/v1` 与密度无关，两种密度共用（§6） |
| 步数 / batch / EMA / 种子 | 12k；64 / 8；0.999；0 | 同 |
| V4C 的 LR | 只有继承参数组，1e-5，预热 500，余弦到 0 @12k；AdamW 重新初始化；从 EMA 权重开始（`train_v5.py:228-231, 296-305`） | 同（沿用历史配方） |
| V5 的 LR 与阶段 | dec 1e-4（预热 200）；继承部分从第 2,000 步起 1e-5（预热 500）；0–1,999 为 Bridge 阶段（encoder / unet 冻结）（`train_v5.py:240-248, 289-305`） | 同 |
| V5 损失 | L_FM + 0.37·L_surf + λ_bridge(s)·L_bridge，λ_bridge = 0.096·max(0, 1 − s/4000)（`train_v5_outdoor.json:107-109`；`train_v5.py:307-311, 438-453`） | 同（λ 不重新标定，§3 D6） |
| 保存 | ckpt_every 1000；EMA 存于 0 / 2k / … / 12k；固定 EMA@12k | 同 |
| 改动的键 | — | version、arm、init（父节点说明）、data_root（S10）；V4C 另改 stages，V5 另改 model（改为相对仓库根目录的路径，内容是同一个文件） |

---

## 3. 剩余差异清单（逐条说明理由与证据）

| # | 差异 | 类型 | 理由 / 证据 |
|---|---|---|---|
| D1 | **V3-T1 的 LR：统一配置写 2e-4（历史为 1e-4）** | **待用户决定** | 1e-4 是事先拟定的初稿值（`EXP6_V3_FROM_SCRATCH_PLAN.md:167`："可尝试峰值 1e-4 作为可执行初稿 …… 数值是否稳定必须以 Smoke 确认"，并提到 KL 训练敏感）。G0 只测过 1e-4（梯度范数 5–15、无 NaN；`task0_vae/report/T0_G0_TESTS.md:31`、`task1_xattn/report/T1_G0_TESTS.md:31`）。**从未测试 2e-4，因此"2e-4 不稳定"没有任何证据。** V4 冻结协议也承认 2e-4 与 1e-4 的差异让跨架构对比只是"匹配预算的性能对照"（`exp6_v4/FROZEN_PROTOCOL.md:33`）。草案取 2e-4 以保持统一，并预先登记回退规则：训练器在损失非有限时会 STOP（`trainer_outdoor.py:200-201`），一旦 STOP，先把失败的 `runs/v3t1_<d>` 改名保存，再用 `run_arm.sh v3t1_<d> --override lr=0.0001` 从零重跑（之后每次 resume 也要带同样的 `--override`），并记入偏差。另一个选择是保留 1e-4 作为架构例外。建议先跑约 2k 步、2e-4 的 4 卡冒烟（≈ 0.7 h） |
| D2 | V3 每步分 2 个 micro-batch（accum 2） | 架构 / 显存 | L_FM 按整步的全局有效射线数归一化（`trainer_outdoor.py:183-199`），梯度等于 64 视角整批的梯度（V4 G0-3 已验证，`exp6_v4/report/G0_VALIDATION.md:9`）。只有 L_recon / KL 的"按状态平均"与 micro-batch 的切分方式有关（`losses.py:30-64`）。显存证据：V3.1 每个 micro-batch 8 视角时峰值 23.6 GB；历史 V3-T1 单次前向 16 视角（8 室内 + 8 室外）时峰值 33.2 GB（本次从 log.jsonl 统计） |
| D3 | V3 的采样器：历史做法是"按 rank 切分轨迹 sorted[rank::world]、每个 micro 各抽 8 视角"（`trainer_outdoor.py:60, 148-149`）；统一做法是 "v2_pool" | **统一协议引起的改动** | 统一后，每个 rank 都加载全部 138 条轨迹，顺序与 V2 划分相同；每步每 rank 用一次 V2 规则抽 16 视角，再按状态边界切成 2 份。dry check：1,200 次抽样中，V3 两份拼起来与 V2 `ViewData`、V4 `CompactViewData` 的批次**逐一相同（1200/1200）**，所以 V2 / V4 / V3 每步看到的（状态、视角、陈旧度）完全一样。代价是单个 micro-batch 最多 12 视角 / 3 状态（分布：8 视角 22%，9 视角 40%，10 视角 30%，11 视角 8%，12 视角 0.3%）。**显存需要 4 卡 profile 冒烟确认【未验证】**；如果 OOM，可把 `unified.sampler` 改成其他值，回到历史采样器（记入偏差）。每个 rank 的内存约 15 GB【估计】 |
| D4 | V3 的输入是 4× TSDF 缓存，不是 V2 的 8³ 粗网格 | 架构必需 | 缓存与分片来自同一个 P1 状态：构建时每个状态都逐位核对粗网格（`exp6_v3/shared/build_tsdf_cache.py:117-118`）。S10 版本见 §6 |
| D5 | V3 的 VAE 重建 + KL 损失（λ_rec 1.0、β 0→0.02 / 5k） | 架构必需 | `losses.py:1-10, 67-71`；`train_v3.json:25-30` |
| D6 | V5：L_surf 0.37、L_bridge 0.096→0（4k 步）、Bridge 2k、参数组 LR | 架构必需（新解码器的热启动） | λ 是在**历史的 V4@40k-D100** 上用 G0 梯度探针冻结的（`exp6_v5/FROZEN_PROTOCOL.md:22`）。统一方案沿用冻结值，不重新探针；新父节点（48k、且有 S10 版本）的梯度比例可能不同【未验证】。Bridge 的教师模型 = 同密度的父节点 V4@48k |
| D7 | V4C / V5 沿用历史续训方式：AdamW 重新初始化、从 EMA 权重开始、LR 1e-5 加 500 步预热 | 沿用历史配方 | 和"V4 主线从 48k 直接训到 60k"并不相同：主线在 48k 时 LR ≈ 1.97e-5，之后余弦降到 0，且保留 Adam 状态。因此 V4@60k 与 V4C-12k 是"同预算、不同续训方式"的对照，报告中必须写清 |
| D8 | 监控（评估）频率不同：V2 采样评估每 10k，V4 每 5k，V3 每 5k，V5 每 2k | 只影响监控 | 评估使用显式的 torch.Generator（`t12_train.py:71-72, 90-91`；`seen2scene/v3/eval_dev.py:56, 75`），只在 rank 0 上运行，不改变训练的 batch 流 |
| D9 | V2 用 `ViewData`，V4 / V5 用 `CompactViewData` | 无影响 | 返回的张量相同（`task2/scripts/t22_data.py:49-54`）；dry check 中两者抽样 1200/1200 相同 |
| D10 | GPU 数值不确定性（scatter / grid_sample 反向等） | 无法消除 | 即使用相同 RNG 重跑，也不能保证逐位相同 |

不再存在的差异：步数（全部 60k）、全局 batch、seed、EMA、精度、FM 定义、划分、采样分布、保存频率、固定最终 EMA。

---

## 4. V2 Original 能否作为统一 V2-S10（复用核查）

| 核查项 | 历史运行 | 统一 V2-S10 | 结论 |
|---|---|---|---|
| 配置 | `train_main.json`，sha 1fcbc77f…（`run_none.json`） | 同一份配方；只有 ckpt_every 从 5000 改为 2000（只影响保存频率，不消耗 RNG） | 相同 |
| 数据 | 180 个 S10 分片的 sha256 记在 `run_none.json` | 同一目录 | **180/180 sha256 与当前磁盘一致**（`work_dirs/exp_unified_outdoor/verify/v2_reuse.json`） |
| 划分 / 采样 / batch / world | train6_minus_Sc，138 条；每 rank `default_rng([0, rank])`；4 × 16；world 4 | 同 | 相同 |
| LR / 步数 / 种子 / 精度 / EMA | 2e-4 / 1k / 余弦 / 60k / 0 / bf16 / 0.999 | 同 | 相同 |
| 是否 resume | resumed_at 0；log 从 1 到 60000 单调（1,201 条，无回退） | — | 从未中断，所以与"统一包装器下不中断运行"的 RNG 流一致 |
| 代码 | `t12_train.py` sha 23ce11cc…，与现在相同 | — | 相同 |
| | `v2_data.py`：运行时 f984c514…，现在 a8f84acd… | — | 与 T1.2 快照对比，只多了 `load_split` 的 train3 分支（`v2_data.py:183-184`），train6 路径不执行这段 |
| | `model.py`：运行时 0e501e74…，现在 9a30386b… | — | 是 V4 的重构；关闭 ray_xattn 时与 V2 逐位相同（`exp6_v4/report/G0_VALIDATION.md:17`，G0-11a） |
| checkpoint | 见下表 | 需要 10k / 20k / 40k / 48k / 60k | **缺 48k** |

`work_dirs/exp5_future_view/task1/training/main/` 的 sha256（CPU 计算，`verify/v2_reuse.json`）：

| 文件 | 步 | sha256 |
|---|---|---|
| ema_010000.pt | 10k | 174d91fb0376b53d9a964a6fafe86c4382834bbac88e6b458be5241e647b0d30 |
| ema_020000.pt | 20k | 35acfeeb768728894497056ecc546f4b12b05e2d1e1c3dd7caffdc71c62c7dce |
| ema_040000.pt | 40k | 20f37de59bbc4b1f9a0622a00046aaeb18ebaff96a26624d82eea81320e6da10 |
| ema_045000.pt | 45k（48k 的近邻） | 158f4464df46304119793dc45a3811de1687f62171af985956bd017fa1572c2c |
| ema_050000.pt | 50k（48k 的近邻） | a798d4c1d4d5d43d6418542f41d59f86bd3e347577834d5309b5f85355d48cdc |
| ema_060000.pt | 60k（最终） | a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc（与运行记录、`CHECKPOINTS.md:14` 一致） |
| ckpt_last.pt | 60k 原始完整状态 | 7e827ecc2ce5360c62328804d04e8142c4d4475163a5cb0c7d315b6f024f7110 |
| ema_048000.pt | 48k | **不存在**（保存间隔是 5k） |

建议：**V2-S10 直接复用（10k / 20k / 40k / 60k）**，节省约 4.7 h × 4 GPU。48k 在本协议中只作为 V4 父节点使用，V2 不需要。若一定要 V2 的 48k 匹配点，有三种做法：(a) 用 45k / 50k 两点夹住，不训练；(b) 用统一包装器把 V2-S10 从 0 跑到 48k（约 3.7 h；批次与噪声流与历史相同，但数值不保证逐位相同，见 D10）；(c) 整条 60k 重跑（约 4.7 h），得到完全自洽的一条线。

---

## 5. Checkpoint / Resume（逐个训练器）

| 训练器 | `ckpt_last.pt` 保存内容 | 频率 | EMA 文件 | resume 恢复什么 | 缺失 |
|---|---|---|---|---|---|
| V2 `t12_train.py` | model、ema、opt、step，以及**仅 rank 0** 的 np / torch / cuda RNG（137-142） | ckpt_every（290-291） | 每个 ckpt_every 一个 `ema_<step>.pt`（292-293） | model / EMA / opt / step（200-202）；DDP：RNG 按 [seed, rank, step0] 重新播种（203-205）；单卡：精确恢复（206-208） | 各 rank 的 RNG |
| V4 `train_v4.py` | 同上（调用 `TR.save_ckpt`，223-224） | 1000 | ema_save_steps（144-150、225-226） | 同 V2 规则（116-127） | 各 rank 的 RNG |
| V3 `trainer_outdoor.py` | `seen2scene/v3/trainer.py:73-78`：同上，RNG 也只有 rank 0 | 1000（257-258） | 每次评估 + ema_save_steps（151-166、253-256） | 119-126：**无论单卡还是多卡都重新播种** | 各 rank 的 RNG |
| V5 / V4C `train_v5.py` | 同上 + V5 查询生成器 `gen`（511） | 1000 | ema_save_steps（404-406、512-513） | 252-265：DDP 下 numpy / torch / gen 全部重新播种 | 各 rank 的 RNG + gen |

- **不需要 scheduler 状态**：LR 是 step 的纯函数（`t12_train.py:38-42`、`seen2scene/v3/trainer.py:61-65`、`train_v5.py:72-79, 296-305`）；V5 的阶段和 λ_bridge（`train_v5.py:289-294, 307-311`）、V3 的 β（`losses.py:67-71`）也都是 step 的函数。数据顺序由固定的轨迹列表和 RNG 决定。
- **补丁（全部是新文件）**：`scripts/uo_common.py` 的 `ExactResume`。它挂在每个训练器的逐步 LR 函数上（`t12_train.lr_at`，V2 / V4 共用；`trainer_outdoor.lr_at`；`train_v5.lr_cos`），这些函数每步在抽 batch 之前调用一次。在每个 checkpoint 步开始时，从训练器 `main()` 的栈帧读到正在使用的 `rng` / `gen` 对象，保存 `rng_states/s<step>_r<rank>.pt`（numpy PCG64、torch CPU、torch CUDA、gen）。resume 时所有 rank 用 all_reduce 确认都有快照，然后原地恢复，接着与不中断的运行完全相同；只要有一个 rank 缺快照，所有 rank 都保留训练器自己的重新播种，并在 `resume_log_r*.jsonl` 里记为 inexact。唯一的窗口是"rank 0 写完 ckpt、barrier 之后"到"下一步 LR 调用"之间（毫秒级）。
- **验证**：CPU 单元测试中，假训练器（局部变量名与真实训练器相同）不中断跑 10 步，与"跑 7 步、被杀、在第 6 步 resume"的结果逐个相等；删掉快照后正确回退并记为 inexact（dry check `exact_resume_unit_test` OK）。**4 卡 DDP 上的 A/B resume 测试尚未做【未验证】**，建议作为 G0：同一臂用 `--out` 指向临时目录，A 不中断跑 40 步，B 在第 20 步 checkpoint 之后杀掉再重启，比较 model / EMA 的差异（预期只剩 D10 的数值噪声）。

---

## 6. 数据与缓存

| 项 | D100 现状 | S10 需要 | 成本 |
|---|---|---|---|
| V2 / V4 / V4C 的视角分片 | 4.7 GB，180 条 | 5.3 GB，180 条，**已存在** | 0 |
| V3 的 4× TSDF 缓存 | `work_dirs/exp6_v3/shared_cache/tsdf4/outdoor`：180 条，磁盘 7.9 GB（blob 合计 6.26 GB）；各轨迹构建时间合计 16,562 GPU-s ≈ **4.6 GPU-h**（按场景平均 49–161 s/条） | **需要新建**：`scripts/build_tsdf_cache_s10.py`（新文件）= 历史 `build_outdoor`（`build_tsdf_cache.py:102-130`），只换两处：融合改用冻结的 S10 `m_p1.P1Fusion`（T1.1 构建 S10 分片时用的就是它，`t11_build.py:57`），核对对象改为 S10 分片的 coarse。输入：`data/macaronspp_belief/v1/trajectories/<scene>/<traj>/records`（t = 0..99）+ 场景几何 + S10 分片。输出：`work_dirs/exp_unified_outdoor/cache/tsdf4_s10/outdoor/` | **CPU 实测**：fushimi/cg-A-0000 和 dunnottar/cg-A-0000 各 3 个状态，与 S10 分片的 coarse **逐位一致**；每个状态 3.2–4.5 s（fushimi）、15–18 s（dunnottar，最大场景），每进程 8 线程。外推：全部约 29 进程·小时，48 核并行约 5 h 墙钟【未验证】；GPU 上不超过 D100 的 4.6 GPU-h【估计：S10 的 BAND 点更少】。磁盘：前 3 个状态的大小约为 D100 的 0.6 倍 → 约 4 GB【估计】。启动：`CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/build_tsdf_cache_s10.sh`（可续跑） |
| V5 多表面标签 | `work_dirs/exp6_v5/labels/v1`：180 条，1.6 GB；历史生成约 4.2 进程·小时（`exp6_v5/reports/V5_RESOURCES.md:9`） | **与密度无关，直接共用**：标签只读分片的 R / T / jitter / k_M / dims / tstar / t / a（`build_multisurface_labels.py:98-100, 124, 242, 255`），这些数组在 S10 与 D100 之间 180/180 逐位相同；标签的 t / a 行与两套分片也都一致（dry check） | 0 |
| V4C / V5 的父节点 | — | 由 V4 臂在 48k 写出 `ema_048000.pt`（V4 的 save_ema 格式为 {ema, step, model_cfg, arm}；V5 的热启动加载已在 CPU 上用历史 V4 EMA 验证：参数 24,988,647） | 包含在 V4 的成本里 |

---

## 7. 吞吐 / 墙钟 / 显存（4 × L40S，来自历史 log）

s/步取 `log.jsonl` 中每 50 步窗口的 `s_per_step`（墙钟，包含评估；中位数基本不含评估）。峰值显存 = rank 0 的 `torch.cuda.max_memory_allocated`（不是 nvidia-smi 的读数）。

| 运行（log） | 步数 | s/步 中位 / 均值 | 实际墙钟 | 峰值显存 |
|---|---|---|---|---|
| V2 Original S10（`task1/training/main`） | 60k | 0.253 / 0.277 | 4.67 h（gpu115） | 14.4 GB |
| V4 D100（`exp6_v4/runs/main`） | 40k | 0.426 / 0.472 | **5.28 h**（日志 20:48:54 → 01:03:33；跨过 11-01 02:00 夏令时回拨，`run_none.json` 起止时间直接相减得 4.31 h，偏小 1 h） | 18.4 GB |
| V3.1 D100（T0 架构，只用室外，accum 2） | 40k | 1.258 / 1.287 | **14.38 h**（同样受夏令时影响，记录为 13.38 h） | 23.6 GB |
| V3-T1 混合（`exp6_v3/task1_xattn/runs/main`） | 40k | 1.374 / 1.434 | 16.13 h | 33.2 GB |
| V3-T0 混合 | 40k | 1.354 / 1.414 | 15.89 h | 33.2 GB |
| V4C（`exp6_v5/runs/v4c`） | 12k | 0.430 / 0.509 | 1.79 h | 18.4 GB |
| V5（`exp6_v5/runs/v5`） | 12k | 0.648 / 0.741 | 2.56 h | 12.9 GB |
| 参考：E2（V2 架构）S10 / D100 | 5k | 0.271 / 0.282（中位） | — | 15.3 GB |

统一运行估计（按均值 s/步 × 步数；数据加载另加约 2–8 min）：

| 臂 | 估计 s/步 | 墙钟 | GPU-h | 备注 |
|---|---|---|---|---|
| v2_s10 | 0.277 | 4.7 h | 18.5 | **建议复用，0 h** |
| v2_d100 | ≈ 0.288 | ≈ 4.8 h | ≈ 19 | 用 E2 的 D100/S10 比例 1.04 外推【未验证】 |
| v4_d100 | 0.472 | 7.9 h | 31.5 | |
| v4_s10 | ≈ 0.45 | ≈ 7.6 h | ≈ 30 | 外推【未验证】 |
| v3t1_d100 / v3t1_s10 | ≈ 1.31 | ≈ 21.8 h | ≈ 87 | V3.1 × (T1/T0 混合比例 1.014)【未验证】；统一采样器使 micro-batch 变大，可能更慢、显存更高 |
| v4c12k_from48k_* | 0.509 | 1.8 h | 7.3 | 需要 V4 跑到 48k |
| v5warm12k_from48k_* | 0.741 | 2.6 h | 10.4 | 需要 V4 跑到 48k |

- 每种密度合计约 155 GPU-h（D100）；S10 在复用 V2 时约 135 GPU-h，再加上 TSDF 缓存。
- 关键路径（每种密度，8 卡）：一组 4 卡跑 V3-T1（约 22 h）；另一组 4 卡依次跑 V2 → V4 →（V4C、V5），约 17 h。V4C / V5 可以在 V4 写出 48k 后立刻开始（不必等 V4 的最后 12k），但需要另外 4 卡。
- 磁盘：每个从零训练的臂约 31 个 EMA × 95–100 MB + `ckpt_last` 0.4 GB ≈ 3.5 GB；每个续训臂约 0.7 GB；两种密度合计约 25 GB。
- 资源不足时的优先级（用户定）：V2 > V4 > (V4C, V5) > V3-T1。

---

## 8. 启动命令（准备好，未执行）

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
bash experiments/exp_unified_outdoor/scripts/dry_check.sh                    # 只用 CPU，不碰 GPU；全部通过时退出码为 0
# 每个臂 4 卡；输出 work_dirs/exp_unified_outdoor/runs/<arm>/，日志 work_dirs/exp_unified_outdoor/logs/train_<arm>.log
# 重新执行同一命令 = 从 ckpt_last.pt + rng_states 精确 resume；已完成的臂直接以 0 退出
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v2_d100     # v2_s10：建议复用历史运行
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v4_d100     # v4_s10 同理
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v4c12k_from48k_d100      # 需要 runs/v4_d100/ema_048000.pt
CUDA_VISIBLE_DEVICES=4,5,6,7 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v5warm12k_from48k_d100
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v3t1_d100
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/build_tsdf_cache_s10.sh            # 仅 S10：先建缓存
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v3t1_s10
# 正式运行前建议的 GPU 冒烟（未执行）：
UO_SMOKE_SCENE=dunnottar CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v3t1_d100 --smoke profile   # 统一采样器 + 最大场景下的显存与 s/步（UO_SMOKE_SCENE 让冒烟只取该场景的 6 条轨迹）
```

退出码：0 = 成功或已完成；2 = 用法 / 环境错误（GPU 数不等于 4、S10 缓存不全等）；3 = 父节点缺失；4 = 训练器正常退出但最终 EMA 不存在；其他值 = torchrun 的退出码。注意：`--smoke` 冒烟会写进 `runs/<arm>/`，冒烟结束后正式运行前应清空该目录（V3 / V4 的冒烟不写 `ckpt_last`，不会被当成 resume 起点；V5 的 `--smoke resume` 会写）。

### 文件清单（全部为新建）
- `configs/*.json`（10 个）：由 `scripts/make_configs.py` 从历史配置生成，`unified.changed_keys` 逐条列出 旧值 → 新值 → 理由；dry check 会重新计算差异，确认没有未记录的改动，并核对历史基准配置的 sha256 没有变。
- `scripts/uo_common.py`（补丁）、`train_unified.py`（入口）、`run_arm.sh`、`dry_check.py` / `dry_check.sh`、`build_tsdf_cache_s10.py` / `.sh`、`verify_v2_reuse.py`。
- 产物：`work_dirs/exp_unified_outdoor/verify/{dry_check.json, dry_check.log, v2_reuse.json, cache_s10_cpu_check*.log}`。
- 注意：仓库 `.gitignore:15` 的规则 `experiments/*` 会忽略本目录（其他实验目录是单独放行的）；按要求本次没有提交，若要纳入 git，需要在 .gitignore 中放行 `experiments/exp_unified_outdoor`。

---

## 9. CPU dry check 结果（2026-11-03，gpu115，`CUDA_VISIBLE_DEVICES=""`，约 2 min，ALL_OK True；最后一次运行 118.6 s）

| 检查 | 结果 |
|---|---|
| 10 个臂：补丁目标都存在、训练器可导入、配置按各训练器自己的方式解析、训练器读取的键全部存在、配置差异 = 记录的改动 | 10/10 OK |
| 参数量与历史一致 | V2 24,822,114；V4 / V4C 24,847,544；V3-T1 23,730,986；V5 24,988,647 |
| 划分 / 分片 | 每个臂 138 + 42，0 缺失 |
| CPU 前向（liberty/cg-A-0000 的 1 个状态、8 视角） | V2 FM 1.92（S10）/ 1.90（D100）；V4 1.87 / 1.85；V3-T1-D100：FM 2.07、recon 1.77、KL 0.21，X-Attn 记忆张量 [8, 512, 66]；V3-T1-S10 跳过（缓存尚未建） |
| V4C / V5 父节点 | 路径与 V4 的 EMA 命名一致；目前不存在（正常）；历史 V4 EMA 能被正确加载 / 热启动 |
| V5 标签 | 0 缺失；行与 S10 / D100 分片都一致 |
| S10 与 D100 的 P1 无关数组 | 180/180 逐位相同 |
| 采样一致性 | V2 = V4 = V3（两份拼接）1200/1200 |
| Exact-resume 单元测试 | 精确恢复：相等；缺快照时正确回退 |
| 冻结文件 | `v2_frozen.verify_frozen()` True；exp6_v4 冻结清单中的代码和配置都没变；`exp6_v4/FROZEN_PROTOCOL.md` 在冻结提交（4cef0f8）里改了 2 行文字（只是文档，记为警告） |
| bash -n | 3 个 .sh 都通过 |

---

## 10. 待用户决定 / 风险

1. **V3-T1 的 LR**：2e-4（统一，草案默认）还是 1e-4（历史，作为架构例外）？
2. **V3 采样器**：用 "v2_pool"（与 V2 / V4 每步批次相同，草案默认）还是历史的按 rank 切分？前者 micro-batch 更大，需要显存冒烟。
3. **V2-S10**：直接复用（建议），还是为了 48k 匹配点重跑？
4. 冒烟门槛（建议，未执行）：V3-T1 的 4 卡 profile；任选一个臂做 DDP A/B resume 测试；V3 若用 2e-4，先跑约 2k 步看稳定性。
5. V5 的 λ 沿用在 V4@40k-D100 上冻结的值，没有在新父节点上重新探针（D6）。
6. V4C / V5 与 V4@60k 的对比属于"同预算、不同续训方式"（D7）。
