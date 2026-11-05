# TRAINING_DATA_COMPARISON — 我们的模型与各基线的训练数据对照（A2，第二部分）

- **来源**：`drafts/TRAINING_FACTS_DRAFT.md` 的 §B、§C。
- **复核**：其中 V3-T1 数据量与 Train-6 起点重叠两项已由主会话独立复核（见 `MAGICIAN_TRAINING_AUDIT.md` 开头）。
- **原则**：训练数据不同，本身并不等于比较不公平；本文件只做披露。

- **标记**：【已核实】= 本次直接读到原文 / 代码 / 文件；【推断】= 由代码或数据推出，未运行确认；【未知】= 证据缺失，注明缺什么。
- **路径缩写**（R = `/bsuscratch/jiahuizhang/projects/occupancy/seen2scene`，下文相对路径都相对 R）

| 缩写 | 指向 |
|
## B. 我们的模型

### B1. 总表（sha 取前 8 位；完整值见 REG 的对应行）

| 行 ID（REG） | sha | 初始化 | 室外训练数据 | MP3D | 本模型训练步数 | 训练密度 | checkpoint 选择 | 主要证据 |
|---|---|---|---|---|---|---|---|---|
| V2O（REG:102-109） | a379c0fb | 随机，seed 0 | train6_minus_Sc，138 条轨迹 / 98,887 视角 | 无 | 60,000 | S10 | 固定 EMA@60k | `experiments/exp5_future_view/task1/configs/train_main.json:6-8,13`；`task1/scripts/v2_data.py:176-191`（`load_split`）；CHECKPOINT_AUDIT.md:79 |
| B0 = B1 = B2（REG:147-152、444-449、189-194） | 343f41ab | V2O EMA@60k（sha 断言） | 同上，S10 | MP3D 官方 train：**59 个房屋 / 114 个楼层 / 228 条轨迹** | 运行 20,000，选出 17,500 | S10 | ES 比值规则，选择集 = MP3D valsel（6 个房屋 / 9 个楼层）+ S-c；不看闭环 | `task2/configs/train_mix_v1.json:6-9,14`；`task2/scripts/t22_train_mix.py:68-69,88-89`；`task2/protocol.md:26-27,57-61`；`work_dirs/exp5_future_view/task2/mp3d/manifests/traj_jobs.json`（本次统计：train 228 jobs / 114 层 / 59 房屋，valsel 9 / 9 / 6） |
| E2D（REG:224-229） | 61439612 | V2-Mix EMA@17.5k（sha 断言） | 同上，D100 重建分片 | 同上（D100） | +5,000 | D100 | 固定 EMA@5k | `task3_full_depth_p1/e2_pilot/configs/e2_d100_v1.json:6-9,14`；`experiments/exp6_v2/E2/configs/e2.json:66` |
| E2S（REG:415-420） | 578855bb | 同上 | 同上，S10 | 同上（S10） | +5,000 | S10 | 固定 EMA@5k | `e2_pilot/configs/e2_s10_v1.json:6-9,14` |
| V3（= V3-T1 40K，REG:267-272） | 639c9d3a | 随机，seed 0 | `e2/out_views_d100` 中非 heldout 部分（= train6_minus_Sc）+ TSDF 缓存 | **有**（混合，32 + 32 视角） | 40,000 | D100 + TSDF | ES 比值（MP3D in_val + S-c，≥ 5k 取最小）→ 40k | `experiments/exp6_v3/configs/train_v3.json:5-6,47-51`；`seen2scene/v3/trainer.py:117-118`；`seen2scene/v3/data.py:43-58`。运行记录 `work_dirs/exp6_v3/task1_xattn/runs/main/run_none.json` 的 `data_counts` = {in_train 228, in_val 9, out_train 138, out_sc 42}，`data_files` 中 out_train / out_sc 只有 6 个 Train-6 场景 |
| V3-Y45K-H0 / V3-Y50K-H0（REG:653-658、665-670） | 37da75b2 / 64cea1d6 | V3-T1 真实 40K 训练状态（`ckpt_last` `3d819782…`，含 AdamW） | 与 V3 **完全相同**（混合） | **有** | +5,000 / +10,000（在 50K 停止） | D100 + TSDF | **按 Dev-5 闭环 AUC@100 选**：Y45K 0.732 < Y50K 0.741 → Y50K | `experiments/exp6_v3/Y/configs/train_y.json:59-61,67-73`；`Y/FROZEN_PROTOCOL.md:7,21,29,33`；`Y/scripts/train_y.py:2-8`（共享训练器不改）；运行记录 `work_dirs/exp6_v3/Y/runs/main/run_none_1793791215.json` 的 `data_counts` 与 V3 相同，`resumed_at` 40000；`Y/reports/G3_DEV_AND_ALL14.md:9-10` |
| V3-H3、V3-D1-H3-NORISK（REG:590-595、605-610） | 639c9d3a | = V3 | — | — | **0**（只换规划器） | — | D1 是在 D1 / D2 / D3 中按 Dev-5 AUC@100 选出的 | `exp6_v3/H/MODEL_AUDIT.md:10`（"本实验中的训练 **0 次**"）；`exp6_v3/ablation_study/experiment_D_h3_lookahead/FROZEN_PROTOCOL.md:29`；`H/FROZEN_PROTOCOL.md:51` |
| V3-YBEST-H3（REG:677-682） | 64cea1d6 | = Y50K | — | — | 0（只换规划器） | — | 权重是按 Dev-5 选出的（见上） | `Y/FROZEN_PROTOCOL.md:33-36` |
| V3.1（REG:468-473） | 13df0a45 | 随机（`scratch`） | 只用室外（138 条轨迹 + S-c 42 条作监控），D100 + TSDF | 0 | 40,000 | D100 + TSDF | 固定 EMA@40k | `exp6_v3.1/configs/outdoor_train.json:5-9,12,63-65`；`exp6_v3.1/scripts/trainer_outdoor.py:53-59`；运行记录 `data_counts` {out_train 138, out_sc 42} |
| V4（REG:308-313） | 1b31afff | 随机（`scratch`） | train6_minus_Sc，`out_views_d100` | 0 | 40,000 | D100 | 固定 EMA@40k | `exp6_v4/configs/outdoor_train.json:5,8-12,20` |
| V4C（REG:497-502） | c4190f1d | V4 EMA@40k，全部参数 | 同 V4 | 0 | +12,000 | D100 | 固定 EMA@12k（"no selection on S-c, Dev-3, Test-3 or MP3D"） | `exp6_v5/configs/train_v4_continue.json:2-7,43` |
| V5（REG:349-354） | 5ef30ac4 | V4 EMA@40k（encoder + U-Net） | 同 V4，另加 Train-6 多表面标签 | 0 | +12,000（Bridge 2k + Joint 10k） | D100 | 固定 EMA@12k | `exp6_v5/configs/train_v5_outdoor.json:2-7,43,102,114` |
| F1-GQ-v1@5k（REG:545-550） | a3f47c03 | B0 的 encoder / UNet2D 逐位复制 + 随机 decoder（因此**间接**继承了 MP3D 混合训练） | train6_minus_Sc，S10 | 本阶段 0 | +5,000 | S10 | **按 Dev-3 闭环 AUC@20 在 2k / 5k 中选** | `exp6_v5/F1/PROTOCOL.md:26,30-31,42,81-84`；`F1/reports/TABLE1_FOLLOWUP.md:4`；运行记录 `work_dirs/exp6_v5/F1/runs/B/run_none.json`：`train_trajs` 138、`train_views` 98,887、`mp3d_training_samples` 0 |

- **外部预训练**：所有行都没有 ShapeNet 或 ImageNet 预训练权重（【推断】）。依据：
  - 配置是 `scratch`，或从我们自己的父 checkpoint 初始化；
  - `seen2scene/futureview`、`seen2scene/v3` 以及上述配置目录中 grep "shapenet"、"imagenet"、`from_pretrained`、`torch.hub` 都没有结果。
- **监督信号**：标签是用光栅化器渲染 GT 网格得到的第一交点 t*（`task2/protocol.md:43`）。这是 GT 3D 监督，不同于 MACARONS 的 RGB 自监督（【推断】：依据是"光栅化器 … 第一交点"这一描述，没有逐行追代码）。V5 另用 GT 网格的多表面标签（`train_v5_outdoor.json:114`）。

### B2. Train-6 的起点核对（本次对 BEL 的 `sim_meta.json` 与 settings 的 `start_positions` 逐条比对；划分取自 FV）【已核实】

| 场景 | S-c 留出的官方起点 | 训练集中以官方起点 k 开始的轨迹数 | 训练集中途经官方起点 k 的轨迹数（任意步） | 起点 4 |
|---|---|---|---|---|
| bannerman | 0 | k1: 5、k2: 4、k3: 4（另有 9 条非官方起点） | 1: 5、2: 4、3: 4 | 从未出现 |
| dunnottar | 3 | k0: 6、k1: 5、k2: 4 | 0: 6、1: 5、2: 4 | 从未出现 |
| fushimi | 0 | k1: 5、k2: 4、k3: 4 | 1: 5、2: 4、3: 4 | 从未出现 |
| liberty | 2 | k0: 6、k1: 5、k3: 4 | 0: 6、1: 5、3: 4 | 从未出现 |
| pisa | 3 | k0: 6、k1: 5、k2: 4 | 0: 6、1: 5、2: **5** | 从未出现 |
| redeemer | 0 | k1: 5、k2: 4、k3: 4 | **0: 1**、1: 5、2: 5、3: 4 | 从未出现 |

- **划分规模**：每个场景 30 条轨迹，train / S-c = 22–24 / 6–8 条；合计 train 138 条，S-c **42** 条（FV `split.scenes.*`）。
- **训练集途经 S-c 起点的唯一一例**：redeemer 的 `rwm-A-0001` 在 t = 17 经过官方起点 0。
- **DATASET_AUDIT §2.4 的原话**是"官方起点 0–3 都有训练轨迹（6、5、4、4 条），起点 4 从未用作起点"。这对 180 条全集成立；在 train6_minus_Sc 中，每个场景只剩 3 个官方起点作为训练起点。
- **Dev-5、Test-3、sofia_church** 没有任何训练轨迹（【已核实】）：
  - BEL 的 `trajectories/` 下只有 6 个 Train-6 目录；
  - `work_dirs/exp5_future_view/task1/data/{views,tstar}` 与 `…/task3_full_depth_p1/e2/out_views_d100` 下也都只有这 6 个场景目录。

### B3. 训练轨迹由哪些策略生成（需要披露）

| 策略 | 是否用 GT | 每个场景的条数（全集 / train） | 证据 |
|---|---|---|---|
| RW、RW-M、FR | 否 | 8 / 6 / 6 | `experiments/exp4_multiworld_belief/task0/trajectory_generation.md:42-44` |
| CG-ε（真实一步增益贪心，ε = 0.2） | **是**（只用来选动作） | 8 | 同上 :45；`sim_meta.json` 中 `"eps": 0.2` |
| **MG = MAGICIAN beam 10×10 + 官方 `trained_macarons.pth`** | 否 | 2；train 中共 **9** 条：bannerman 1、dunnottar 2、fushimi 1、liberty 2、pisa 2、redeemer 1 | 同上 :46；`work_dirs/exp4_multiworld_belief/task0/pilot/mg_runs/bannerman/episodes/ep_magician_bannerman_s0_r0.json` 的 `model.trained_sha256` = `34c5247f…`、`beam` [10, 10] |

- **含义**【推断】：我们所有模型的训练观测分布中，有一小部分来自 MAGICIAN / MACARONS 权重驱动的相机轨迹，以及 GT 驱动的轨迹。这两类只决定"看哪里"，标签仍来自 GT 网格。

### B4. 评估场景在训练之外的使用（选择与开发）

| 用途 | 涉及的行 | 证据 | 标记 |
|---|---|---|---|
| 用 Dev-5 闭环选 checkpoint | V3-Y50K-H0、V3-YBEST-H3 | `Y/FROZEN_PROTOCOL.md:33`；`Y/reports/G3_DEV_AND_ALL14.md:9-10` | 【已核实】 |
| 用 Dev-3 闭环选 checkpoint | F1-GQ-v1@5k | `F1/PROTOCOL.md:81-84`；`F1/reports/TABLE1_FOLLOWUP.md:4` | 【已核实】 |
| 用 Dev-5 选规划器变体 | V3-D1-H3-NORISK | `experiment_D_h3_lookahead/FROZEN_PROTOCOL.md:29` | 【已核实】 |
| Dev-3 是 V2 系列的开发集；B2 / E2 的设计是在看过 B0 / B1 在 Dev-3 + Test-3 的结果之后定的 | V2O、B0/B1/B2、E2D/E2S 及后续 | `task2/protocol.md:35`；CHECKPOINT_AUDIT.md:167 | 【已核实】（文档记载） |
| Test-3 已多次评估，不是盲测 | 全部 | DATASET_AUDIT.md:311-314 | 【已核实】（文档记载） |
| 不用评估场景选择 | V2O、E2D、E2S、V3.1、V4、V4C、V5（固定步数）；B0/B1/B2、V3（只用 S-c + MP3D valsel） | 见 B1 | 【已核实】 |

### B5. 附注：MP3D（Table 1 之外）
- 我们的 MP3D 训练房屋包含 MAGICIAN 5 个 MP3D 评估场景中的 2 个：GdvgFV5R1Z5、HxpKQynjfin；pLe4wQe7qrG 在我们的 valsel 中。
- 影响：只在拿我们的模型与 MAGICIAN 在 MP3D 上比较时需要披露；对 Macarons++ Table 1 没有影响。
- 证据（【已核实】）：`work_dirs/exp5_future_view/task2/mp3d/manifests/traj_jobs.json`（例如 :624、:709、:2100）；OC README.md:63。

---

## C. 公平性披露表（只是披露：训练数据不同本身不等于比较不公平）

| 方法 | ShapeNet 预训练 | 用 Macarons++ 场景数据训练 | 用 MP3D 训练 | 训练过的场景 ∩ 14 个评估场景 | 评估起点重叠 | 评估场景用于选择 |
|---|---|---|---|---|---|---|
| MAGICIAN（MAG-B / MAG-G，`trained_macarons.pth`） | 是（论文 + 配置；【推断】正是这份权重） | 是：8 个场景，RGB 自监督，无 GT 3D | 否（表 5 的微调版本未发布、未使用） | **8 个**：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | 训练起点是全网格随机位姿；精确重合的期望很小（【推断】），途经【未知】 | alhambra、neuschwanstein 标为 val，实际是否使用【未知】 |
| MACARONS（论文列） | 同上 | 同上 | 否 | 同上 | 同上 | 同上 |
| SCONE（论文列） | 是（v1，8 个类别） | 原方法：否；MAGICIAN 实际所用权重【未知】 | 否 | 原方法：0；实际【未知】 | 原方法：不适用 | — |
| FisherRF（论文列） | 原方法：否 | 原方法：否；MAGICIAN 改编版可能用了 MACARONS 权重（【未知】） | 否 | 原方法：0；改编版【未知】 | — | — |
| Random Walk | 否 | 否 | 否 | 0 | 不适用 | — |
| V2O | 否 | 是：Train-6 去 S-c（GT 监督） | 否 | 6 个 Train-6 场景 | 每个 Train-6 场景有 3 个官方起点是训练起点；起点 4 没有；Dev / Test 没有 | 否（Dev-3 作开发参照） |
| B0 / B1 / B2、E2D、E2S | 否 | 同上 | 是 | 同上 | 同上 | 否（B2 / E2 的设计参考过 Dev-3 + Test-3 结果） |
| V3、V3-H3 | 否 | 同上 | 是 | 同上 | 同上 | 否 |
| V3-D1-H3-NORISK | 否 | 同上 | 是 | 同上 | 同上 | 规划器按 Dev-5 选 |
| V3-Y50K-H0、V3-YBEST-H3 | 否 | 同上 | 是 | 同上 | 同上 | checkpoint 按 Dev-5 选 |
| V3.1、V4、V4C、V5 | 否 | 同上 | 否 | 同上 | 同上 | 否 |
| F1-GQ-v1@5k | 否 | 同上 | 间接（经 B0 初始化） | 同上 | 同上 | checkpoint 按 Dev-3 选 |

- 双方都"见过"的场景：Train-6 对我们和 MACARONS 权重都是训练场景。pantheon、colosseum 只对 MACARONS 权重是训练场景，对我们是 Dev。
- 双方都没训练过的场景：eiffel、bridge、barts、sestino_museum。alhambra、neuschwanstein 两边都没训练过，但 MACARONS 一方的 val 用途【未知】。

---
