# CHECKPOINT_AUDIT —— CVPR Table 1 模型、checkpoint 与对齐审计

- **日期**：2026-11-03
- **性质**：只读审计。没有训练，没有启动闭环，没有使用 GPU，没有修改或删除任何 checkpoint、配置或历史结果。
- **本目录的配套文件**：
  - `MODEL_REGISTRY.yaml`：每个模型一条，含路径、sha256、训练 / 推理 / 评分密度、规划器、启动命令和状态；
  - `EVALUATION_AUDIT.md`：历史室外 episode 清单、覆盖率口径、AUC 定义、运行时间；
  - `audit/historical_outdoor_episodes.{json,csv}`：逐 episode 核对结果，由 `scripts/audit_model_episodes.py` 只读生成。
- **标注约定**：文中“推断”或“未验证”表示没有通过运行或逐位比较确认，只来自读代码或读文档。

## 0. 结论
1. **13 个模型全部 READY**。
   - 主表 8 个：MAG-B、V2O、B0、B2、E2D、V3、V4、V5；
   - 可选 5 个：MAG-G、E2S、B1、V3.1、V4C。
   - 每个模型的权重都已定位，sha256 与已核验的注册表一致；启动链都存在；评估代码自历史运行以来没有变化，或者变化后已逐位复现过（§7）。
2. **13 行只对应 10 份不同的权重**：
   - **B0、B1、B2 是同一份权重**（V2-Mix EMA@17.5k，sha `343f41ab…`），区别只在规划器的 P1 推理 / 评分密度，不能写成三个不同的训练模型；
   - **MAG-B、MAG-G 是同一份权重**（官方 `trained_macarons.pth`，sha `34c5247f…`），区别只在 beam 宽度 / 深度。
3. **sofia_church 在冻结协议下无法评估**，对所有模型都一样（已由同目录 `load_test.log`:15 实测确认，§8.4）。其余 14 个 Macarons++ 场景的数据文件齐全（每个场景 1 个 .obj、1 个 map_Kd、有 `occupied_pose.pt` 和 `settings.json`）。
4. 发现的文档 / 流程问题见 §8，都不影响已有权重与历史结果。

## 1. 来源与核对方法
- **路径与 sha256（前 16 位）**：`experiments/CHECKPOINTS.md`，由 `experiments/checkpoint_registry.py` 于 2026-11-02 18:56:03 生成（CHECKPOINTS.md:3）。
- **完整 sha256**：`work_dirs/checkpoints/exp6_*/MANIFEST.json`。
- **核验结果**：`experiments/CHECKPOINTS_VERIFY.json`。
  - 时间 2026-11-02 19:23:17，七个实验 `all_ok = true`，`orphan_links` 为空；
  - 每个链接都核对了：指向原文件（samefile）、读穿链接后的 sha256、大小、step、参数量，EMA 权重 strict 加载（checkpoint_registry.py:5-7）。
- **本次没有重算已登记 checkpoint 的 sha256**（任务要求：已核验的注册表中有就不重算）。
- **唯一重算的是 MAGICIAN 的 `trained_macarons.pth`**（不在 exp6 注册表中）：
  - 运行时文件、`work_dirs/weights/W00_external/W00-2_magician_official_e697009` 中的副本、`/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/weights/macarons/` 中的副本，三者都是 `34c5247f13bedda08bb57e7e7e7a5baba53af6da7faf3d5f6bb57a27c2dbb59c`；
  - 与 W00-2 `MANIFEST.json`:19-22 一致。
- **闭环中实际加载的权重**：
  - 我们的 11 个组共 330 个历史室外 episode，`planner_hello.ckpt_sha256` 全部等于本表的 sha256（`audit/historical_outdoor_episodes.json` 中的 `sha_ok`）；
  - 规划器 stderr 第一行记录的 P1 模式全部符合预期（§5）。

## 2. 路径与 sha256 总表
注册表软链接都在 `work_dirs/checkpoints/` 下；“实际文件”是软链接的目标。

| ID | 注册表软链接 | 实际文件 | sha256（完整） | 类型 / step | 参数 |
|---|---|---|---|---|---|
| MAG-B / MAG-G | （不在 exp6 注册表）`work_dirs/weights/W00_external/W00-2_magician_official_e697009/macarons/trained_macarons.pth` | `work_dirs/exp5_future_view/task0/magician_comparison/weights/official_drive/macarons/trained_macarons.pth`；运行时经硬链接 `…/external/MAGICIAN/weights/macarons/trained_macarons.pth` 读取（magician_core.py:34-38） | `34c5247f13bedda08bb57e7e7e7a5baba53af6da7faf3d5f6bb57a27c2dbb59c` | 官方发布；另先加载 `pretrained_macarons.pth`（`c7d66c9f…`） | — |
| V2O | `exp6_v2/B2/lineage/v2orig_ema60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt` | `a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc` | EMA / 60000 | 24,822,114 |
| B0 = B1 = B2 | `exp6_v2/B2/v2mix_ema17500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt` | `343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516` | EMA / 17500 | 24,822,114 |
| E2D | `exp6_v2/E2/e2d100_ema5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_005000.pt` | `61439612cb6132afbe74976b82a531c8f827802b06b4f87ad6de1a197d0a2fa5` | EMA / 5000 | 24,822,114 |
| E2S | `exp6_v2/E2/control_e2s10/e2s10_ema5000.pt` | `…/e2/training/s10/ema_005000.pt` | `578855bb149956262f3eba976db837f09e61172542a4412a0404efcb69f1521b` | EMA / 5000 | 24,822,114 |
| V3（V3-T1） | `exp6_v3/T1_xattn/v3t1_ema40000.pt` | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt`（= `runs/main/ema_040000.pt`） | `639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb` | EMA / 40000 | 23,730,986 |
| V3.1 | `exp6_v3_1/V3_1/v31_ema40000.pt` | `work_dirs/exp6_v3_1/checkpoints/final.pt` | `13df0a45df56d125d474a27f0998b9be596be33d0fad1defbb9e82c863263265` | EMA / 40000 | 23,539,622 |
| V4 | `exp6_v4/V4/v4_ema40000.pt` | `work_dirs/exp6_v4/checkpoints/final.pt` | `1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4` | EMA / 40000 | 24,847,544 |
| V4C | `exp6_v5/V4C/v4c_ema12000.pt` | `work_dirs/exp6_v5/checkpoints/v4c_final.pt` | `c4190f1d05dd29d999f23657ac9be381910ad06e3e66d8b6ffbd77b8235cdbe4` | EMA / 12000 | 24,847,544 |
| V5 | `exp6_v5/V5/v5_ema12000.pt` | `work_dirs/exp6_v5/checkpoints/v5_final.pt` | `5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543` | EMA / 12000 | 24,988,647 |

- 出处：CHECKPOINTS.md:11、14、23、25、55、73、92、111、120；各 `MANIFEST.json`。
- 所有我们的模型都用 **EMA（0.999）权重**，没有一个用 raw 权重。raw 训练状态（`*_raw*.pt`）只用于续训。

## 3. 继承关系
```
随机 seed 0 ──60k，Train-6 去 S-c，S10──▶ V2O（a379c0fb）
   └─ +20k 混合（MP3D train + Train-6 去 S-c，S10），冻结规则选 17.5k ──▶ V2-Mix（343f41ab）= B0 / B1 / B2
         ├─ +5k 混合，D100 分片，固定 EMA@5k ──▶ E2D（61439612）
         └─ +5k 混合，S10 分片，固定 EMA@5k ──▶ E2S（578855bb）    （与 E2D 逐步配对）
随机 seed 0 ──40k 混合（MP3D + Train-6），D100 + 部分 TSDF──▶ V3-T1（639c9d3a）    （同协议的 V3-T0 7ed0ddd7 未登记）
随机 seed 0 ──40k 只用室外，D100 + 部分 TSDF──▶ V3.1（13df0a45）
随机 seed 0 ──40k 只用室外，D100──▶ V4（1b31afff）
   ├─ 全部参数 +12k，同数据，L_FM ──▶ V4C（c4190f1d）
   └─ 复制 encoder + U-Net，新解码器，+12k（Bridge 2k + Joint 10k）──▶ V5（5ef30ac4）
作者训练的 MACARONS（官方 Drive）──▶ trained_macarons.pth（34c5247f）= MAG-B / MAG-G
```

出处：
- `experiments/exp6_v2/B2/configs/b2.json`（training_lineage）、`experiments/exp6_v2/E2/configs/e2.json`（training）；
- `experiments/exp6_v4/CHECKPOINT_LINEAGE.md`:5-13；`experiments/exp6_v5/CHECKPOINT_LINEAGE.md`:5-12；
- `experiments/exp_density_ab/CHECKPOINT_AUDIT.md`:25-45（两组 E2 都从 B2 初始化，第 0 步 EMA 与 B2 逐位相等；417 对分片结构逐位相同）；
- 各 `run_none.json` 的 `init` 字段（V5：复制 382 个张量，不用 `ray_xattn` / `rays`）。

## 4. 训练数据、步数与训练密度
| ID | 初始化 | 训练数据 | 室外预训练步数 | 混合微调步数 | 训练密度 | checkpoint 选择 | 训练时代码版本 |
|---|---|---|---|---|---|---|---|
| MAG-B / G | 官方 | MACARONS 作者训练：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | — | — | 完美深度（官方） | 官方 | 官方 e697009 |
| V2O | 随机 | Train-6 去 S-c：138 条轨迹 / 98,887 视角（run_none.json） | 60,000 | 0 | S10 | 固定 EMA@60k（task1/configs/train_main.json:7,13） | 未记录 git；code_sha256（11 个文件）；归档 9fcbdb2 |
| B0 / B1 / B2 | V2O EMA@60k | MP3D train（61 个房屋，114 个楼层，228 条）+ Train-6 去 S-c；32 + 32 视角 | 60,000 | 17,500（运行 20k） | S10 | 冻结 ES 比值规则（task2/protocol.md:57-61），score 1.4087 | 未记录 git；code_sha256（14 个文件） |
| E2D | V2-Mix EMA@17.5k | 同上，D100 重建分片，lr 2e-5 | 60,000 | 17,500（S10）+ 5,000（D100） | D100（最后 5k） | 固定 EMA@5k（e2.json:66） | 未记录 git；code_sha256（16 个文件） |
| E2S | V2-Mix EMA@17.5k | 同上，S10 分片 | 60,000 | 17,500 + 5,000（都是 S10） | S10 | 固定 EMA@5k | 同 E2D |
| V3（T1） | 随机 | MP3D 228 + Train-6 去 S-c 138；E2 D100 分片 + 4× 部分 TSDF 缓存（exp6_v3/protocol.md:8-12） | 0 | 40,000（从零混合） | D100 + TSDF | ES 比值规则（≥5k 中取最小）→ 40k | git 2021604（dirty，diff `bc222222…`） |
| V3.1 | 随机 | Train-6 去 S-c；D100 + TSDF；MP3D 0 | 40,000 | 0 | D100 + TSDF | 固定 EMA@40k | git 4cef0f8（clean） |
| V4 | 随机 | Train-6 去 S-c；`e2/out_views_d100`（exp6_v4/configs/outdoor_train.json:8-9）；MP3D 0 | 40,000 | 0 | D100 | 固定 EMA@40k | git 4cef0f8（clean） |
| V4C | V4 EMA@40k（全部参数） | 同 V4，L = L_FM，新 AdamW | 40,000 + 12,000 | 0 | D100 | 固定 EMA@12k | git ad8dc5c（clean） |
| V5 | V4 EMA@40k（encoder + U-Net） | 同 V4 + Train-6 多表面标签 `labels/v1`；冻结 V4 Teacher 只用于 L_bridge | 40,000 + 12,000 | 0 | D100 | 固定 EMA@12k | git ad8dc5c（dirty，diff `7ee76ec4…`） |

- Train-6 = bannerman、dunnottar、fushimi、liberty、pisa、redeemer（task1/configs/data_v1.json:5）。
- S-c 是 Train-6 内留出的 42 条轨迹，用于 checkpoint 选择（task2/protocol.md:26-27）。
- git 信息取自各训练目录的 `run_none.json`：`work_dirs/exp6_v3/task1_xattn/runs/main/`、`work_dirs/exp6_v3_1/runs/main/`、`work_dirs/exp6_v4/runs/main/`、`work_dirs/exp6_v5/runs/{v5,v4c}/`。V2 系列训练时还没有记录 git 提交。

## 5. 推理、评分与规划（室外闭环）
| ID | P1 模式（实测） | 推理输入密度 | DistP1 评分密度 | 编码器刷新 | 规划器 | K | Euler 步数 | 规划区域 |
|---|---|---|---|---|---|---|---|---|
| MAG-B | — | 同一服务记录（10% 采样） | 官方增益 | 每步占据 + novelty；每 20 循环重建表面 | MAGICIAN beam 10 × 10 | — | — | 整个场景 |
| MAG-G | — | 同上 | 同上 | 同上 | beam 1 × 1（一步贪心） | — | — | 整个场景 |
| V2O | sparse（无补丁） | S10 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| B0 | sparse | S10 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| B2 | b2 | D100 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| B1 | full | D100 | D100 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| E2D | b2 | D100 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| E2S | sparse | S10 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |
| V3 / V3.1 / V4 / V4C / V5 | b2 | D100 | S10 | R10 | Greedy-NV | 4 | 10 | 整个场景 |

出处：
- **MAGICIAN**：
  - beam 设置：adapters/magician_planner.py:36、t24_mag_out_planner.py:47；
  - 种子键 `sha256("exp5-T2|MAG|<tag>|<t>")`：t24_mag_out_planner.py:9、50、83-85；
  - 每 20 循环重建表面：macarons_default_training_config.json:64（e697009 克隆）。
- **我们的模型**：
  - 组定义（distp1、K = 4、R10）：t24_v2_out_planner.py:25、t24_out_sim.py:25-26；
  - 10 步 Euler：v2_plan_core.py:25、futureview/predictor.py:11、planner_v3.py:34；
  - P1 模式：p1_full.py:85-96。V3 / V4 / V5 包装把 `T3_P1_MODE` 固定为 b2（sim_out_v3.py:23、sim_out_v4.py:23、sim_out_v5.py:22）。
- **实测**：每个 episode 规划器 stderr 第一行的 `T3_P1_MODE=…`（V2O、B0 无此行，即冻结的稀疏 P1）。

## 6. 对齐说明：System-Level 与 Controlled 比较
### 6.1 所有行共享的部分（环境层面完全对齐）
所有历史 episode 都来自同一套冻结的 M 协议模拟器（`task2/scripts/t24_out_sim.py` → `task1/scripts/v2_sim.run` → exp3b `m_t6_sim.EpisodeSim`），以下各项完全相同：
- 动作网格、合法性规则（官方邻居 → `get_valid_neighbors` → GT 线段检测）、hover；
- 传感器、带种子的观测和 10% 采样；
- 预算（C0 + 100 次移动）、覆盖率评估器；
- 每个组拿到同一模拟器生成的服务记录，没有哪个组多看一帧（task2/protocol.md:63-75）。

### 6.2 System-Level（Table 1 主比较）
每一行是一个完整系统：权重 + 训练配方 + 推理 / 评分密度 + 规划器。与 MAG-B 相比同时变化的因素：
- **规划器**：一步 Greedy-NV 对 beam 10 × 10；
- **世界模型**：K = 4 未来深度 + DistP1，对比 SconeOcc 想象高斯 + novelty 增益；
- **训练数据和训练场景**（§6.4）；
- **计算量**：每个候选 64 × 114 条射线 × K = 4，对比整场景想象高斯渲染（task2/protocol.md:75）。

因此 Table 1 的主结论只能写成“系统对系统”。

### 6.3 可以写成 Controlled 的配对（每对只差一个因素）
| 配对 | 唯一差别 | 备注 |
|---|---|---|
| B0 ↔ B2 ↔ B1 | 规划器的 P1 推理 / 评分密度（S10/S10、D100/S10、D100/D100） | 同一权重（exp_density_ab/CHECKPOINT_AUDIT.md:115-122） |
| MAG-G ↔ MAG-B | beam 宽 / 深 | 同一权重、同一种子键 |
| V4 ↔ V4C | +12k 同配方训练 | 训练长度对照 |
| V4C ↔ V5 | 结构 / 损失（V5 解码器 + 表面损失 + bridge 损失），步数相同 | 同一父节点、同数据、同 batch、同种子 |
| V2O ↔ B0 | +17.5k 混合微调（MP3D + 室外，S10） | 推理都是 S10/S10 |
| B2 ↔ E2D | +5k D100 混合微调 | 推理都是 D100/S10；同时改变了训练密度和训练步数 |

**不是单因素对照的配对**：
- **E2S ↔ E2D**：训练是严格配对的，只有训练分片密度不同（exp_density_ab/CHECKPOINT_AUDIT.md:31-45）。但历史推理也不同（E2S 是 S10/S10，E2D 是 D100/S10），所以是两个因素同时变化（同文件:113）。完全受控的格子（如 E2S|D100|S10、E2D|S10|S10）只在 `work_dirs/exp_density_ab/g3` 的 Dev-3 + Test-3 上存在，不是 Table 1 的行。
- **V3 ↔ V3.1**：结构（T1 有 xattn，V3.1 = T0 结构）和数据（混合 vs 只用室外）同时不同。数据受控的配对是 V3-T0 ↔ V3.1，但 V3-T0 未登记。
- **V3.1 ↔ V4**：都是室外从零 40k、D100，但结构不同（VAE 潜空间 + TSDF 输入 vs V2 + 射线交叉注意力）。
- **V2O ↔ V4**：训练密度、步数、结构、推理密度都不同，只能做系统级比较。

### 6.4 训练场景与评估场景的重叠（影响“seen / unseen”标注）
| 场景 | 我们的室外训练（Train-6） | MACARONS / MAGICIAN 权重 | 历史用途 |
|---|---|---|---|
| bannerman、dunnottar、fushimi、liberty、pisa、redeemer | 训练（含 S-c 选择集） | 训练 | — |
| pantheon、colosseum | 否 | 训练 | Dev-3（V2 开发集） |
| barts | 否 | 否（论文新增室内场景） | Dev-3 |
| alhambra、neuschwanstein | 否 | 验证 | Test-3 |
| bridge | 否 | 测试 | Test-3 |
| eiffel | 否 | 测试 | — |
| sestino_museum | 否 | 否 | — |
| sofia_church | 否 | 否 | 冻结协议下无法运行（模拟器 + E-M2 窗口表，§8.4） |

出处：magician_protocol_facts.md:33-37；task1/configs/data_v1.json:5；task2/protocol.md:28-35。

- **Dev-3 与 Test-3 都不再是盲测**：
  - Test-3 在 Task 2 只用过一次（task2/protocol.md:28）；
  - 之后 B2、E2D、V3、V3.1、V4、V4C、V5 都在这 6 个场景上反复评估；
  - B2 和 E2 的设计也是在看过 B0 / B1 在这些场景上的结果之后决定的；
  - MODEL_VERSIONS.md:8 也写明这些评估场景都不是盲测。
- **另外 8 个 MAGICIAN 测试场景的情况**：
  - 6 个（Train-6）是我们所有室外模型的训练场景；
  - eiffel、sestino_museum 两个场景对我们和 MAGICIAN 都没有参与训练。

## 7. 代码版本与漂移核对（评估端）
- **Task 2 评估冻结时的文件哈希**（task2/FROZEN_CONFIGS.json）：评估阶段的全部条目与当前文件逐个一致，包括：
  - `t24_out_sim.py`、`t24_v2_out_planner.py`、`t24_mag_out_planner.py`、`t25_out_cov.py`；
  - `task1/v2_sim.py`、`v2_planner.py`、`v2_common.py`、`v2_plan_core.py`；
  - exp3b 的 `m_t6_sim.py`、`m_magician_replay.py`；
  - MAGICIAN 适配层 `magician_core.py`、`magician_planner.py`、`magician_planner_grid.py`、`grid_env.py`。
  - 这些文件在本次审计时重新计算 sha256 比对。
- **规划器日志里记录的 code_sha256 与当前文件的比较**：

  | 组 | 记录的文件数 | 变化的文件 |
  |---|---|---|
  | V2O、B0、E2D、V3 | 34–44 | 只有 `seen2scene/futureview/model.py` |
  | B2 | 36 | `model.py` 和 `t32_out_planner.py` |
  | V4、V5 | 36 / 39 | 无 |

  - `model.py` 在 f57be9a（2026-10-31）加入了默认关闭的 `ray_xattn`；关闭时与 V2 逐位相同（exp6_v4/report/G0_VALIDATION.md:17，测试 `test_g0_11a`）。
  - `t32_out_planner.py` 后来加入了 `T3_CKPT` 覆盖（t32_out_planner.py:15-26）；不设该环境变量时行为不变（读代码判断）。
- **用当前代码做过的逐位复现**：
  - `exp_density_ab` G0（2026-11-03）：B0 / B2 / B1 / E2S / E2D 各 1 个 colosseum_s0，执行动作、官方覆盖率、单调覆盖率全部相同（G0_REPRODUCTION.md:30-38）；
  - Camera Pose C0：V4、V5 各 6 个 Dev-3 episode 的覆盖曲线与历史逐位相同（exp_camera_pose/C0/README.md）；
  - Refresh R10 回放：E2D colosseum_s0（exp_temporal_refresh/G3_OUTDOOR.md:4）；
  - exp6_v4 G0-11b：V3-T0 与 V2-Mix 路径 2 步闭环，共 4/4 通过。
  - 结论：MAG-B、MAG-G、V2O、V3-T1、V3.1、V4C 没有针对其自身权重的近期回放。它们的代码路径要么已被上述复现覆盖，要么文件哈希没有变化。对这几个具体 checkpoint 的逐位重现**未验证**。
- **代码不在 git 中的部分**：
  - `experiments/exp3b_macaronspp_transfer/` 整个目录被 `.gitignore:15`（`experiments/*`）忽略，git 中没有任何文件；冻结模拟器（`m_t6_sim.py`、`m_magician_replay.py`、`m_common.py`、`m_p1.py` 等）只靠 sha256 记录保护；
  - `MAGICIAN_clean_9521d69` 和 e697009 官方克隆都在仓库之外。

## 8. 发现的问题与注意事项
1. **E2 的 checkpoint 选择描述不一致**：
   - `exp6_v4/CHECKPOINT_LINEAGE.md`:7 写“ES 比值最小 → 5k”；
   - `exp6_v2/E2/configs/e2.json`:66 已在 2026-11-02 更正为“按协议固定 EMA@5k”；
   - `e2_pilot/configs/e2_d100_v1.json`:49 仍保留从 Task 2 复制来的 `choose` 规则。
   - 实际使用的都是 EMA@5k，不影响结果，但写论文时应引用 e2.json 的说法。
2. **默认输出目录是历史目录**：`t24_out_sim.py`:70 在不给 `--root` 时写入 `work_dirs/exp5_future_view/task2/eval/outdoor/closed_loop/<group>`。所有包装都调用 `T24.main()`，所以都继承这个默认值：t33_out_sim.py:42、da_out_sim.py:41、sim_out_v3.py:39、sim_out_v4.py:39、sim_out_v5.py:38。
   - 例如在新场景上跑 V3 / V4 / V5 时忘了 `--root`，结果会写进历史 B0（Task 2 V2M）目录。
   - 模拟器拒绝覆盖已存在的 episode（v2_sim.py:73），但会混入新文件。
3. **环境变量会改变权重或模式**，且会被规划器子进程继承：
   - `T3_CKPT`：t33 路径下替换权重（t32_out_planner.py:15-26），影响 B0 / B1 / B2 / E2；
   - `V4_P1_MODE=full`：V4 / V4C / V5 改为 D100 评分（sim_out_v4.py:23、sim_out_v5.py:22）。
   - 新运行前应 `unset`；运行后应核对 episode 的 `planner_hello.ckpt_sha256` 和规划器 stderr 第一行。
4. **sofia_church 无法评估（模拟器端已实测；规划器端为读代码结论）**：
   - 冻结模拟器从 `MAGICIAN_clean_9521d69` 导入 `load_scene`（m_magician_replay.py:38、47；m_t6_sim.py:51）；
   - 该函数在纹理数 > 1 时进入多纹理分支，其中有一行 `torch.tensor()`（MAGICIAN_clean_9521d69/macarons/utility/macarons_utils.py:470-494），会抛出 TypeError；
   - sofia_church 的 .mtl 有 4 个 `map_Kd`，其余 14 个场景各 1 个；
   - 官方 e697009 克隆中这一行已注释（external/MAGICIAN/macarons/utility/macarons_utils.py:481），MAGICIAN 自己的 LMDB 也没有 sofia_church；
   - 实测：同目录 `load_test.log`:15（数据集审计代理运行）在 EpisodeSim 阶段报 `TypeError: tensor() missing 1 required positional arguments`；去掉该行的进程内补丁可以加载（`sofia_diag_result.json`），但那已不是冻结协议；
   - 规划器端还有第二处阻断：我们的 V2 系列规划器（V3 / V4 / V5 也走这条路径）在 v2_planner.py:112 调用 `t5_diagnostic.scene_geometry` → `frame_of`（t5_diagnostic.py:110-112），读取 `work_dirs/exp3b_macaronspp_transfer/task1_protocol/em2_windows.json`。该文件只有 MAGICIAN 的 14 个测试场景，没有 sofia_church，会抛出 KeyError。另外 8 个场景（bannerman、dunnottar、eiffel、fushimi、liberty、pisa、redeemer、sestino_museum）都在该文件中；
   - 这与 magician_protocol_facts.md:34 记录的“loader 多纹理 bug”一致。若 SCENE_REGISTRY 包含 sofia_church，需要同时改模拟器和 E-M2 窗口表，这属于协议变更。
5. **MAGICIAN 根目录的工作树有未提交改动**：`git diff --stat`：`macarons/utility/macarons_utils.py` +277 / −21。
   - 冻结模拟器（用 clean 9521d69）和 MAG-B（用 e697009 克隆）都不使用它；
   - 但 `magician_lmdb` 是 2026-06 从该根目录跑出来的，当时的代码状态无法从文件确定（未验证）。
6. **命名提醒**：
   - 注册表目录 `work_dirs/checkpoints/exp6_v2/B2/` 存放的是 V2-Mix（B0 / B1 / B2 共用）和 V2O 的 lineage，不代表有一个“B2 专用训练”；
   - `task1/configs/closed_loop_v1.json` 中的 “B2” / “E2” 是 EXP5 Task 1 / EXP4 的旧名，与本表无关（MODEL_VERSIONS.md:44）。
7. **未登记的模型**：
   - V3-T0（`exp6_v3/T0_vae/v3t0_ema40000.pt`，`7ed0ddd7…`）有完整的 30 个 Dev-3 + Test-3 episode（`work_dirs/exp6_v3/eval/closed_loop/outdoor/T0`），但用户没有列入；
   - V5.2 / V5.2-M 只有 pilot / smoke，冻结门槛未通过（MODEL_VERSIONS.md:40-41），不适合进 Table 1。
