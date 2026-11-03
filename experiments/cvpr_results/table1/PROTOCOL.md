# PROTOCOL — CVPR Table 1：Macarons++ Active Mapping Benchmark（protocol v1.0）

- **冻结时间**：2026-11-03，在任何新 episode 结果产生之前冻结。
- **机器可读版**：`FROZEN_CONFIG.yaml`。
- **修改规则**：任何改动都必须新建 protocol 版本（v1.1 …），不能静默覆盖。

## 1. 数据集与划分（详见 `SCENE_REGISTRY.yaml`、`DATASET_AUDIT.md`）
**场景**：官方 15 个场景，本地 15 个都有，资产与冻结清单一致。

**划分**：依据冻结的 `exp4_multiworld_belief/task0/configs/dataset_manifest_v1.json`。

| 划分 | 场景 | 说明 |
|---|---|---|
| Train-6 | bannerman、dunnottar、fushimi、liberty、pisa、redeemer | 我们所有室外模型的训练场景；官方起点 0–3 出现在训练轨迹中。**属于见过的场景（in-distribution）** |
| Dev-5 | barts、colosseum、eiffel、pantheon、sestino_museum | barts / pantheon / colosseum 是 Dev-3，历史上多次用于开发与选择；eiffel、sestino_museum 此前没有 V2 系列的闭环 |
| Test-3 | alhambra、bridge、neuschwanstein | **已不是盲测**：历史上已运行过多次（`MODEL_VERSIONS.md`） |
| Other / Unassigned | sofia_church | manifest 中为 `reserved`（不使用） |

**MAGICIAN 自身的训练覆盖**：MACARONS 权重的训练场景包括 pantheon、colosseum 等；alhambra、neuschwanstein 是其验证场景（`DATASET_AUDIT.md`）。所以每个划分对 MAGICIAN 来说同样不是完全未见过的场景。

**汇总组**：
- **All-14**：在冻结协议下可运行的 14 个场景，与 MAGICIAN 官方测试表的 14 个场景相同。这是本表的完整汇总组。
- **All-15**：sofia_church 在冻结的模拟器中无法加载（MAGICIAN `load_scene` 多纹理分支多出的一行 `torch.tensor()`，见 `DATASET_AUDIT.md`）。规划器窗口表中也没有它。因此每个模型的 All-15 只能输出为 `all15-partial14of15`，**不标为 All-15**。
  - 使 sofia_church 可运行需要改动冻结的加载代码并新增派生的场景参数。这属于协议改动，记为 PENDING，等待用户决定。

**All-14 / All-15 是完整数据集上的表现，不等于对未见场景的泛化。**

## 2. 闭环设置（所有方法相同）
| 项 | 设置 |
|---|---|
| 模拟器 | 冻结的 M 协议 `exp3b m_t6_sim.EpisodeSim`：官方 settings 网格与邻居、合法性检测、无合法动作时 hover、带种子的观测、456 × 256 完美深度、量程 70 |
| 起点 | 每个场景 `settings.json camera.start_positions[0..4]`，与 MAGICIAN LMDB 的起点逐一相同 |
| 预算 | 起点观测 C(0) + 100 步，共 101 帧 |
| MAGICIAN | 官方 e697009 代码 + 官方权重；Beam 10 × 10（MAG-B，主基线）；Greedy（MAG-G，可选行）；种子 `exp5-T2|MAG|<scene>_s<k>|<t>` |
| 我们的模型 | 各自的历史配置，**不做修改**：Greedy-NV、K = 4、10 步 Euler、R10、S10 DistP1 评分；推理密度按模型登记（B0 / V2O / E2S 为 S10，B2 / E2D / V3 / V4 / V5 为 D100，B1 为 D100 / D100）。启动链见 `MODEL_REGISTRY.yaml` |
| 场景参数 | 规划器在非 Dev-3 场景上沿用历史规则（`t24_v2_out_planner.py:46` 运行时 `z0_of`；`em2_windows.json` 冻结窗口表），**没有新增任何参数** |

**比较性质**：本表是**系统级比较**，比较的是完整的探索系统。各模型的训练数据、预算、密度和选择规则并不一致（`CHECKPOINT_AUDIT.md`），**不能当作纯架构消融**。只改动一个因素的配对见 §6。

## 3. 覆盖评价（两套，冻结）
- **主口径：MAGICIAN 官方覆盖（`coverage_official`）**
  - 模拟器内由 MAGICIAN `compute_partial_point_cloud(gathering_factor × 2 = 10%)` 采样点 → `covered.fill_cells`（含每格容量上限）→ `scene_coverage(eps = 1.0)` / `visibility_ratio` 计算。
  - 每个 episode 都由 `t25_out_cov.py` 重放核对，与存档曲线逐位一致。
- **辅口径：单调 10% 覆盖（`coverage_mono`）**
  - 同样的 GT 点、eps 和分母，但所有采样点都保留（无容量上限），所以单调不减。
  - 历史报告（Task 2 / Density A/B 等）用的是这一口径。
- **不混用**：所有方法都用同一套主口径进入主表；辅口径另出一份完整表（`results/*_mono.csv`）。
- **三类数值明确区分**：
  1. MAGICIAN 论文公布值（`SCENE_REGISTRY.yaml` 的 `magician_paper` 字段，抄自 Magician.pdf 第 12 页表 6 / 7；15 场景均值 AUC 0.721 / Final 0.919），**只作参照，不填入本地结果**；
  2. MAGICIAN 上游循环在本地运行的 LMDB 轨迹（14 场景，0.711 / 0.906，未设种子），只作参照；
  3. 本地冻结协议下的 MAG-B 结果，即本表数值。

## 4. 指标（冻结）
曲线 C(t)，t = 0…100（C(0) 为起点观测后的覆盖）。

| 指标 | 定义 |
|---|---|
| Cov@k | C(k)，k ∈ {10, 20, 40, 50, 60, 80, 100} |
| Final | C(100)，与 Cov@100 恒等（自动检查） |
| AUC@k | trapz(C(0..k), dx = 1) / k，k 取同上 7 个值 |

- AUC@100 即历史 AUC 定义（`t25_out_cov.py:76-77`，`task2/protocol.md:81`）。
- MAGICIAN 自带的 `evaluation_lmdb.py` 用不归一化的 trapz(C(0..99))，丢掉了 C(100)。换算后与本定义相差约 0.002（`EVALUATION_AUDIT.md`），本表**不采用**。
- **重点指标**：AUC@20（早期探索效率）、AUC@100（整体效率）、Final（完整度）。

## 5. 汇总与统计
- **四个层级**：
  1. episode；
  2. 场景（5 个起点的均值与标准差；**只有 5 个起点都完整才计入**）；
  3. 划分（Train / Dev / Test / Other，场景等权）；
  4. All-14 / All-15。
- **完整性**：某组场景不完整时输出 `<组>-partialNofM`，不计入完整汇总。**缺失的 episode 不用其他起点或均值填补。**
- **不确定性**：场景间标准差；场景级 bootstrap 95% CI（10,000 次，seed 0）。
- **与 MAG-B 的配对差**：每个场景先对同一起点的差取均值，再在场景层面 bootstrap，并报告胜 / 负场景数。指标为 ΔAUC@20、ΔAUC@50、ΔAUC@100、ΔCov@20、ΔFinal。
- 只有 CI 不含 0 时才称“显著”。

## 6. 只改动一个因素的配对（用于受控解读）
| 配对 | 改动的因素 |
|---|---|
| B0 ↔ B2 ↔ B1 | 同一权重，只改 P1 推理 / 评分密度 |
| MAG-G ↔ MAG-B | 只改 beam |
| V4 ↔ V4C | 只多训练 12k |
| V4C ↔ V5 | 同父模型、同预算，换解码器与多表面监督（复合干预） |
| V2O ↔ B0 | 多了混合微调 17.5k |
| B2 ↔ E2D | 多了 D100 混合微调 5k |

其他行之间的比较都是系统级比较。

## 7. 复用与补跑
- **复用**：历史 Dev-3 + Test-3 共 30 个 episode / 模型，13 个模型合计 390 个。审计结论为：
  - episode 完整；
  - 官方曲线重放逐位一致；
  - planner 记录的 checkpoint sha 与 P1 模式都和登记一致。
- **补跑**：其余可运行的 8 个场景（Train-6 + eiffel + sestino_museum）× 5 个起点。使用与历史完全相同的启动链（`scripts/t1_queue.sh`），输出到 `work_dirs/cvpr_results/table1/g3`。
- **回归**：每个主模型用当前代码重跑一个历史 episode（barts_s0），要求执行动作和官方曲线逐位一致（`scripts/replay_check.sh`，结果见 `EVALUATION_AUDIT.md` 附录与 `reports/BASELINE_REPRODUCTION.md`）。
- **作为 episode 结果的唯一来源**：`work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json`。所有表格和图表都由 `scripts/aggregate.py` 与 `plot.py` 从这些记录自动生成。

## 8. 偏差记录
| 日期 | 偏差 | 原因 | 影响 |
|---|---|---|---|
| — | — | — | — |
