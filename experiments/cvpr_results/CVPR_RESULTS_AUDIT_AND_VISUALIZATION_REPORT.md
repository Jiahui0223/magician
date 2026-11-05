# CVPR Results：协议审计与论文图表报告（2026-11-04）

**本轮做了什么**（全部为非破坏性工作）：
- **Task A**：MAGICIAN 评估与训练协议审计，见 `protocol_audit/`；
- **Task B**：论文风格的表格与图，见 `paper_tables/`；
- **Task C**：把两者接入现有 Table 1 流水线。

没有修改任何历史数值、Table 1 文件或已有审计文档，没有训练，没有新的闭环实验。

## 数据流（Task C）
```
官方评估器（冻结模拟器，MAGICIAN 官方代码与权重）
  → 规范 episode 记录：work_dirs/cvpr_results/table1/episodes/（官方覆盖曲线、ckpt_sha256、来源文件）
  → table1/scripts/validate.py + aggregate.py（已验证的场景 / 划分 / All-14 汇总）
  → paper_tables/scripts/make_tables.py + make_figures.py（展示层，只读）
```
- **刷新**：`bash paper_tables/refresh.sh`。
- **新模型**：
  1. 仍然只在 Table 1 注册并运行；
  2. 完成后，把 id 加进 `paper_tables/config.yaml`；
  3. 运行刷新，论文表格与图会自动更新。
- **不完整的模型**：在表中显示 `--` / `partial`，不进入完整均值，也不会出现在图中（记入 `MANIFEST.json`）。
- **自动检查**：每次生成运行 166 项表格检查和 371 项图检查，输入文件 sha256 与 git HEAD 记入 `paper_tables/MANIFEST.json`。
- **审计结论不覆盖结果**：所有更正都写在 `protocol_audit/` 中，原结果与原文档都不改。

## 七个问题的回答
### 1. 是否与 MAGICIAN 原论文完全采用相同的五个初始 Camera Pose？
**与官方发布的起点完全相同（PASS）；与论文实际使用的起点是否相同，无法验证（UNKNOWN）。**
- **一致的部分**：
  - 起点 = 每个场景 `settings.json camera.start_positions[0..4]`，官方测试器（`magician_planning.py:834-835`）与我们的模拟器使用的是同一组；
  - 26 个模型的 1655 条 episode，第 0 步位姿全部等于对应起点；
  - 与 MAGICIAN 上游循环的复现运行逐个比较，70 / 70 相同（位置差 ≤ 4.5e-5，朝向差 0）；
  - 没有重复、缺失、错位，也没有挑选起点。
- **无法验证的部分**：论文只说每个场景用了“相同的随机初始位姿”，没有公布位姿或轨迹。官方 README 称 settings 中的起点就是“随机采样的 5 个起点”，所以“相同”是合理推断，但无法证实。
- 明细见 `protocol_audit/INITIAL_POSES_COMPARISON.csv`。

### 2. Coverage、AUC 和 Final 的公式是否与其一致？
- **Coverage：与官方发布代码一致（PASS），与论文文字描述不一致（UNRESOLVED）。**
  - **与发布代码一致**：我们导入的覆盖相关函数与官方 e697009 逐字相同，包括 mesh 采样 10 万 GT 点、每格上限 1000、10% 观测点、ε = 1.0、除以 `visibility_ratio`。
  - **与论文描述不同**：论文附录 B.1 说它的 GT 点云是“从所有可达视角渲染深度图再反投影”得到的，与发布的代码不同。论文的 GT 没有公开，无法重算。
- **AUC：** 我们用 AUC@k = trapz(C[0..k]) / k，两端点都计入、权重 1/2，已用 5 条合成曲线做单元测试。
  - 论文没有给出 AUC 的归一化方式；
  - 官方 `evaluation_lmdb.py` 用不归一化的 trapz(C[0..99])；
  - 各种写法在同一数据上相差 ≤ 0.009。
- **Final：** 取 C[100]。官方覆盖曲线可能下降，取最大值只会高 0.002–0.005。
- **平均方式：** 先对场景内 5 个起点平均，再对场景等权平均；每个场景都恰好 5 条，因此与 70 条直接平均完全相同。

### 3. MAGICIAN Paper Reported 与 Reproduced 的差距是多少，能否解释？
| | AUC | Final |
|---|---|---|
| Paper，All-15（报告值） | 0.721 | 0.919 |
| Paper，相同 All-14（由逐场景值重算） | 0.7214 | 0.9211 |
| Reproduced，冻结协议（MAG-B） | 0.7108 | 0.9063 |
| Reproduced，上游循环原样运行（LMDB） | 0.7111 | 0.9060 |
| 差（MAG-B − Paper，All-14） | −0.010 [−0.025, +0.002]（不显著） | **−0.015 [−0.032, −0.001]（显著）** |

- **两份独立复现彼此一致。**
- **逐场景**：AUC 14/14、Final 11/14 个场景的差值在起点噪声的 2 倍标准误以内；与论文的相关系数分别为 0.96 / 0.94。
- **差距集中在**：pantheon（Final −0.102）、neuschwanstein（−0.051）、sestino_museum（−0.046）、bannerman（−0.021）。其中 pantheon 起点 2、neuschwanstein 起点 2 和 4 对所有方法都难，两次复现中都出现。
- **已排除**：AUC 写法（其他写法都更低）、取最大值（只差 0.002）。
- **无法确认**：GT 点云定义、论文的起点、论文时期的代码版本（当时官方代码连 sofia_church 都无法加载）。
- **结论：UNRESOLVED。不宣布已对齐。**
- **另外**：论文表 7 的 FisherRF 平均 Final（0.786）与它 15 行的均值（0.772）不一致。

### 4. MAGICIAN 的训练数据与我们相同吗？是否可能存在数据泄漏或训练场景重叠？
**不相同；双方都有训练场景重叠，需要披露。没有发现评估数据泄漏到训练标签的证据。**

| | MAGICIAN（即 MACARONS 权重） | 我们（以 V3-D1 为例） |
|---|---|---|
| 预训练 | ShapeNet（论文；权重对应关系为推断） | 无外部预训练 |
| 场景训练 | 8 个 Macarons++ 场景（dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum），RGB 自监督 | Train-6 去掉 S-c（138 条轨迹，GT 渲染标签），另加 MP3D 228 条 |
| 与 14 个评估场景的重叠 | 8 个，其中 pantheon、colosseum 在我们的 Dev 中 | 6 个（Train-6） |
| 与评估起点的重叠 | 训练起点在全网格随机抽，恰好重合的期望约 0.06 次；途经评估起点无法核实 | 每个 Train-6 场景有 3 个官方起点是训练轨迹的起点；起点 4 与 Dev / Test 没有 |
| 用评估场景做选择 | val（alhambra、neuschwanstein）的实际用途未知；发布代码中没有使用 | D1 与 Y50K 按 Dev-5 选出；Test-3 不是盲测 |

- **双方都没训练过的场景**：eiffel、bridge、barts、sestino_museum（alhambra、neuschwanstein 也是，但 MACARONS 一方的 val 用途未知）。
- **我们的训练轨迹中**，有 9 条由 MAGICIAN（MACARONS 权重）驱动，另有用 GT 选动作的 CG-ε 轨迹。它们只决定相机走到哪里，标签都来自 GT 网格。
- **对已有审计的 4 处更正**见 `protocol_audit/MAGICIAN_TRAINING_AUDIT.md` §D。

### 5. 哪些模型和数值可以直接放入 CVPR 主表？
- **可以放（复现组，同一协议）**：
  - **主表**：MAGICIAN（复现，beam 10 × 10）、B2、V3-D1；
  - **补充表**：MAGICIAN-Greedy、B0、V3-T1（H0）、V3-YBEST-H3。
  - 指标：All-14 的 AUC@20 / AUC@100 / Final 及逐场景值。
- **主要结论**（V3-D1 vs MAGICIAN 复现）：

  | 分组 | AUC@20 | AUC@100 | Final |
  |---|---|---|---|
  | All-14 | **+0.118 ✱**（13/1） | **+0.054 ✱**（12/2） | +0.004（不显著） |
  | 未见场景（Dev + Test 共 8 个） | **+0.121 ✱**（8/0） | **+0.070 ✱**（7/1） | +0.015（不显著） |
  | Train-6 | — | — | −0.011 ✱（1/5，MAGICIAN 略高） |

  - Train-6 上 MAGICIAN 的 Final 略高，应如实写出。
  - V3-D1 与 V3-YBEST-H3 之间没有显著差异（见 Experiment E）。
- **只能作参照**：论文公布的 Random Walk / SCONE / MACARONS / FisherRF / MAGICIAN。已在表中单独成组、标为 reported，不与我们的结果做显著性或“最优”比较。
- **不能放**：All-15；任何把论文公布值和我们的复现值当成同一协议的比较。

### 6. 哪些表格和图适合正文，哪些放 Supplementary？
| 产物 | 建议 |
|---|---|
| `paper_tables/tables/table_auc_full_scenes.{tex,md,csv}` 与对应图 | **正文**：逐场景 AUC，左侧论文值作参照，右侧为同一协议下的 MAGICIAN（复现）/ B2 / V3-D1 |
| `paper_tables/tables/table_final_full_scenes.*` | **正文或附录**（篇幅紧张时放附录） |
| `paper_tables/figures/coverage_curves_3panel.*` | **正文**：All-14 / Dev-5 / Test-3 的平均覆盖曲线，带场景级 95% CI |
| `paper_tables/figures/improvement_over_magician.*` | **正文**：逐场景的 ΔAUC@20 / ΔAUC@100 / ΔFinal，同时显示提升与退化的场景 |
| `paper_tables/tables/*_supp.*` | **Supplementary**：加入 MAGICIAN-Greedy、B0、V3-T1（H0）、V3-YBEST-H3 |
| `paper_tables/figures/coverage_curves_{all14,dev5,test3}.*` | 视版面放附录 |
| `paper_tables/figures/exploratory/radar_all14.*` | **不进论文**（只作探索；所有轴统一 0–1） |
| `protocol_audit/*` | 内部审计，可作为 Supplementary 中 “Evaluation protocol” 一节的依据 |

### 7. 是否还存在尚未解决的协议差异？
**存在，但都已记录并标明影响范围：**
1. **GT 点云**：论文用渲染的可见点，发布的代码用 mesh 采样（UNRESOLVED）。只影响与论文公布值的比较，不影响复现组内部的比较。
2. **论文实际用的起点**：没有公布（UNKNOWN）。同上，只影响与论文公布值的比较。
3. **sofia_church**：冻结评估器无法加载，所以只有 All-14。修复需要 protocol v1.1，属于协议变更，等待你决定。
4. **种子**：官方不设种子，我们设种子。只影响评估噪声。
5. **无合法动作时的处理**：官方结束轨迹，我们原地停留一步。主表模型从未触发。
6. **训练侧未知项**：U1–U9，例如论文中 SCONE / FisherRF 列实际用的权重。
7. **复现缺口**：SCONE、MACARONS、FisherRF 没有在我们的协议下重新运行，只能引用论文值。若要放进同一公平比较组，需要单独补跑（发布的测试器有已知问题，需先修复）。

## 产物索引
- **审计**：`protocol_audit/` 下的
  - `MAGICIAN_PROTOCOL_AUDIT.md`（总览）
  - `MAGICIAN_EVAL_AUDIT.md`
  - `AUC_COVERAGE_FORMULA_AUDIT.md`
  - `MAGICIAN_TRAINING_AUDIT.md`
  - `TRAINING_DATA_COMPARISON.md`
  - `INITIAL_POSES_COMPARISON.csv`
  - `SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv`
  - `AUDIT_CHECKLIST.md`
  - `FINAL_AUDIT_REPORT.md`
  - `scripts/`、`results/`
- **论文表格与图**：`paper_tables/` 下的
  - `config.yaml`、`refresh.sh`、`README.md`（含图注与追溯方法）、`MANIFEST.json`
  - `tables/`（md / tex / csv）
  - `figures/`（png 300 dpi / pdf，以及 `exploratory/`）
- **LaTeX**：生成的 TeX 使用 `booktabs`。本机没有 LaTeX，无法实际编译；已用脚本检查每行列数与括号配平。正式放进论文前，请在你的 CVPR 模板中编译一次确认。
