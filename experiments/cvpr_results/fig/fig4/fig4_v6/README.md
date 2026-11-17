# Fig. 4 v6：第一行（Bannerman）的替换候选（2026-11-16）

## 问题
v5 第一行是 Bannerman s3 t0。它的放大框里 GT **全是天空**（GT 命中 0%，候选 RGB 放大后是一片空），但 4 个生成中有 3 个在那里预测出了墙（命中 32% / 85% / 37% / 5%）。读者会理解为"模型凭空生成了不存在的表面"。v5 README 当时已把这一点记为问题，并建议的修复是：要求放大框内至少 30% 的射线有真实表面。

## 规则（v6 = v5 + 一个条件）
放大框的选法与 v5 相同：
- 窗口大小为 21 × 38 个打分单元（每边 1/3）；
- 窗口内未知射线的 GT 命中必须全部在场景 ROI 盒内；
- 得分 = 4 个生成"命中 / 不命中"不一致的未知射线数，取最高者。

v6 只**加一个条件**：窗口内至少 **30%** 的射线有真实表面（GT 在传感器范围内命中）。这正是 v5 README 预先写下的阈值，没有在看过候选之后调整。

- v5 第 2、3 行（Colosseum、MP3D Scene3）在 v6 规则下的窗口与 v5 相同（脚本断言），所以只有第一行受影响。
- 候选池仍是 v4 / v5 的 69 个合格室外状态（`manifests/fig4_selection_outdoor_maxp90_inbox.csv`），合格条件不变。
- 排序按 v6 窗口得分从高到低；排除第 2 行的 Colosseum s3 t0；同一（场景, 起点）只取一个；取前 6 名。另外规定每个未见场景（alhambra、neuschwanstein）至少入选一个，这次两个都已在前 6 名中。
- Bannerman s3 t0 本身也按 v6 规则重选窗口，作为保留原行的选项。

## 候选

| 编号 | 状态 | v6 窗口得分（不一致射线数） | 窗口 GT 表面占比 | 说明 |
|---|---|---|---|---|
| #1 | Bannerman s2 t0 | 591 | 86% | GT 是远处表面，生成多为近处植被 |
| #2 | **Bannerman s0 t10** | 521 | 82% | 一面带窗洞的残墙，4 个生成对"哪里有洞"各有说法；**如想保留 Bannerman，选这条** |
| #3 | Pantheon s4 t0 | 515 | 88% | 当前视角大半是天空 |
| #4 | **Alhambra s4 t0** | 512 | 83% | **未见场景**；放大框里是真实的塔楼和树，生成对塔楼的范围和远近不一致；当前视角内容丰富。**推荐** |
| #5 | Colosseum s0 t0 | 510 | 100% | 有已知 / 未知分界线，但与第 2 行同一场景 |
| #6 | Neuschwanstein s0 t0 | 418 | 34% | 未见场景，但当前视角几乎是空的 |
| 保留原行 | Bannerman s3 t0（v6 窗口） | 320（v5 窗口为 643） | 31% | 换到有墙的位置，分歧减半 |

完整排名见 `ranking_v6.csv`（69 个状态，每个状态的 v5 / v6 窗口与统计），候选清单见 `candidates.json`。

## 文件

| 文件 | 内容 |
|---|---|
| `candidates_sheet_1.{png,pdf}` | 候选 #1–#3 |
| `candidates_sheet_2.{png,pdf}` | 候选 #4–#6 |
| `candidates_sheet_3.{png,pdf}` | 保留原行的选项 |

画法与 v5 完全相同：配色、×3 放大、图例都直接调用 `fig4_v5.composite`。

## 数据与校验
- **数据来源：** 深度和 4 个生成都来自原有的状态文件，没有重新推理。
- **彩色图：** 用 `fig4_v4_render.run_case` 原样重新光栅化，输出写到 `work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v6/render/`。v4 的 C1 / A1–A4 守卫全部通过：当前与候选视角的掩码一致率为 1.0，深度逐位相同。
- **与 v4 对照：** Bannerman s3 t0 的候选 RGB 与 v4 相同，差异像素为 0。

## 注意
- 选中的行仍然是"分歧大"的状态，不是典型状态，图注须保留 "Rows are chosen to show large disagreement and are not typical"。
- 放大框的说明改为："the 1/3 × 1/3 window with the most disagreeing unknown rays among windows whose real surfaces lie inside the modelled region and cover at least 30 % of the window"。
- 换行之后，第一行的色尺、面板和放大窗口都要整体替换。确定选哪一条后，我按 v5-2 版式重出整张图。

脚本：`experiments/cvpr_results/exp9_ablation_qualitative/scripts/fig4/fig4_v6.py`，运行方式为 `scan` / `render` / `sheet` 三个阶段。
