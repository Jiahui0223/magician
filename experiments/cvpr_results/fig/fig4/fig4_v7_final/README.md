# Fig. 4 v7 final（2026-11-16）

用户定稿：第一行换成 **Neuschwanstein s3 t0**；第二、三行（Colosseum s3 t0、MP3D QUCTc6BB5sX L1 S34 t40）以及版式都沿用 **v5-3b**。

| 文件 | 内容 |
|---|---|
| `fig4_v7_final.{pdf,png,svg}` | **定稿**，6.875 × 3.07 in，版式 = v5-3b（连线与白框同样式，一行图例） |
| `panels/row1_neuschwanstein/` | 第一行的单面板，与 v5 的 `panels/<row>/` 结构相同：单面板、`box/`、`zoom/`、`hires_912x512/`，以及三种色尺 |
| `colorbars/row1_neuschwanstein/` | 第一行色尺的原件（v4 的三种画法，参数相同） |
| `zoom_windows.csv` | 三行的放大窗口与统计 |

第二、三行的单面板没有变，直接使用 `../fig4_v5/panels/row2_colosseum/`、`../fig4_v5/panels/row3_QUCTc6BB5sX_L1/`。

## 校验
- 与 `fig4_v5/fig4_v5-3b.png` 逐像素比较：第二、三行和图例的像素**完全相同**，差异只在第一行。
- 第一行的数据：
  - 深度和 4 个生成来自 v7 扩充池的状态文件，与旧池同一份扫描结果，没有重新推理。
  - 彩色图来自 `fig4_v4_render.run_case`，v4 的 C1 / A1–A4 守卫全部通过。
- 第一行的放大窗口按 v6 规则选取（v5 规则 + 窗口内至少 30% 是真实表面），由脚本重算并断言与 `ranking_v7.csv` 一致。第二、三行的 v5 窗口在 v6 规则下不变（同样断言）。
- 第一行色尺：19–70 场景单位。粘贴版字号 6.75 pt，与原三行拟合出的字号相同。

## 第一行：Neuschwanstein s3 t0
- 属于 Unseen-6，所有方法都没在这个场景上训练。
- 候选视角中 59% 的射线未知，所以候选 RGB 上同时有已知（不着色）和未知（橙色）区域。
- 未知射线上 4 个生成"命中 / 不命中"不一致的比例为 7.4%。
- 放大窗口：起点 (38, 53)，21 × 38 个打分单元；窗口内 100% 有真实表面；不一致的未知射线 156 条。

**如实说明**：放大处 4 个生成给出的表面都比真实表面**更远**（GT 偏黄绿，生成偏蓝），彼此之间在左侧有"命中 / 不命中"的分歧。也就是说，这一行展示的不只是假设之间的分歧，也有共同的深度偏差。图注不要写成"至少一个生成与 GT 一致"。

## 建议图注（英文，v5-3b 无第二行图例）
> **Figure 4. Multi-hypothesis future observations.** Each row is one planner state of Seen2Map. *Current view:* the real image at the agent's pose at step *t*. *Candidate view:* the real image of the view the planner executed next; tinted pixels are rays still unknown in the partial map, untinted pixels are rays the map already determines. *GT surface depth:* the depth actually observed from the candidate view. *Generated 1–4:* the first-hit depth decoded from each of the *K* = 4 belief samples the planner scored for this candidate; on known rays all four coincide with the partial map. *Surface-hit variation:* white = known rays; light orange = unknown rays on which all four generations agree; pink to dark magenta = 1, 2 or 3 of the 4 generations predict a hit. The second line of each row enlarges (×3) the boxed window, i.e. the 1/3 × 1/3 window with the most disagreeing unknown rays among windows whose real surfaces lie inside the modelled region and cover at least 30 % of the window. Grey: no surface within the sensor range or no predicted hit. Rows are chosen to show large disagreement and are not typical. Outdoor depth is in Macarons++ scene units, indoor depth in metres; Neuschwanstein is one of the six scenes no method was trained on.

脚本：`experiments/cvpr_results/exp9_ablation_qualitative/scripts/fig4/fig4_v7_final.py`（CPU，seen2scene env，`CUDA_VISIBLE_DEVICES=""`）。它直接调用 `fig4_v5_layouts.composite` 与 `fig4_v5.export_panels`，这两个文件本身没有改动。
