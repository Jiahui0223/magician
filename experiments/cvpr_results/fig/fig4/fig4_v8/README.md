# Fig. 4 v8（2026-11-16）

在 v7 final 的基础上**去掉第一个场景**（Neuschwanstein），只保留两个场景、共 4 行面板：

- **Colosseum s3 t0**：主行 + ×3 放大行；
- **MP3D QUCTc6BB5sX L1 S34 t40**（"MP3D – Scene3"）：主行 + ×3 放大行。

| 文件 | 内容 |
|---|---|
| `fig4_v8.{pdf,png,svg}` | 6.875 × 2.13 in，版式 = v5-3b（连线与白框同样式，一行图例） |

## 与 v5-3b / v7 final 的关系
- 两行的数据、放大窗口、配色、色尺与版式参数都与 v5-3b 的第二、三行相同。窗口由脚本重算，并断言等于 `fig4_v5/zoom_windows.csv`：Colosseum (21, 5)，MP3D (34, 69)。
- 整图高度变了，matplotlib 按图高比例定位，所以光栅化后的像素与 v5-3b 不是逐位相同，只有亚像素级的位置差异，内容完全一致。
- 单面板不需要新出，直接用 `../fig4_v5/panels/row2_colosseum/` 和 `../fig4_v5/panels/row3_QUCTc6BB5sX_L1/`。

## 图注
沿用 `../fig4_v7_final/README.md` 中的英文图注，删去最后一句关于 Neuschwanstein 的说明。

放大窗口的描述"…cover at least 30 % of the window"对这两行同样成立：它们的窗口在 v6 规则下不变。

脚本：`experiments/cvpr_results/exp9_ablation_qualitative/scripts/fig4/fig4_v8.py`（CPU，seen2scene env）。
