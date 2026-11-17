# Fig. 6 v4 候选素材（第 20 步观测到的表面，2026-11-16）

**目的**：用户希望 Figure 6 改用没有任何方法训练过的场景。初步设想是左边 3–4 个未见室外场景，右边 2 个未见室内楼层。本目录提供**全部候选**供挑选：6 个 Unseen-6 场景的所有可用起点，以及 HO-12 的全部 24 条室内 episode。V3 的 demo 原样保留在 `v3_kept/`。

画法与 V3 相同：
- 第 1 行是完整 GT；
- 第 2、3 行只画出该方法在第 0–20 步真实观测到的表面，按真实纹理着色，不涉及任何预测或重建；
- 同一列三行共用一个相机；
- 面板下方标注 Cov@20：室外用 official coverage，室内用 C_obs，与 Table 1 v4 是同一批记录。
- 面板高度与 V3 主图完全相同（1.508 in），任一面板都可以直接放进最终的 Figure 6。

## 文件

| 路径 | 内容 |
|---|---|
| `ALL_unseen6_outdoor.png` | **总览**：6 个未见场景的对比图上下拼接，最后附 barts / sestino 的剖切版，半分辨率 |
| `ALL_HO12_indoor.png` | **总览**：24 条室内 episode |
| `sheets/outdoor_<scene>.{png,pdf}` | 每个场景一张，每列一个起点，列标题为"场景 / 起点 · 方位°/仰角°" |
| `sheets/indoor_{1,2,3}.{png,pdf}` | 室内，按房屋分组：2azQ + QUCT、TbHJ + Z6MF、oLBM |
| `interior_cull/` | barts 和 sestino_museum 的剖切版（见下文） |
| `viewpoints/<scene>_viewpoints.png` | 每个室外场景在 24 个候选视角下的 GT 渲染，红字表示被哪些起点自动选中 |
| `panels/<case>__{gt,mag,ours}.png` | 单面板，400 ppi |
| `views/<case>.json` | 视角网格的全部得分、选定的视角、相机参数 |
| `checks/` | 每列的渲染统计 |
| `candidates_cov20.{csv,md}` | 每条候选的 Cov@20（两种方法）、差值 Δ、在场景内和全体中的排名 |
| `v3_kept/` | V3 主图与面板（Pisa、Colosseum、Redeemer 及两个室内列），未改动 |

## 候选范围与校验
- **室外**：Unseen-6 × 全部官方起点，排除起点位于网格内部的 alhambra s2、neuschwanstein s2 / s4（之前已确认不能用于作图），共 27 条。
- **室内**：HO-12 的 12 个楼层 × 2 个起点，共 24 条，与 Table 1 v4 的室内列相同。
- **掩码**：用 V3 已验证的代码（`fig6/fig6_masks.py`，未改动）。51 条全部通过硬性检查：
  - 由存储的服务记录重算的第 0–20 步覆盖曲线，与 Table 1 记录逐位相同；
  - 两种方法起点相同，C[0] 相同；
  - Ours 的权重 sha 与冻结的 V3-T0 一致；
  - Cov@20 与候选清单一致。

  与 V3 重合的 4 条（bridge s0、neuschwanstein s0 和两个室内列），掩码与 V3 逐位相同。
- **新导出的 GT 点**：eiffel 和 sestino_museum 的起点 1–4 原来没有 GT 评估点。用 EXP5 冻结代码（`a1_gtpoints_out.py` 的同一路径）在 GPU 上导出，见 `scripts/fig6_v4/f6v4_gtpoints.py`。上面的逐位曲线检查同时验证了这批 GT 点。

## 视角
- **室外**：在 8 个方位 × 3 个仰角（20° / 35° / 50°）共 24 个视角中自动选取。评分规则与 V3 相同：只被一种方法观测到的可见像素越多越好，两种方法对称计算。平分时优先 35°，再取方位最小者。V3 的规则（只看 35°）选出的视角也记录在 `views/*.json` 的 `v3_rule_view` 字段。
  - **结果**：自动规则总是选 20° 或 50°，从不选 35°，因为极端仰角下差异像素更多。
- **改视角**：在 `viewpoints/` 里挑一个标签，例如 `az 0 / el 50`，告诉我"场景 + 起点 + 视角"即可，单列重渲染约 30 秒。
- **室内**：沿用 V3 的规则（60° 俯视，剔除背面，去掉 2 m 以上的表面，方位取两种方法观测像素并集最多者）。

## barts / sestino_museum 的剖切版（`interior_cull/`）
这两个 Macarons++ 场景其实是室内：barts 是一个大厅，sestino 是一座博物馆。按室外方式渲染时不剔除背面，近处的墙和屋顶会把房间挡住。更麻烦的是，没被观测到的外墙反而让里面的楼梯、展品露出来，使 MAGICIAN 那一行看起来比 GT 还"丰富"，容易误导读者。

`interior_cull/` 改用与 MP3D 室内相同的方式渲染：剔除背面，形成剖切视角。两种方法完全相同处理，视角规则不变。

| 场景 | 效果 | 建议 |
|---|---|---|
| barts | 明显好读，例如 s3：MAGICIAN 漏掉整面大壁画和天花板 | 用剖切版 |
| sestino | 屋顶椽子仍占大半画面 | 可在 `interior_cull/viewpoints/` 中换成 50° 仰角 |

原来不剖切的版本保留在 `sheets/`。图注需说明这两个场景"按室内方式渲染（剔除背面）"。

## 与 V3 渲染代码的一处偏离（已记录）
V2 的分块相机自检会抽样约 2000 个网格点，把每个分块相机的投影加上偏移后与整帧投影比较，阈值为 0.05 渲染像素。

在 indoor_TbHJrupSAjP_L2_S86 上，对全部抽样点算出的偏差是 0.117 像素；只看落在分块内的点，偏差为 0.0002 像素。前者来自分块外点在 float32 下的外推误差，而分块只会光栅化落在其内部的点。

因此 V4 改为对分块内（±2 像素）的抽样点检查，阈值不变；全部点的偏差另行记录。渲染结果本身不受影响。

## 选图时要注意
- **选图依据要写进图注。** 室外 27 条的 Δ 中位数为 **+0.178**，Ours 赢 24/27；室内 24 条的中位数为 **+0.038**，Ours 赢 20/24。挑差距最大的例子没有问题，但图注需写明，例如 "examples; median Δ over the 27 unseen outdoor episodes is +0.178"。
- **alhambra s1**：MAGICIAN 的 Cov@20 只有 0.01，几乎没有移动，属于异常情况。不建议放进正文；若要放，图注里要说明。
- **Ours 更差的例子**：sestino s0（−0.230）、alhambra s3（−0.247）、bridge s1（−0.001）。室内有 QUCTc6BB5sX L0 S31、L1 S34，TbHJrupSAjP L0 S21、L1 S114。全部在 `candidates_cov20.md` 中，没有删除。
- **室内差距普遍小**，与 V3 的观察一致。Δ 最大的 3 条：oLBMNvg9in8 L3 S8 和 L2 S95（都是 +0.085）、TbHJrupSAjP L2 S86（+0.084），都属于完全留出的 HO 楼层。
- **划分说明**：Unseen-6 中，alhambra / neuschwanstein / bridge 是我们的 Test-3，eiffel / barts / sestino 是 Dev，用于选模参考。所有方法都没有在这 6 个场景上训练。

## 脚本（`experiments/cvpr_results/exp9_ablation_qualitative/scripts/fig6_v4/`）

| 脚本 | 作用 | 环境 |
|---|---|---|
| `f6v4_gtpoints.py` | 导出缺失的 GT 点 | magician，GPU |
| `f6v4_masks.py` | 计算掩码并做硬性检查 | seen2scene，CPU |
| `f6v4_prep.py` | 网格预处理（V2 代码不改，6 个分片并行） | magician，CPU |
| `f6v4_render.py --group outdoorA\|outdoorB\|interior\|indoor [--redraw]` | 渲染与排版 | magician，1 GPU |
| `f6v4_table.py` | 生成候选表 | CPU |
| `f6v4_overview.py` | 拼接总览图 | CPU |

工作文件在 `work_dirs/cvpr_results/exp9_ablation_qualitative/fig6_v4/`，包括掩码 npz、网格和日志。

## git
只提交文本：脚本、本 README、`candidates_cov20.*`、`views/*.json`。图片约 360 MB，只保存在磁盘上，与 V3 素材的做法相同。需要时运行上面的脚本即可逐字节重新生成。
