# EXP7 — UAVScenes 可行性报告（Step 5.1–5.2）

日期：2026-11-10 · 执行节点：gpu114（**仅 CPU**，`CUDA_VISIBLE_DEVICES=""`）· 未提交任何 GPU / Slurm 作业 · 未训练、未微调、未运行模型前向。
本文所有数字均来自下文列出的已下载文件或已运行脚本的输出（JSON/PNG 路径见 §8）。

## 0. 结论

| 判定 | 结果 | 一句话理由 |
|---|---|---|
| **GO-DEPTH** | **GO（数据层面已满足）** | AMvalley mesh 为公制 ENU、可载入、无内部孔洞；相机位姿与 RTK、LiDAR、照片三方独立对齐（RTK 水平偏差中位 0.34 m；LiDAR 距离残差中位 0.29–0.91 m @ 87–135 m）；用 MAGICIAN 相机模型 + PyTorch3D 渲染出的深度与 embree 光线投射一致到 0.2 mm 以内。冻结 T0 的前向**尚未运行**（需要 GPU，本任务禁止）。 |
| **GO-NBV** | **暂不 GO：可构建，没有数据层面的阻塞** | 自由空间相机网格、线段–mesh 合法性检测、GT 表面采样、可见性都已在 CPU 原型中跑通（1,263 个合法位置，构成一个连通图；ROI 内 99.86% 的 GT 点至少能被一个合法位置看到）。但闭环 adapter、尺度/切块协议、MAGICIAN 兼容性和 GPU smoke 都还没做，因此不能宣称 GO-NBV。 |
| **NO-GO** | 否 | mesh、坐标、合法候选都没有发现硬缺陷。 |

**另一个科学风险（不是数据缺陷）**：AMvalley 是自然山谷地形。1 m 网格下 99.38% 的柱只有单层表面（2.5D），向上的面积占 81%，表面几乎从任何地方都可见。在这里做 NBV 接近 2D 面积覆盖，未必能考验 3D 推理，Random 或割草机式轨迹可能很有竞争力。含建筑的场景（AMtown、HKairport、HKisland）只读了文件头，没有下载检查。

---

## 1. 资产、来源与许可

- 官方 GitHub：https://github.com/sijieaaa/UAVScenes ，读取版本为 commit `909250d14a4460b2a62ea25f08f12ce6367427ac`（README、`calibration_results.py`、`cmap.py`、`LICENSE`）。
- HF 数据集：`sijieaaa/UAVScenes`，revision `04a224fd14b0b8fdfd2633e68f20471b85c611a5`（2025-08-06）。API 返回 `gated: false`、`private: false`，下载不需要登录，也没有需要接受的条款。
- **许可证存在矛盾**：GitHub 的 `LICENSE` 文件和两份 README 正文都写 **CC BY-NC-SA 4.0**（"academic use only"），但 HF 卡片 YAML 标签写的是 `cc-by-sa-4.0`。本研究按更严格的 **NC-SA** 执行：仅用于学术用途，派生数据（转换后的 mesh 等）以同协议共享，原始数据不进 git。许可证原文已复制到 `work_dirs/cvpr_results/exp7/downloads/uavscenes/LICENSE_CC-BY-NC-SA-4.0.txt`。
- HF 仓库实际文件树来自 `api/datasets/.../tree/main?recursive=1`，**只有 4 个整体 zip，没有分场景的文件**：

| 文件 | 大小 | LFS sha256 |
|---|---|---|
| `interval5_CAM_LIDAR.zip` | 28,682,115,865 B | `c964fca7…db9a36d9` |
| `terra_3dmap_pointcloud_mesh.zip` | 6,779,103,543 B | `a41e29a8…3eb2a643` |
| `interval5_CAM_label.zip` | 1,488,319,248 B | `461909f5…5c9fd5` |
| `interval5_LIDAR_label.zip` | 105,289,464 B | `14e5fbc5…39942` |

- **下载方式**：HF/xet CDN 支持 HTTP Range（已验证返回 206）。脚本 `http_zip.py` 只读取 zip 的中央目录和指定成员，不下载整个 zip。每个成员解压时由 `zipfile` 校验 CRC-32，sha256 写入 `raw/_provenance.tsv`。
  - 整次任务约传输 166 MB，其中 4 个中央目录约 36 MB，8 个场景的 PLY 文件头约 4.6 MB。
  - 所用 URL 形式为 `https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/04a224fd…/<zip>`。
- **场景选择**：README 的 4 个主场景中，AMvalley 的 `Mesh.ply` 最小（117.6 MB，压缩后 34.8 MB）。AMvalley03 是该场景帧数最少的 run（interval5 共 961 帧）。HKairport_GNSS 的 mesh 只比 AMvalley 小 0.9 MB，但它不在 4 个主场景里。
- 已下载的成员（均为 AMvalley 一个场景）：

| 成员 | 字节 | sha256（前 16 位） |
|---|---|---|
| `AMvalley/Mesh.ply` | 117,641,753 | `7e8e3be4f68a9ba0` |
| `AMvalley/terra_ply/metadata.xml` | 355 | `5a161c10ca6e7cf7` |
| `interval5_AMvalley03/sampleinfos_interpolated.json` | 6,312,630 | `c9a4467226b885ce` |
| `interval5_AMvalley03/rtk_positions_raw.csv` | 334,123 | `3b8411be579e3667` |
| `interval5_AMvalley03/posest_viz.png`、`rtk_positions_raw_viz.png` | 30,238 / 38,197 | `2f6b78b0…` / `5b2fd96b…` |
| 7 张 interval5 图像（2448×2048 JPEG） | 0.83–0.98 MB/张 | 见 `_provenance.tsv` |
| 4 帧 LiDAR（`image*_lidar*.txt`，18k–24k 点/帧） | 0.57–0.74 MB/帧 | 见 `_provenance.tsv` |

未下载：任何场景的 `cloud_merged.ply`、`terra_ply/Block*.ply` 及纹理、语义标签、其它场景和其它 run 的数据。

## 2. 清单（详表见 `uavscenes_inventory.csv`）

| 场景 | 检查深度 | Mesh.ply 大小 | #顶点 | #面 | interval5 帧数（各 run） |
|---|---|---|---|---|---|
| **AMvalley** | **完整** | 117.6 MB | 1,247,145 | 2,493,761 | 01: 2260 / 02: 1266 / **03: 961** |
| AMtown | 仅文件头 | 1,080.4 MB | 11,048,765 | 22,050,702 | 2589 / 1380 / 1120 |
| HKairport | 仅文件头 | 366.8 MB | 3,824,677 | 7,637,139 | 1440 / 786 / 605 |
| HKisland | 仅文件头 | 158.1 MB | 1,696,388 | 3,385,164 | 1317 / 775 / 599 |
| HKairport_GNSS | 仅文件头 | 116.7 MB | 1,268,985 | 2,539,712 | 1356 / 747 / 630 |
| HKairport_GNSS_Evening | 仅文件头 | 306.3 MB | 3,217,549 | 6,433,488 | 1415 |
| HKisland_GNSS | 仅文件头 | 167.7 MB | 1,786,523 | 3,565,061 | 1356 / 760 / 534 |
| HKisland_GNSS_Evening | 仅文件头 | 141.1 MB | 1,519,537 | 3,032,865 | 2230 |

所有 `Mesh.ply` 都是 ASCII PLY，由 CloudCompare 合并 Terra block 而成，**只有逐顶点 RGB，没有 UV 纹理**（UV 纹理在 `terra_ply/` 的 block 中）。

**AMvalley 详细参数**

- **坐标与单位**
  - `metadata.xml`：`SRS = ENU:39.946784145786722,44.898231974206659`，`SRSOrigin = 0,0,1368.923`。
  - 世界系为 ENU（z 向上），单位为米，原点海拔 1368.92 m。单位证据见 §3.1。
- **包围盒**：x [−804.4, 446.4]，y [−421.0, 247.2]，z [−165.2, −2.4] m，尺寸 1250.8 × 668.2 × 162.8 m。
  - 表面积 724,212 m²；俯视足迹（1 m 网格）527,418 m²。
  - 边长中位 0.58 m（p1 0.06 m，p99 2.84 m）。
- **内参**
  - `sampleinfos_interpolated.json` 的 `P3x3` 在全部 4,805 条记录中相同：fx = fy = 1467.06，cx = 1172.53，cy = 1048.18，图像 2448×2048，K1/K2/K3/P1/P2 全为 0。
  - MARS-LVIG 的相机–LiDAR 标定（`AMvalley_calibration`）给出的是另一组值：fx 1453.88，fy 1452.85，cx 1182.53，cy 1045.82，畸变 (k1, k2, p1, p2, k3) = (−0.052, 0.1168, 0.0015, 0.00013, −0.0686)。
- **外参**
  - `T4x4` 是 **camera→world** 变换，相机轴为 OpenCV 约定（x 右、y 下、z 前）。证据：det(R) = 1，正交误差 1.8e-15；光轴与天底的夹角中位 8.6°（p95 19.0°，最大 26.2°），说明相机主要朝下。
  - LiDAR→相机的约定为 `p_c = R·p_l + t`；反向约定的残差为 57–221 m，已排除。
- **位姿记录**
  - JSON 共 4,805 条（相当于 interval=1 的插值位姿），其中 961 条对应 interval5 图像。
  - 相机中心范围：x [−597, 203]，y [−325, 114]，z [−102, 0.9] m；轨迹长 4,317 m，是割草机式航线。
  - 离地高度（AGL）：最小 10.5 m，中位 120.8 m，最大 158.2 m。所有记录位姿都在 mesh 上方。

## 3. 几何与传感器验证（Step 5.2）

### 3.1 单位与坐标系（独立 RTK 证据）
把 `rtk_positions_raw.csv`（5 Hz）中的 lat/lon/alt 按 WGS84 转换到 SRS 原点的 ENU，再插值到 4,805 个图像时间戳，与 `T4x4` 平移比较：
- 平均偏差 (−0.02, +0.02, −0.19) m，标准差 (0.36, 0.30, 0.11) m；
- 去掉均值后，水平误差中位 0.34 m，p95 0.49 m；
- 相机轨迹长度与 RTK 轨迹长度之比 0.9995。

结论：位姿与 mesh 使用同一个公制 ENU 坐标系，尺度误差小于 0.1%。（`units_rtk_check.json`）

### 3.2 Mesh 质量（`mesh_stats.json`、`mesh_layers_g1.json`、`mesh_topdown.png`）
- **可载入性**：trimesh 读入约 8 s；顶点全部有限；退化面 8 个；无重复顶点索引，无未引用顶点。
- **不是 watertight**：mesh 是开放的地形片。
  - 边界边 679 条，占全部边的 1.8×10⁻⁴；非流形边 0 条；绕序一致。
  - 只有 **1 条边界环**，即外轮廓，周长 3,597 m。
  - 1 m 俯视网格上，足迹内部的孔洞为 **0**。
- **连通分量**：共 251 个。最大分量占 99.62% 的面；其余 250 个小碎片共 9,448 个面、20.5 m²，可视为漂浮噪声，可以删除。
  - 最大分量的 Euler 数为 −575，约 288 个小柄（隧道），对渲染没有影响。
- **2.5D 程度**（1 m 网格的竖直射线，命中点按 5 cm 去重）：
  - 单层 99.38%，两层 0.33%，三层及以上 0.29%；层间距中位 1.25 m（树冠、悬垂）。
  - 按法线统计面积：向上（nz > 0.5）81.1%，陡面（|nz| < 0.5）18.7%，向下 0.21%。
- **射线穿透 / 背面命中**：
  - 7 帧真实位姿的渲染（612×512）：命中率 0.979–1.000，命中中背面比例 ≤ 0.021%。
  - 200 个合法网格位置 × 8 个方位角的候选视角：背面比例 ≤ 0.055%（详见 §4）。
  - 结论：mesh 没有可被看穿的孔洞。

### 3.3 位姿–mesh 对齐（`align/alignment_summary.json`、`align/align_<stamp>.png`）

**LiDAR 与 mesh 的距离残差**：沿每条 LiDAR 射线对 mesh 做光线投射，与实测距离比较（4 帧）。

| 帧 | LiDAR 距离中位 | 残差绝对值 p50 / p90 / p95 | 1 m 内比例 | 有符号残差中位 |
|---|---|---|---|---|
| 1658229755 | 105.4 m | 0.29 / 0.56 / 0.72 m | 97.6% | −0.29 m |
| 1658229885 | 135.4 m | 0.91 / 3.91 / 5.02 m | 53.1% | −0.13 m |
| 1658230066 | 86.9 m | 0.54 / 2.34 / 3.35 m | 70.7% | −0.28 m |
| 1658230141 | 117.3 m | 0.77 / 2.10 / 2.77 m | 60.7% | −0.49 m |

- 残差中位约为距离的 0.3–0.7%，在 LiDAR 和照片上有明显的空间结构（树冠、斜坡）。
- 在 `align_1658230066.050822215.png` 中，残差还呈左右梯度，**可能**存在亚度级的姿态偏差（未定量）。
- 这对用 mesh 渲染深度没有影响，因为闭环里的观测完全由 mesh 渲染，不使用真实 LiDAR。

**照片与 mesh 顶点色渲染的对齐**：在 6×5 个分块上做相位相关（1/4 分辨率），位移换算到全分辨率像素。

| 帧 | 相关响应 | 原图：位移中位 / 仿射尺度 x, y | MARS-LVIG 去畸变后：位移中位 / 仿射尺度 x, y / 仿射残差 |
|---|---|---|---|
| 1658229755 | 0.39 | 13.0 px / −1.30%, −1.68% | 11.2 px / +0.15%, +0.16% / 1.6 px |
| 1658229810（倾斜 25°） | 0.56 | 12.5 px / −1.16%, −1.30% | 11.3 px / +0.18%, +0.18% / 2.3 px |
| 1658229885 | 0.44 | 10.1 px / −1.11%, −1.32% | 11.3 px / −0.93%, +1.73% / 13.2 px |
| 1658229975 | 0.47 | 11.6 px / −0.85%, −0.81% | 11.0 px / +0.38%, +0.25% / 2.8 px |
| 1658230066 | 0.42 | 12.0 px / −1.18%, −1.28% | 11.3 px / +0.17%, +0.18% / 2.1 px |
| 1658230141 | 0.34 | 12.8 px / +0.25%, −2.59% | 11.1 px / +0.08%, +0.00% / 1.8 px |
| 1658229736（起飞，AGL 约 35 m） | 0.10 | 不可靠 | 不可靠 |

- **原图**：直接用 JSON 中的 K（无畸变）时，存在约 −1.2% 的尺度差，即畸变加焦距不一致。
- **去畸变后**：用 MARS-LVIG 的畸变参数去畸变后，6 帧中有 5 帧尺度误差 ≤ 0.4%、仿射残差约 2 px，只剩一个恒定平移，约 (−10.5, −2.5) px（约 0.43°）。
- **换算到仿真分辨率**：约 11 px 的误差在 456×256（fx = 221.7）下约为 1.7 px。
- **起飞帧**：目视对齐良好，房屋和车辆都对上（见 `align_1658229736.063320980.png`），但 mesh 顶点色在近距离太模糊，相关失败。
- **影响范围**：这个误差只影响“真实照片与 mesh 对比”。我们的系统只使用由 mesh 渲染的深度，不受影响。

### 3.4 渲染器一致性（`proto/p3d_crosscheck_s1.json`、`proto/p3d_crosscheck_s0.2.json`）
- **设置**：用冻结仿真器的相机模型（PyTorch3D 0.7.8 `FoVPerspectiveCameras`，短边 fov 60°，456×256，znear 1，zfar 750），mesh 转为 y-up `(x, z, −y)`，尺度 s ∈ {1, 0.2}。在 CPU 上光栅化（magician env），把 zbuf 与 embree 的 z-depth 比较。
- **结果**
  - 两种方法命中率都是 100%。
  - |差| 的 p50 / p95 / p99 = 0.018 / 0.076 / 0.16 mm（s = 1）。
  - 左右翻转后差值中位为 13.0 m，说明图像朝向和手性一致。
- **耗时**：CPU 光栅化 33 s/帧（178 万面）；embree 渲染 456×256 一帧 0.067 s。GPU 耗时未测（本任务不允许）。

## 4. 闭环要素原型（CPU，参数仅作示意，**不是冻结协议**）（`proto/prototype_grid.json`、`proto/prototype_grid.png`、`proto/view_hit_stats.json`）

**原型参数**
- 水平网格间距 40 m；高度层 z ∈ {−150, −125, …, 0}，共 7 层（ENU 绝对高度）。
- 合法位置条件：
  - 离 10 m 邻域内的最高地形至少 10 m；
  - 在足迹内且离边界至少 20 m（见 §6 的边界问题）；
  - 26 方向探针射线检测是否在几何体内部。
- 移动合法性：沿单轴移动一格，用 embree 做线段射线检测，阈值为线段长度加 2 m 余量。

**结果**

| 项目 | 数值 |
|---|---|
| 网格候选 / 足迹内 / 满足离地间隙 | 3,689 / 1,995 / **1,263** |
| 26 探针背面命中（视为“在内部”） | 3 / 1,263；作为对照，真实记录位姿也有 3 / 961 被判为“内部”，即误报率约 0.3%。实际实现应改为“≥ k 条射线背面命中”而不是“任一条” |
| 相邻合法位置之间的边 / 被 mesh 阻挡 | 3,222 / 0；整个图只有 1 个连通分量，节点度中位 5 |
| 线段检测自检 | 2,000 条穿过地形的竖直线段 100% 被阻挡；离地 3 m 的随机 40 m 水平线段 47% 被阻挡 |
| GT 表面 ROI（足迹向内收缩 30 m） | 1,527,496 个面，599,069 m²，占 mesh 面积 82.7%；按面积加权采样 100k 点（0.167 点/m²） |
| ROI 中被至少一个合法位置看到的比例（视线检测，不计 FOV） | 99.86%（R = 150 m）/ 99.87%（R = 350 m） |
| 不同俯仰角的视角命中率（200 位置 × 8 方位角） | −60°：94.5%；−30°：78.0%；0°：36.7%；+30°：3.9%；+60°：0.03% |
| 命中距离中位 | 98 / 134 / 182 m（−60° / −30° / 0°） |

对照 Macarons++ 的 `settings.json`（实测 14 个场景）：
- 场景盒 36–195 runtime units；
- 室外相机网格每个水平轴 5–16 格，间距约 10 units；
- **室外 12 个场景都是 5 个俯仰角 × 10 个方位角**（室内 barts/sestino 为 4 × 6），不是 5 × 8。

## 5. GO 判据逐条核对

| 判据（任务书 §5） | 状态 | 证据 |
|---|---|---|
| Mesh 可载入、面非空、bounds 有限、尺度可信 | ✅ | §2、§3.1、§3.2 |
| 非流形 / 孔洞 / 法线 / 射线穿透比例 | ✅ 已测 | 非流形 0；内部孔洞 0；背面 ≤ 0.055%；非 watertight 只是开放边界 |
| 外参与 mesh 对齐，渲染与标定图 / 点云大致对应 | ✅ | RTK 0.34 m；LiDAR 0.29–0.91 m；照片约 11 px @ 2448（约 0.43°） |
| 深度 near/far 与单位 | ✅ | 真实位姿 z-depth p1–p99 范围 31–236 m；PyTorch3D 与 embree 一致到 0.2 mm |
| 未把 LiDAR 稀疏深度当作完整真值 | ✅ | LiDAR 只用于对齐检验；GT 深度来自 mesh 渲染 |
| 可通行相机候选 / 动作图，且不穿越建筑 | ✅ 原型 | §4：1,263 个合法位置、单连通图；线段检测通过自检 |
| GT mesh 观测仿真器 | ⚠️ 部分 | 渲染器已验证等价，但**闭环 adapter 未实现** |
| 无泄漏在线 planner | ⚠️ 未测 | planner 本身不变，但 adapter 中的 GT 访问边界还没写 |
| 可重复 coverage evaluator | ⚠️ 原型 | GT 采样和可见性已有；ε、cell cap、ROI 尚未冻结 |
| 可运行的公平 baseline | ❌ 未验证 | Random 实现简单；MAGICIAN 需要 Macarons++ 格式（OBJ/MTL + `settings.json`），能否接受只有顶点色的 mesh 未验证 |
| 模型尺寸 / OOM | ⚠️ 未测 | 禁止使用 GPU。AMvalley 有 249 万面，AMtown 有 2,205 万面，后者风险较高 |

## 6. 阻塞项与风险

1. **B1 adapter 未实现（GO-NBV 的主要阻塞）**：需要一个 `uavscenes_sim.py`，内容参照 EXP5 MP3D 的 `experiments/exp5_future_view/task0/scripts/mp3d_sim.py`（373 行），包括：
   - 坐标变换 ENU m → y-up runtime，尺度 S_RT；
   - 场景盒和 `settings` 字典；
   - 位姿网格（5 俯仰角 × N 方位角）；
   - 合法性检测（离地间隙 + 线段视线）；
   - 渲染与 10% 采样；
   - coverage 计算（ROI、ε、cell cap）。
2. **B2 尺度 / 切块是协议决策，需要用户批准**：
   - 整场景方案：例如 s = 0.15，得到 188 × 100 × 24 units，地形起伏被压扁。
   - 切块方案：例如 300 m 一块，s = 0.4，得到约 120 units 一块，传感器量程 70 units = 175 m；在 −60° 俯视下，87% 的命中点在 175 m 内。
   - 参照 MP3D 先例（S_RT = 7，使量程 70 = 10 m），无论选哪种，都必须确定、公开，并对所有方法一致。
3. **B3 MAGICIAN 兼容性未知**：需要导出 OBJ/MTL（顶点色转纹理或打补丁）和 `settings.json`。EXP5 的 `magician_comparison/adapters/magician_planner_grid.py` 可能可以复用，但未测试。
4. **B4 科学代表性**：AMvalley 是 2.5D 自然地形（单层 99.38%，可见性 99.86%），NBV 问题偏 2D。含建筑的场景只读了文件头：
   - HKisland：339 万面，158 MB；
   - HKairport：764 万面，367 MB；
   - AMtown：2,205 万面，1.08 GB。
   若要考验 3D 能力，建议对其中一个重复本流程（同样只下载一个场景）。
5. **B5 mesh 边界效应**：足迹外没有几何。靠近边界的水平或斜视视角会把本应存在的地形当成天空。原型已限制相机离边界 ≥ 20 m、ROI 向内收缩 30 m，正式协议需要冻结这两个值。
6. **B6 域差异**：T0 在 Macarons++ 地标建筑上训练，AMvalley 是地形加树木，无法预判零样本表现。
7. **B7 许可证矛盾**：HF YAML 写 `cc-by-sa-4.0`，LICENSE 写 BY-NC-SA 4.0。按 NC-SA 执行，派生 mesh 不入库、不再分发。
8. **未运行**：没有做任何 GPU 渲染、T0 前向或闭环（按任务要求）。

## 7. 闭环 adapter 工作量估计（未开始）

| 步骤 | 内容 | 估计 |
|---|---|---|
| A1 | `uavscenes_sim.py`：坐标 / 尺度 / `settings`、网格、合法性（探针改为 ≥ k 条背面命中）、PyTorch3D 渲染、ROI + GT 10 万点 + ε / cell cap 的 coverage；附 CPU 单元测试 | 1–1.5 天 |
| A2 | 起点：吸附记录位姿到网格，或用种子化随机合法格点；冻结 manifest（scene, start, method, protocol, hash） | 0.5 天 |
| A3 | GO-DEPTH smoke（GPU）：在记录位姿与网格位姿上，用 T0 做 future-view 深度预测并计算误差，1 个场景多个候选 | 0.5 天加少量 GPU 时间 |
| A4 | GO-NBV pilot：1 个场景 × 2 个起点 × 100 步，Ours（T0 + D1）与 Random | 0.5 天加 GPU 时间（未实测） |
| A5 | MAGICIAN 适配：OBJ/MTL/`settings.json` 导出与 loader 验证，复用 EXP5 adapter | 1–2 天，有风险 |

合计约 **3–5 个工作日**才能得到有效的配对 pilot。A1–A2 可以完全在 CPU 上完成。

## 8. 文件与复现

**脚本**（`experiments/cvpr_results/exp7/scripts/uavscenes/`，均用 `python -I` 运行）

| 脚本 | 用途 |
|---|---|
| `http_zip.py` | 列出远程 zip 内容 / 按成员 Range 下载并写 sha256 |
| `remote_ply_headers.py` | 只读取远程 PLY 文件头 |
| `inspect_poses.py` | 位姿统计 |
| `check_units_rtk.py` | RTK→ENU 对照 |
| `inspect_mesh.py` | 拓扑、法线、俯视图 |
| `inspect_layers.py` | 层数与孔洞 |
| `check_alignment.py` | 渲染、照片相位相关、LiDAR 残差 |
| `prototype_grid.py` | 网格、合法性、GT、可见性 |
| `view_hit_stats.py` | 不同俯仰角的命中率 |
| `p3d_crosscheck.py` | PyTorch3D 与 embree 对照（magician env） |

**环境**
- `envs/seen2scene`：trimesh 4.10、embreex、cv2；
- `envs/magician`：PyTorch3D 0.7.8；
- 两者都有 huggingface_hub 0.36.0（未使用；直接用 HTTP Range）。没有安装任何包。

**数据**：`work_dirs/cvpr_results/exp7/downloads/uavscenes/`
- `raw/`：原始成员和 `_provenance.tsv`；
- `_meta/`：HF/GitHub 元数据、zip 内容清单、远程文件头；
- `derived_AMvalley_mesh_cache.npz`：顶点、面、颜色缓存，120 MB；
- 许可证副本。

**可视化与 JSON**：`work_dirs/cvpr_results/exp7/visuals/uavscenes/`
- `trajectory.png`、`frames_contact.jpg`、`mesh_topdown.png`；
- `align/align_*.png`、`proto/prototype_grid.png`；
- 全部 JSON。

**清单**：`experiments/cvpr_results/exp7/report/uavscenes_inventory.csv`
