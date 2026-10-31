# E1 — 原 DistP1 审计（Phase 0；2026-10-31 17:00；代码 `seen2scene/futureview/scoring.py`、`exp3b m_score.py / m_p1.py`、`task1 v2_plan_core.py`、`task3 p1_full.py`、`task0 mp3d_sim.py`、`task2 t25_out_cov.py`）

## 1. 评分链路（B2 / V3-T1 规划器，每候选每样本）
| 步 | 实现 | 事实 |
|---|---|---|
| 1 射线 | `v2_frozen.view_rays` → `m_common.pixel_rays`：64 × 114 条打分射线 = 4 × 4 像素块中心（MAGICIAN 反投影约定，fov 60°，短边 256）；每条射线 d（模型帧单位向量）、dz（单位射线长度的 z 深度）、tcap = 70 / dz / k_M（射程 70 运行单位 = MP3D 10 m） | 射线在切平面上等距（Δ(X/Z) = 8 tan30°/255 = 0.01811 / 块），因此一条射线在深度 z 的足迹 = (0.01811 z)² · dz / cosθ（精确推导见 `t2_scorers.py`）|
| 2 预测深度 | `geometry.decode`：模型输出 (b, y) 只在射线的 UNKNOWN 采样段上解码为首次命中 t（模型单位）；非 UNKNOWN 射线取投影结果（已知表面 t_known 或无命中）| 预测命中**结构上**不可能落在 EMPTY、稠密已知表面之后或域外（B2 模式下 t_known 来自 D100 稠密地图，`Geometry.from_fusion(P1FusionB2)`）|
| 3 反投影 / 体素化 | `new_mask_from_depth`：q = o + d t，ijk = floor(q / VS)，VS = 0.011 模型单位（MP3D k_I 6.8–11.2 → 1.0–1.6 cm；Macarons++ k_M 11.6–29.6 → 0.13–0.33 运行单位）；域外 → 不计 | 区域出口之外的 t 在 `view()` 的 tmax 处截断（MP3D n_win = 5 区域；Macarons++ 整场景）|
| 4 已知排除 | `new = hit & ~band_dil[ijk]`，band_dil = 稀疏 10% P1 的 BAND 体素（\|v\| < 0.999 TRUNC）3×3×3 最大池化 | BAND 只沿 10% 采样射线、±3 体素厚；膨胀 1 体素 ≈ 1–2 cm（MP3D）；**与覆盖率 eps（7.8–12.8 cm）相差 5–13 倍**；稀疏 BAND 有洞（EXP5：自观测射线 25–54% 穿过）|
| 5 EMPTY / UNKNOWN | EMPTY 来自全深度雕刻（稀疏与 D100 地图相同）；UNKNOWN = 既非 BAND 也非 EMPTY | 评分只看 band_dil；EMPTY / UNKNOWN 通过解码（步 2）间接起作用 |
| 6 距离权重 | w = min(1, (z / z0)²)，z = t k_M dz（运行时 z 深度），z0 = MAGICIAN scene_scale / 2 = 楼层 / 场景框三边均值 / 2 | MP3D：z0 = 32.7 / 40.0 / 73.2 / 84.7 / 91.4 运行单位（oLBMNvg9in8 4.7 m、TbHJrupSAjP 5.7 m、其余 10–13 m > 射程 10 m）；Macarons++：15.1–49.8（射程的 22–71%）。(z/z0)² 恰好补偿射线密度 ∝ 1/z²，使计数 ∝ 面积，**但在 z0 处封顶、且无 1/cosθ** |
| 7 去重 | 唯一体素，权重取该体素上各射线的最大值，求和 | 体素（1–2 cm）远小于射线在表面上的间距（z = 3 m 时 5 cm），去重只在 < 1 m 处起作用 |
| 8 K 聚合 | K = 4 样本各自 DistP1，取均值 | 并列 → 最小索引；全 ≤ 0 → 种子随机 fallback |
| 9 输入 vs 评分地图 | B2 / T1：模型输入 D100（全部有效像素 BAND），评分 band_dil 来自稀疏孪生（`p1_full._patch_geometry_b2`）；t_known（解码上限）来自 D100 | 两种密度并存是冻结设计（Task 3 B2）|
| 10 耗时 | Experiment C 实测：C1 子进程（稀疏融合 + 几何 + 10 候选 GTD）1.7 s / 步（室内）、0.33 s（室外）；B2 决策 2.6 / 2.0 s（含 Flow 采样）；本 Task 冒烟：17 张深度图 × 9 候选的全部变体（含 KD 树查询）2 s / 状态，峰值 11.8 GB（含 D100 重放）| 新评分器的额外成本主要是历史采样点 KD 树（室内 ≤ 75 万点）查询，约 0.1 s / 候选 |

## 2. 覆盖率口径
| 数据集 | 定义 | 关键量 |
|---|---|---|
| MP3D C_obs（`mp3d_sim.Coverage`）| 300k 面积加权 GT 面点（楼层框内），可见 = 任一位姿可见；covered \|= 落入任一**历史 10% 采样点**的 eps 邻域；值 = covered ∧ visible / visible | eps = 0.08 模型单位 × k_I = 0.55–0.90 运行单位（7.8–12.8 cm）|
| Macarons++ 单调覆盖（`t25_out_cov`）| MAGICIAN GT 单元点；covered \|= 落入任一历史 10% 采样点的 eps 邻域；/ \|G\| / 可见比例 | eps = 2 × 0.05 × 10 = 1.0 运行单位 = 3.1–7.9 体素 |
| 共同点 | 增益 = 新落入 eps 邻域的 GT 点数 ∝ 新样本点 eps 圆盘并集覆盖的表面面积；10% 采样在射程内的间距（3.2 像素 ≈ 3.2·0.0045·z）< 2 eps 对 MP3D 全射程成立（z_trans = 2 eps / 0.0145 = 76–124 > 70）、Macarons++ 亦然（138）→ **真实增益在射程内 ≈ 可见新表面面积**（无 z0 封顶、无 cosθ 衰减），只在采样点稀于圆盘时饱和 |

## 3. 四处口径差（→ 评分器组件）
1. **新颖性尺度**（S2）：DistP1 用 1 体素膨胀的稀疏 BAND 判新；覆盖率用 eps（5–13 体素）邻域、参照集 = 全部历史采样点。后果：已覆盖表面上略偏的命中（视角变化、BAND 洞）被计为"新"→ 高估已观测区域的收益。
2. **z0 上限**（S1c / S1d）：覆盖率 ∝ 面积直至射程；DistP1 在 z > z0 处按 1/z² 衰减（oLBMNvg9in8、TbHJrupSAjP 在 4.7–5.7 m 以外；Macarons++ 在 15–50 运行单位以外）。
3. **入射角**（S1a / S1b）：面积 ∝ 1/cosθ；DistP1 无补偿 → 斜视表面（地面、走廊侧墙）低估。
4. **区域裁剪**（R1）：MP3D 只计 n_win = 5 区域内命中；覆盖率是全层。预测深度无法越出模型域，GT 诊断可（G_\* 变体）。
另：EXP5 A1 已证明去掉距离权重（纯计数）更差（Top-1 0.69 → 更低），S1 不是删权重而是用物理面积替换；EXP5 A2 的"幻象遮挡"（已知表面前的未知区命中）由 S3 的薄前缘规则（命中距稠密已知表面 < 3 体素）处理，"命中在 EMPTY / 已知表面之后"在冻结解码器下结构上不可能，E1 用计数核实。

## 4. 复现检查（`e1_score_fixed.py` 每状态断言）
S0(t\*) = Experiment A `gtd_distp1` 逐位；S0(缓存样本) = 缓存 `distp1_k` 逐位；两者在 2 状态冒烟与正式运行中逐状态记录（`checks` 字段），分析汇总于 `E1_GT_SCORING.md`。
