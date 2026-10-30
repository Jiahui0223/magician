# Task 3 / Experiment 1 — Stage 0 源码审计（2026-10-28）

本报告的路径均相对于 `experiments/`；W2 = `work_dirs/exp5_future_view/task2`，W3 = `work_dirs/exp5_future_view/task3_full_depth_p1`。

## 1. P1 的实现（冻结代码 `exp3b_macaronspp_transfer/m_p1.py`，类 `P1Fusion`）
| 项 | 实际行为 |
|---|---|
| 存储 | GPU 上的稠密扁平数组，每个体素 3 个量：`sum`（int64 定点 SDF 累加和，`FIXED = 2^40`）、`cnt`（int32 融合次数）、`empty`（bool 挖空标记）。约 13 字节 / 体素 |
| BAND 更新 `integrate_band(points, cam_center)` → `_band_rays` | 对每个点沿"相机 → 点"射线做体素遍历，范围为 [L − trunc − √3·vs, L + trunc]。体素中心到该点的距离作为 SDF，符号为：体素在点前方为正。满足 \|sdf\| < trunc 且在区域内的体素，sum += round(sdf·FIXED)，cnt += 1。trunc = 3 voxel = 0.033 模型单位 |
| EMPTY 更新 `integrate_free(zbuf, mask, R, T)` | **全分辨率**像素网格，每条像素射线从 NEAR_CLIP（z = 0.5）挖到 min(z/dz − trunc_rt, 70)；mask 为假的像素挖到 70；采样步长为 vs/2。挖到的体素 empty = True（只设不清） |
| 顺序 | `integrate_record`：先 BAND（`rec["sample_points"]`），再 EMPTY，然后 n_frames += 1 |
| 转换 `input_volume()` | 有 BAND（cnt > 0）→ SDF 均值截断到 ±trunc；否则 EMPTY → +trunc；否则 UNKNOWN → −trunc。**BAND 优先于 EMPTY** |
| 分类 `structure()` | \|v\| < 0.999·trunc → BAND；v ≥ 0.999·trunc → EMPTY；其余 → UNKNOWN |
| 溢出 | 每次贡献 < 0.033·2^40 ≈ 3.6e10，int64 可容纳约 2.5e8 次；实测最大 cnt = 3,987（全深度，室内 t = 145），无风险 |

## 2. 10% 表面采样（已用数据核实）
- **MP3D**：`task0/scripts/mp3d_sim.Sensor.sample`。深度 = clamp(zbuf, 0.5, 750)，有效像素 = mask ∧ depth < 70；按种子 `derived_seed(level, start, t, "sample")` 做 `randperm`，取前 int(0.1·N) 个。
- **Macarons++**：MAGICIAN `compute_partial_point_cloud(gathering_factor·2 = 0.1)`。`apply_perfect_depth_simple` 中 error_mask = mask，有效像素规则与 MP3D 相同，种子 `derived_seed(scene, start, t, "sample")`。
- **核实结果**：
  - 采样点数恰为有效像素的 10%（如 11,295 / 112,951），且全部是有效像素；
  - 用 `m_common.pixel_rays` 对有效像素反投影，与记录中的 `sample_points` 最大差 ≤ 1.2e-3 运行单位（float32 精度）。

## 3. 规划器中的 P1 与输入
| 项 | 位置与行为 |
|---|---|
| 室内局部网格 | `task0/scripts/t06_planner.Planner.refresh/step`。t % 10 == 0 时以当前相机为中心建区域（n_win = 5，由 `task2/scripts/t23_v2_in_planner` 打补丁），新建 `m_p1.P1Fusion` 并重放第 0..t 步的全部记录；同时重算编码器特征 h 和噪声。其余步增量融合 |
| 室外全局网格 | `task1/scripts/v2_planner.main`。每个场景一个 `TD.scene_geometry` 网格，`m_p1.P1Fusion` 只建一次，每步增量融合；R10：t % 10 == 0 时重算 h 和噪声 |
| 射线通道（11 个） | `v2_plan_core.Geometry.view` → `futureview.geometry.project` → `channels`：类别 one-hot、t_known、d_lo、d_hi、L_u（相对 tcap 的 float16）、n_seg、方向 z、相机 z、log tcap。**每步用当前 P1 计算** |
| 3D 粗网格 | `geometry.coarse_grid(struct, domain)`（8³ 块：EMPTY / BAND / 区域内占比）+ 高度通道 → `model.encode`。**只在刷新步**计算 |
| 射线特征 | `RayEncoder`：沿每条射线 32 个点取 F0 / F1 |
| DistP1 | `futureview.scoring.new_set_from_depth`。命中点须在区域内、且不在 `band_dil`（BAND 的 3×3×3 膨胀）中；对体素去重；权重 min(1, (z/z0)²)，同一体素取最大值后求和。**`band_dil` 由同一个 P1 得到，所以 B1 会同时改变输入和"是否算新"的判断** |
| 覆盖评估 | GT 侧，与 P1 无关：MP3D 为 C_obs，Macarons++ 为单调覆盖（`task2/scripts/t25_out_cov.py`）。不改 |

## 4. 闭环脚本（Task 2，不改）
| 数据集 | 闭环脚本 | 规划器子进程 |
|---|---|---|
| MP3D | `task2/scripts/t23_in_sim.py`（magician env） | `t23_v2_in_planner.py`（seen2scene env） |
| Macarons++ | `task2/scripts/t24_out_sim.py` → `task1/scripts/v2_sim.run` | `t24_v2_out_planner.py` → `v2_planner.main` |

- 子进程从 `os.environ` 继承环境变量。
- 模型：V2-Mix，`work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt`，sha256 为 `343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516`（Task 2 `eval_*_v1.json`）。

## 5. 准备做的最小改动
1. **`task3_full_depth_p1/scripts/p1_full.py`**：`P1FusionFull(m_p1.P1Fusion)` 只重写 `integrate_record`。
   - BAND 点 = 记录中的 10% 采样点（逐位，与 B0 相同）+ 其余有效像素的反投影；
   - 其余（`integrate_band`、`integrate_free`、优先级、截断、分辨率）全部继承不变；
   - `patch()` 根据环境变量 `T3_P1_MODE=full` 在**规划器进程内**替换 `m_p1.P1Fusion`。
2. **规划器包装脚本**（MP3D / Macarons++ 各一个）：调用 `patch()` 后，原样运行 Task 2 的规划器子进程脚本。
3. **闭环启动脚本**：原样运行 Task 2 的闭环脚本，只把子进程脚本换成上述包装，输出写到 W3。
   - B0 与 B1 用同一套启动脚本，只差环境变量；
   - B0 重跑的结果应与 Task 2 的 V2M 逐位相同，以此作为复现检查，并在同等负载下测量耗时和显存。

## 6. 资源影响（实测，同一条轨迹）
| 项 | 稀疏 P1（B0） | 全深度 P1（B1） |
|---|---|---|
| 每步增量融合 | 0.07 s | 0.11 s |
| 室内 10 步重建（平均 / 最后一次 t = 140） | 5.2 / 9.8 s | 8.0 / 16.1 s |
| 峰值显存 | 6.9 GB（室内）/ 5.6 GB（室外） | 不变 |
| BAND 体素 | 室内 3.1M / 室外 2.5M | 5.0M / 5.3M |
| EMPTY 体素 | 69.7M / 78.3M | 68.9M / 77.4M（BAND 覆盖了少量 EMPTY） |

估算闭环耗时：MP3D 每条 episode 增加约 1–2 分钟；Macarons++ 每步增加约 0.04 s。

## 7. 待确认 / 风险
- **B1 对模型是分布外输入**（模型在稀疏 P1 上训练）。闭环测的是"冻结模型直接换地图"的效果，不是重训后的上限。
- **B1 同时改变输入和 DistP1 的 `band_dil`**。若结果有变化，再在固定状态上做 B2 / B3 归因（只换输入 / 只换评分）。
- **本实验不改区域裁剪**：室内 13 m 区域不变。
