# EXP6 V5 —— 实现架构（对应 `EXP6_V5_ARCHITECTURE_DESIGN.md` v1.0，用户已确认）

代码：`seen2scene/futureview/v5_model.py`（模型）、`experiments/exp6_v5/scripts/train_v5.py`（训练）、`scripts/v5_core.py`（规划器加载）。
配置：`configs/model_v5.json`。旧代码（`seen2scene/futureview/model.py`、`experiments/exp6_v4/*`）与 V4 权重均未修改。

## 1. 数据流
```
D100 P1 粗网格 uint8 [3, X', Y', 32]（EMPTY / BAND / 域内比例）
  └─ encoder_input → [1, 4, X', Y', 32]
       └─ 3D U-Net（V4 EMA@40k 原样继承）→ F0 [1, 32, X', Y', 32]（格 8 VS = 0.088 模型单位）
                                          F1 [1, 64, X'/2, Y'/2, 16]（格 16 VS）
候选相机（原生成器，不变）→ 每视角 7,296 条射线 o, d, tcap（模型坐标，与 V4 / 标签同一 RY.model_rays_torch）
  └─ 11 个解析投影通道（不变）
V5RayDecoder（新，242,757 参数）每条射线：
  q_r     = MLP[Fourier(d), Fourier(o 归一化), log(tcap / S0), tcap / ext, 11 投影通道] → d = 128
  Local   F0 ≤ 24、F1 ≤ 12 token：从一个格到 tcap 的对数深度分层；每层 3 个探针格中取离射线最近的格（整数索引 + gather，
          无插值、无 grid_sample、无 R × N 稠密矩阵）；网格外 / 域外 / 相机后 / 与上一层重复 → mask；
          UNKNOWN 的域内格保留（不是无效）；BAND 之后更深的格不删除
  Coarse  8 token：F1 自适应平均池化到 2 × 2 × 2 区域（来自 F1，不来自 GT）
  Null    1 个可学习 token（所有局部 token 都被 mask 时的回退）
  token   = LN(W_f · feature + GeoMLP(e))，e = [Fourier(log τ), τ / tcap, δ / s, log s, cos, level, coarse 标志,
            观测状态 (EMPTY, BAND, 域内), 方向]；τ = (x − o)·d，δ = 垂直距离
  注意力  4 头，logit = q·k / √32 + GeoBias(e)，masked softmax；z_r = q_r + W_o · Attn；z_r += FFN(z_r) → [N, 128]
  SurfaceMLP b_r(t) = σ(MLP[W_z z_r + W_t PE(log(1 + t / S0), t / tcap)])：任意深度可查询；有限厚度 Δ 邻域内存在
          表面的概率（不是 PDF，可多峰，墙 A 后的墙 B 可同时为高）
  Summary 16 个对数深度探针 t ∈ [S0, tcap] 上的 b 与 logit + log tcap → MLP → 32 维
  CondAdapter LN([z_r, Summary]) → Linear → 64 通道 × out_scale（初值 0.05，V4 射线上下文 RMS ≈ 0.030）
cond [V, 76, 64, 128] = [11 投影通道, 64 V5 射线通道, 有效列]（与 V4 完全相同的布局）
  └─ 原 2D Rectified Flow U-Net（V4 EMA@40k 原样继承；时间采样、目标、10 步 Euler、编码、K = 4、种子不变）
       └─ K = 4 个未来 Depth / HIT → 原解码 → DistP1（S10 稀疏 P1 评分，不变）→ 原规划器
```
Belief 只通过 Summary → CondAdapter → cond 进入 Flow；它**不**进入 coverage、P1、DistP1 或候选生成（`tests/test_g0_invariants.py`）。

## 2. 与设计的对应和实现取舍
| 设计项 | 实现 | 说明 |
|---|---|---|
| II-A Ray Query | 方向 / 原点 / tcap / 11 投影通道的 Fourier + MLP | 投影通道已含解析的已观测几何（Geometry.view 的 ray state） |
| 度量尺度 | 模型单位与体素单位（S0 = 8 VS）+ 归一化深度 | Macarons++ 运行时单位非公制（OBJ × 10），MP3D 1 m = 7 运行时单位；V2/V4 的全部几何本来就在模型单位里（记为偏差说明 D1） |
| II-B 检索 | 对数深度分层最近格（局部）+ 区域池化（粗） | 设计允许"确定性 top-k 或分层选择"；分层保证远处深度也有 token，不被近处稠密格占满 |
| ρ_l(τ) 门控 | 每层 3 个探针格中取 δ 最小者，δ ≤ √3/2 格 | 等价于 ρ = 半格对角线的固定门控；锥形扩张由 F1 / 粗 token 承担 |
| hidden d | 128（设计建议约 192） | 显存 / 速度审计后取 128；4 头 |
| R_chunk | 16,384 射线 / 块，训练时块内梯度检查点 | 显存峰值 4.5 GB / 卡（16 视角） |
| Δ | 8 VS = 一个 F0 格 = 0.088 模型单位 | 标签正例：表面在 Δ/2 内 |
| Summary | 16 个对数深度探针（MLP 输出，不是 U-Net 采样） | 同设计 |

## 3. 继承与新模块
| 模块 | 来源 | 参数 |
|---|---|---|
| encoder（3D U-Net） | V4 EMA@40k 逐名逐形状复制 | 2,412,000 |
| unet（2D Flow） | V4 EMA@40k 逐名逐形状复制 | 22,333,890 |
| dec（V5RayDecoder 全部） | 随机初始化（seed 0） | 242,757 |
| V4 rays（32 点 RayEncoder）/ ray_xattn | **不属于 V5**；只在训练时作为冻结 Teacher（另一个 V2Model 实例）计算 H_old | 76,224 / 25,430 |

V5 推理（规划器 / 固定状态评估）只加载 V5Model：模型没有 `rays` / `ray_xattn` 属性，且 `v5_core` 在预测器前向期间把 `torch.nn.functional.grid_sample` 换成抛错函数（G0-5）。

## 4. 损失
L = L_FM + λ_surf · L_surf + λ_bridge(step) · L_bridge
- L_FM：V2 / V4 原 rectified-flow 速度 MSE（`v2_data.fm_sum`，全局有标签射线数归一化，DDP 梯度求和），不变。
- L_surf：Train-6 网格多表面标签上的 BCE，正例 / 首表面前负例 / 首表面后负例三组各取均值再平均；查询 = 每个存储表面（±Δ/4 抖动）、每个表面旁 Δ..2Δ 的难负例、8 个对数均匀查询、8 个首表面后均匀查询；标签统一为"Δ/2 内有表面"；[t_lo, t_hi] 外不监督（不是负例）。
- L_bridge：mean(((H_new − H_old) / RMS(H_old))²)，H_old = 冻结 V4 Teacher 在同一 P1 / 相机上的 64 维射线上下文；标量 RMS 归一化保留原始幅值匹配；同时记录 H_new / H_old 均值、标准差、余弦。
- λ 取值按梯度尺度在训练前冻结（`FROZEN_PROTOCOL.md`）。
