# ARCHITECTURE_AUDIT —— 空间表示审计（2026-11-02，只读）

**标注**：
- **【代码】** 代码确认（路径:行号）；
- **【实测】** CPU 形状检查或日志测量；
- **【报告】** 历史报告；
- **【假设】** 未验证的判断。

路径前缀 `seen2scene/futureview/` 省略为 `fv/`。

## 1. P1 地图 → 细网格 → 8³ 粗网格
**细网格**【代码】
- 体素 `VS = 0.011` 模型单位（`m_common.py:28`；`fv/geometry.py:15`；`fv/model.py:13`）。模型单位 = 运行时单位 / k_M。
- 高度固定为 256 体素（断言在 `m_common.py:143`）。水平维度 `round(extent/VS)`，每轴至少 256（`m_world.py:30-36`）。
- domain 为“体素中心在场景框内且被窗口覆盖”（`m_world.py:53-60`）。截断 `TRUNC = 3 VS`。

**P1 融合**（`m_p1.py`）【代码】
- **BAND**：每个采样点做精确体素 DDA，写入 |sdf| < trunc 的 SDF 均值（:69-118）。
  - S10 只用 10% 的采样点；D100（FullDepth-P1）用全部有效像素（`p1_full.py:24-46`）。
- **EMPTY**：整幅深度图每半个体素步进一次，雕刻到 `min(t_hit − trunc, 70)`（:121-148）。
- **结构分类**：`structure()` 中 BAND 优先于 EMPTY，阈值 `TRUNC × 0.999`；编码 EMPTY = 0、UNKNOWN = 1、BAND = 2（:161-176）。

**粗网格** `fv/geometry.py:220-230 coarse_grid(struct, domain, block=8, pad_multiple=4)`【代码】
- 3 个通道：`(struct==EMPTY)&domain`、`(struct==BAND)&domain`、`domain`。
- 用 `avg_pool3d(·, 8)`（核 = 步长 = 8，不重叠）求块内占比，再 × 255 量化成 uint8。
  - 分母恒为 512，**不按块内 domain 体素数归一化**；V3 是按 domain 归一化的。
- X'、Y' 向上补齐到 4 的倍数（高端补 0，即补出的部分算作 domain 外）；Z' = 256 / 8 = 32。
- UNKNOWN 是隐含量：domain − EMPTY − BAND。
- 【实测】块在 X 方向 2/8 位于 domain 内时，存储值为 64 = round(255 × 0.25)，证实分母是 512。

**编码器输入** `fv/model.py:64-68 encoder_input`【代码】
- `u8 / 255` 加上归一化高度 `(k + 0.5) / Z'`，得到 4 个通道 [EMPTY, BAND, domain, z]。

**粗网格在哪里生成**【代码】
- 预计算进分片：`task1/t11_build.py:75`（室外；E2 D100 分片经 `e2_build.py` 复用）、`task2/t21_build.py:74`（MP3D，存为 `coarse_b*`）。
- 规划器实时生成：`v2_plan_core.py:52-53`。`t12_eval_sc.py:116-119` 断言实时网格与分片逐位相同。

## 2. 三维编码器（Encoder3D，`fv/model.py:27-61`）与 F0 / F1
| 层 | 结构 | 输出（以粗格为单位） |
|---|---|---|
| stem + l0 | Conv3d 4→32（k3）+ 2 × Res3D(32)（GN8、SiLU） | x0：[32, X', Y', Z'] |
| d1 + l1 | 步长 2 卷积 32→64 + 2 × Res3D(64) | x1：[64, X'/2, Y'/2, Z'/2] |
| d2 + l2 | 步长 2 卷积 64→96 + 2 × Res3D(96, dilation 2) | x2：[96, X'/4, Y'/4, Z'/4]，**不输出** |
| u1 + r1 | 最近邻上采样 + 拼接 → 64 | **F1** = [64, X'/2, Y'/2, Z'/2]，格边长 2·S0 = 16 VS |
| u0 + r0 | 上采样 + 拼接 → 32 | **F0** = [32, X', Y', Z']，格边长 S0 = 8 VS = 0.088 |

- 编码器参数 2,412,000。`feat_ch = 32 + 64 = 96`（断言在 `model.py:282`）。【代码】【实测】
- **坐标映射**：点 p（模型单位）→ `g = 2p/ext − 1`，其中 `ext = F0.shape × BLOCK × VS`；用 `grid_sample(align_corners=False, padding_mode="zeros")` 采样，网格轴顺序为 (Z', Y', X')（`model.py:92-98`）。F0 的第 i 格覆盖 [8i, 8i+8) VS。【代码】
- 【实测】B2 checkpoint 严格加载。dummy 粗网格 [3, 16, 20, 32] 得到 F0 (1, 32, 16, 20, 32)、F1 (1, 64, 8, 10, 16)、x2 (1, 96, 4, 5, 8)。
- 【假设】按卷积下标推算，F1 的采样可能有 0.5·S0（4 VS）的固定偏移：d1 是 k3 / s2 / p1 卷积，F1[j] 中心在 (2j+0.5)·S0，而 grid_sample 假定在 (2j+1)·S0。训练和推理一致，网络可以学着适应。未测量。
- V5 的 Memory 中，`F0` / `F1` 即上表的 y0 / y1（`fv/v5_model.py:41-44、204-208`）。【代码】

## 3. 32 点射线采样与 Conv1D → 64 维（`fv/model.py:74-109`）
**射线**【代码】
- 64 × 114 = 7,296 条，对应 256 × 456 图像中 4×4 像素块的中心（`fv/rays.py:11-25`）。训练时每视角抖动 ±2 像素。
- `tcap = 70 / dz / k_M`：运行时 z 深度 70 对应的射线长度（`rays.py:58-81`）。

**32 个采样点**【代码】
- 公式 `t = (arange(32) + 0.5) / 32`，`p = o + d · tcap · t`（`model.py:83-86`）。
- 从相机中心到 tcap **均匀分布**，间距为 tcap / 32。不用 t_near、盒子入口、t_known，也不抖动。网格外的点特征为 0，标志位为 0。

**每点特征**【代码】
- 对 F0（32 维）、F1（64 维）做三线性采样，对 dom 做最近邻采样并转成“在 domain 内且在网格内”的标志。
- 合计 [R, 97, 32]，在 fp32 下采样（`model.py:88-102`）。

**Conv1D 头**（`model.py:78-81、104-109`）【代码】
1. `c1` Conv1d 97→64（k3）+ SiLU；
2. `c2` 64→64（k3、步长 2，长度 32→16）+ SiLU；
3. `c3` 64→64（k3、步长 2，16→8）+ SiLU；
4. 展平 64 × 8 = 512，再接 `fc` Linear(512 → 64)。

- 没有池化，沿射线的位置信息由展平顺序保留。
- **`fc = Linear(hidden × (n_points // 4), out_ch)` 依赖采样点数**，B2 中权重形状为 (64, 512)。
- 射线头参数 76,224。

**实际间距**（中心射线 tcap/32）【实测】

| 场景 | k_M | 间距（细体素） | 间距（F0 格） |
|---|---|---|---|
| liberty | 28.7 | 6.9 | 0.87 |
| bannerman | 21.7 | 9.2 | 1.15 |
| pisa | 14.0 | 14.2 | 1.8 |
| dunnottar | 9.7 | 20.6 | 2.6 |
| MP3D 两个区域 | 5.5–6.7 | 30–36 | 3.7–4.5 |

- 室内的射线采样远比 F0 格稀疏，相邻两个采样点之间要跨过约 4 个 F0 格。
- 【假设】这可能限制室内对近处细结构的利用，未验证。

**条件输入与 U-Net**【代码】
- 11 个投影通道（`fv/geometry.py:169-191`）：
  - 射线类别 one-hot（3 个）；
  - t_known、d_lo、d_hi、L_u，均除以 tcap（4 个）；
  - n_seg、d_z、cam_z、log tcap（4 个）。
  - 这些通道由**细网格**每半体素步进得到，与 BLOCK 和采样点数无关。
- 投影通道与 64 维射线特征一起组成 [V, 75, 64, 114]，右侧补到 128 列，再加 valid 列，共 76 个通道；加上 x 的 2 个通道，U-Net 输入为 78 通道（`model.py:297-312`）。
- U-Net 通道 (64, 128, 192, 256)，带 FiLM 时间条件，在 L3 和 mid 层做自注意力，共 22.3M 参数。
- **预测目标**：每条射线 x = (b, y) 的 rectified-flow 速度，推理用 10 步 Euler，K = 4。
  - b = ±1 表示是否命中；**y = 2s − 1 是首个命中点在该射线 UNKNOWN 采样中的分位**，不是公制深度。
  - 解码时只把命中点放到 t_known 之前的 UNKNOWN 体素里（`fv/geometry.py:101-166、194-217`；`task1/v2_data.py:141-162`）。

## 4. V3 / V4 / V5 与 V2 的差异【代码】
- **V3（`seen2scene/v3`）**：
  - 输入换成 **4× 聚合的部分 TSDF**：sdf、band、free 三通道按 domain 内体素数归一化，加上 valid。网格补齐到 8 的倍数（`partial_tsdf.py:31-81`）。
  - VAE 只有一级（步长 2），潜变量 μ 共 16 个通道，格边长 8 VS，**与 F0 相同，但没有 F1**。
  - LatentRayEncoder 用同样的 32 点和 Conv1D 头，每点特征为 μ（16）+ valid（1）= 17 维。
  - T1 在 U-Net 瓶颈（8×16）对场景 token 做每视角交叉注意力。场景 token 是已观测的格（band 或 free），UNKNOWN 格不作 token，每视角最多 512 个。
- **V4**：V2 不变，另加 RayXAttn——每条射线对自己的 32 个点 token 做 4 头注意力，结果作为残差加到 64 维射线特征上。RayXAttn 的权重与采样点数无关（`model.py:126-178、314-319`）。
- **V5**：编码器和 U-Net 来自 V4；RayEncoder 和 RayXAttn 换成 V5RayDecoder：
  - F0 最多 24 个、F1 最多 12 个 token，按**对数深度分层**取最近格，用整数下标 gather，不插值；
  - 另有 8 个粗 token（F1 做 2× 平均池化）和 1 个 null token；
  - 1 层交叉注意力加 SurfaceMLP，在 16 个对数间隔探针上评估；
  - 输出仍是 64 维射线特征，所以条件布局不变（`fv/v5_model.py`）。

## 5. 改块大小或射线采样数需要改什么，是否需要训练
| 修改 | 需要改的位置 | 权重形状 | 数据 | 是否需要训练 |
|---|---|---|---|---|
| **块 8³ → 4³ 或 16³** | ① `fv/geometry.py:220` 的 `block=` 默认值，以及它的所有调用方（`v2_plan_core.py:53`、`t11_build.py:75`、`t21_build.py:74`、`e2_fixed`、ablation 等）；② `fv/model.py:15 BLOCK`（用于射线的 `ext` 和 V4 的 `ext`；V5 的 `S0 = BLOCK × VS`）。**两者必须一致，但代码里没有断言，也没有 model_cfg 键记录**——不一致时采样会悄悄错位 2 倍。 | **不变**【实测：B2 在 4³ 网格上能前向，F0 = (1, 32, 32, 32, 64)】 | **所有分片的粗网格都要重建**（室外 T1.1、E2 D100、MP3D `coarse_b*`）。分片里没存细结构，需要重新回放 P1（t11_build / t21_build / e2_build），没有现成的“只重算粗网格”脚本 | **需要**【假设，但很有把握】：块内占比的粒度从 1/512 变成 1/64、感受野的公制尺度、采样间距与格的比例都会改变。4³ 时编码器显存约增加 8 倍 |
| **射线 32 → 16 或 64 点** | `model_cfg["ray"]["n_points"]`（model_v1.json:5）。必须是 4 的倍数（N = 30 时前向报错） | **`rays.fc` 形状改变**：(64, 512) 变为 (64, 256) 或 (64, 1024)，严格加载失败【实测】。c1–c3 不变；V4 的 RayXAttn 不变；V3 的 fc 同样依赖 N；V5 不使用 N | 不需要重建：采样点由 o、d、tcap 实时计算 | **需要**【代码：fc 必须重训】；射线头和 U-Net 实际上也需要微调或重训【假设】。可选：加载除 fc 以外的所有权重，或沿位置轴插值 fc（启发式） |
| **采样点数不变，改间距**（例如对数间距或从 t_near 起） | `model.py:83-86 points()`；V4 的 RayXAttn 写死了 `t_j = s_j × tcap`（:156-158）；V3 的 LatentRayEncoder 继承 `points()` | 不变 | 不需要 | **需要**【假设】 |
| **刷新周期 R10 → R5 / R2 / R1** | `task0/t06_planner.py:112`、`task1/v2_common.py:51`、`task2/t24_v2_out_planner.py:25`（见 REFRESH_AUDIT） | 不变 | 不需要 | **不需要**。训练时的陈旧增强覆盖 0–9 步，R1–R10 都在训练分布之内【代码】 |

## 6. 其他记录
- `seen2scene/futureview/__init__.py` 提到的 `coarse.py` 并不存在，实际函数是 `geometry.coarse_grid`。
- `t21_build.py:9` 的文档写粗网格有 4 个通道，但实际存储是 3 个通道，z 通道在加载时才加上。【实测】
- V3 的 `model_v3.json` 里 `input.factor`、`pad_to` 等键在代码中没有被读取（grep），实际常量在 `partial_tsdf.py`。
