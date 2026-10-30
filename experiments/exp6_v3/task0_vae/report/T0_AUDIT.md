# T0 — 审计（2026-10-29 / 30；G0 数值见 `T0_G0_TESTS.md`）

## 1. 代码与数据路径
| 项 | 路径 / 版本 |
|---|---|
| 仓库 | `Jiahui0223/magician` 分支 `seen2scene`；V2 归档 `efd7e14`；V3 代码在其后的 `[V3][*]` 提交 |
| V3 包 | `seen2scene/v3/`（partial_tsdf、vae、cross_attention、flow_unet、model、data、losses、trainer、predictor、eval_dev） |
| 复用的 V2 代码 | `seen2scene/futureview/`：`model.Res3D / Res2D / UNet2D / RayEncoder / pad_rays`，`geometry.channels_from_stats / decode / coarse_grid / CAT / CLS_*`，`rays.model_rays_torch / sha_seed` |
| 冻结的融合 | `experiments/exp3b_macaronspp_transfer/m_p1.py`（P1Fusion）+ `experiments/exp5_future_view/task3_full_depth_p1/scripts/p1_full.py`（P1FusionFull = D100） |
| 射线通道 / 标签 | E2 D100 分片：`work_dirs/exp5_future_view/task3_full_depth_p1/e2/w2_d100/mp3d/views/<level>/<traj>.npz`（237 条）、`e2/out_views_d100/<scene>/<traj>.npz`（180 条）；由冻结的 `t21_build.py` / `t11_build.py` 生成（E2 审计：稀疏模式重建与原分片逐位相同） |
| VAE 输入缓存 | `work_dirs/exp6_v3/shared_cache/tsdf4/{mp3d,outdoor}/<scene>/<traj>.npz`，`shared/build_tsdf_cache.py` |
| 环境 | seen2scene conda env（PyTorch 2.4.1 cu120），L40S 46 GB |

## 2. 物理尺度（实测）
| 量 | MP3D | Macarons++ |
|---|---|---|
| 细体素 VS | 0.011 模型单位 = VS × k_I 运行时单位 = VS × k_I / 7 m：**8.7–17.6 mm**（k_I 5.5–11.2） | 0.011 模型单位 = VS × k_M 场景单位：**0.11–0.33 scene units**（k_M 9.7–29.6；scene units = OBJ × 10 的非米制单位，不按米解释） |
| 截断 TRUNC | 3 体素（2.6–5.3 cm） | 3 体素（0.3–1.0 scene units） |
| 细网格 | 区域 n_win = 5 每 10 步刷新：典型 1064 × 1064 × 256 = 2.9 亿体素 | 整场景：1680 万（barts / liberty）到 4.5 亿（dunnottar）体素 |
| 4× 网格（VAE 输入） | 266 × 266 × 64（450 万格，每格 3.5–7 cm） | 例 bannerman 100 × 116 × 64；dunnottar 278 × 398 × 64（700 万格） |
| 潜变量（8×） | 133 × 133 × 32 | 例 bannerman 50 × 58 × 32 |
| 坐标 | 模型帧 z-up，原点 = 区域 / 场景框最小角；细体素 (i, j, k) 覆盖 [i VS, (i+1) VS)；射线由 `model_rays_torch(R, T, jitter, origin_z, k_M)` 变换，与 V2 训练数据相同 | 同 |

## 3. 输入语义（`seen2scene.v3.partial_tsdf`）
- 细体素三态（冻结规则）：BAND = cnt > 0（优先），EMPTY = 挖空且无 BAND，UNKNOWN = 其余；`input_volume` 的 UNKNOWN 哨兵 −TRUNC 不作为距离使用。
- 4× 聚合（每 4³ 块）：`sdf` = BAND 体素上 clamp(带值, ±TRUNC)/TRUNC 的均值（int8 × 127，无 BAND 为 0）；`band`、`free` = 占域内体素的比例（uint8）；`dom` = 域内体素比例。高端填充到 8 的倍数（域外），4× 网格为偶数维。
- 重建目标：BAND if band > 0，else EMPTY if free ≥ 0.5，else UNKNOWN；sdf 只在 BAND 格回归；域外格不计入损失。
- 缓存核对：每个状态由重放的 D100 融合计算的 V2 粗网格与 E2 分片中存储的粗网格**逐位相同**（同一历史、无未来记录），否则 STOP。

## 4. 张量形状
| 张量 | 形状 |
|---|---|
| VAE 输入 x / valid | `[1, 3, X4, Y4, Z4]` / `[1, 1, X4, Y4, Z4]`（float32，valid 作第 4 输入通道） |
| μ / logvar / valid8 | `[1, 16, X4/2, Y4/2, Z4/2]`；logvar 截断到 [−10, 10] |
| Decoder 输出 | `[1, 4, X4, Y4, Z4]` = [sdf_hat, logits EMPTY / UNKNOWN / BAND] |
| 射线 | o `[V, 3]`，d `[V, 7296, 3]`，tcap `[V, 7296]`（模型单位） |
| cond | `[V, 76, 64, 128]` = 11 投影通道 + 64 射线上下文 + 1 有效列 |
| Flow 状态 / 速度 | `[V, 2, 64, 128]` |

## 5. 实现决策（含依据）
| 决策 | 依据 |
|---|---|
| 4× 聚合而非 2× | 2× 缓存约 60–80 GB（压缩）且 GPU 激活 ×8；4× 实测单状态压缩 0.25–0.7 MB，MP3D 150 状态 103 MB |
| 整图卷积、不分块 | 4× 网格最大 700 万格；G0 实测显存（见 G0 报告）；无拼缝 |
| 潜变量 8× 一级下采样 | 与 V2 F0 分辨率相同；Cross-Attn token 预算可控 |
| 缓存核对改用 `structure_lean` | 冻结 `structure()` 用 float64 全图，MP3D 峰值 > 20 GB；lean 版本只在 cnt > 0 的体素上用 float64，含 0.999 TRUNC 边界规则，在 MP3D 7y3sRwLe3Va_L0/rnv_p93 t = 0–2 与原函数逐位相同 |
| 样本流 | V2 / E2 规则（每卡每域 1 状态 8 视角，陈旧度 Δ ≤ 9 不跨刷新，t ≤ 20 权重 ×2） |

## 6. 失败与修复（G0 期间）
| 问题 | 修复 |
|---|---|
| 缓存构建 OOM（每卡 3 进程） | 逐量池化、半精度指示量、`structure_lean`（偏差 #5） |
| 第一版 `structure_lean` 忽略 0.999 TRUNC 边界 → MP3D 分片核对 STOP | 按冻结规则在 cnt > 0 体素上用 float64 分类；逐位核对通过 |
| 重写时误删 `aggregate_fusion` | 恢复；重启构建 |
| CPU 上 `avg_pool3d` 无 half 内核（单元测试） | CPU 用 float32 |
| 单元测试中 FM 梯度为 0 | V2 U-Net 输出卷积零初始化（速度恒 0），测试改为扰动输出卷积后检查；训练一步后自然非零 |
