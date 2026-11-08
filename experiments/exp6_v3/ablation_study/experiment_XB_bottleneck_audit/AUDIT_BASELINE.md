# Experiment XB — 基线审计（AUDIT_BASELINE）

只读审计，2026-11-08 16:20–17:00。每条都对应到代码、配置、日志或 checkpoint；没有核对到的地方标明。被审计系统：V3-T1@40K（`639c9d3a…`）+ D1，XA-SR（`60069dc4…`），XA-MI（`71825e2f…`）。阅读的材料：`CVPR_MODEL_AUDIT.md`、XA `FINAL_REPORT.md` / `FROZEN_PROTOCOL.md` / 配置与结果表、`seen2scene/v3/{model,vae,cross_attention,flow_unet,losses,data,partial_tsdf,predictor,trainer}.py`、`seen2scene/v3/xa/*`、`experiment_D_h3_lookahead/{scripts/d_policy.py,configs/d_config.yaml}`、`H/scripts/h_policy.py`、`task2/scripts/t25_out_cov.py`、T1 与 XA 的运行清单。

## 1. D100 / S10 / R10 的实际含义

| 记号 | 含义 | 代码 |
|---|---|---|
| **D100** | P1 的 BAND 层用观测的**全部有效深度像素**反投影写入（`mask ∧ clamp(z, 0.5, 750) < 70`），EMPTY 层用全部深度雕刻（不变）。是 V3 编码器输入和训练缓存的融合方式 | `task3_full_depth_p1/scripts/p1_full.py`（`P1FusionFull.integrate_record`、`full_points`） |
| **S10** | 模拟器按 MAGICIAN 规则每帧抽取的 **10% 表面采样点**写成的稀疏 BAND 地图（原始 P1）。部署时 DistP1 的 `band_dil` 和 D1 的 `value_cov` 新颖性判定（KD-tree，ε = 1.0）都用它 | `exp3b/m_p1.py`（`integrate_band` 用 `rec["sample_points"]`）；`p1_full.patch()` 的 b2 模式：编码器看 D100，打分用 S10（`T3_P1_MODE=b2`，`v3_out_planner.py`） |
| **R10** | 规划器每 10 步（t ∈ {0, 10, …, 90}）重新编码场景并重新抽 K = 4 张噪声图；中间 9 步复用 `h` 与噪声，只重算候选的投影通道。训练用陈旧度 Δ ∈ [1, 9]（p = 0.5）模拟 | `task1/scripts/v2_planner.py`（`refresh`，`LC.refresh_time`）；`data.py`（`sample_batch` 的 `dlt`、`stale_cap`） |

室内部署用局部 P1（n_win = 5 的区域），每 10 步切换区域（`REFRESH = 10`，`TSDFCache.refresh`）。

## 2. 3D VAE 如何编码 EMPTY / BAND / UNKNOWN

- 输入 4× 网格 4 通道：`sdf`（BAND 格子的平均带符号距离 / TRUNC，int8 量化到 ±127，非 BAND 为 0）、`band`（4×4×4 块内有 BAND 的细体素比例）、`free`（块内 EMPTY 且非 BAND 的比例）、`valid`（块内域内比例 > 0）。**UNKNOWN 没有显式通道**：`band = 0 ∧ free = 0 ∧ valid = 1` 就是 UNKNOWN（`partial_tsdf.aggregate` / `unpack`）。
- 域外（`valid = 0`）的所有通道被乘零；潜变量 `μ, logvar` 也乘 `valid8`（`VAEEncoder.forward`）。
- 重建目标（仅训练）：BAND 若 `band > 0`，否则 EMPTY 若 `free ≥ 0.5`，否则 UNKNOWN；sdf 回归只在 BAND 格子（`partial_tsdf.recon_targets`）。所以解码器把“free 比例 < 0.5 且无 BAND”的格子当作 UNKNOWN 学习。
- 编码器输出 μ、logvar 各 16 通道 @ 8× 体素（stride-2），`logvar` 截断到 [−10, 10]。

## 3. μ、logσ²、KL 的梯度路径

- 流匹配条件只用 **μ**：`LatentRayEncoder` 沿射线采 μ，T1 的 `SceneMemory` token 也是 μ；`L_FM → μ`（经编码器）。
- `L_recon`：`z = μ + exp(½ logvar)·ε` → 解码器 → 类别 CE + sdf L1，梯度到 μ 和 logvar。
- KL：`½(μ² + e^{logvar} − logvar − 1)` 在有效 8× 单元上求均值，再对通道求均值；β 从 0 线性升到 0.02（5,000 步）后恒定。
- `logvar` 只从 `L_recon` 和 KL 得到梯度，不影响推理（推理只用 μ）。代码：`vae.py`（`reparameterize`、`kl_per_unit`）、`losses.py`、`trainer.py` 第 257 行的损失组合。

## 4. 三种 Query 的实现（`seen2scene/v3/xa/queries.py`）

| Query | 数量 | 特征 | M 记忆的局部采样范围 |
|---|---|---|---|
| 2D 单元 | 16 × 32 = 512（填充列的 48 个被屏蔽，有效 ≤ 464） | 单元平均射线方向、相机位置的 Fourier 编码 | 单元平均射线的 [0, 量程] 均匀 16 点 |
| 代表射线 | 每单元 1 条：UNKNOWN 长度 `L_u = ch[6]·tcap` 最大的射线（并列取最靠近单元中心的） | 64 维射线特征、11 个投影通道、单元内偏移、是否有 UNKNOWN、方向 / 相机位置 / UNKNOWN 包络中点的 Fourier 编码 | UNKNOWN 包络前后各放宽一个 4× 格 |
| UNKNOWN 区间 | ≤ 512，层序选择（先每条射线的第 1 个区间，再第 2 个…） | 代表射线的特征 + 区间起止、长度、占比、次序 one-hot、区间数、是否回退 | 区间前后各放宽一个 4× 格 |

区间来自 4× 编码器输入网格（`1 − band − free ≥ 0.5` 的连续格子，沿射线每 2 个细体素采样，裁到 `t_known` 之前和投影通道的 UNKNOWN 包络内，每条射线 ≤ 4 个；找不到时回退为整个包络）。工程检查记录：室外约 76% 的有 UNKNOWN 的射线回退（近裁剪处的小段不足一格）；射线覆盖率 100%。多区间在注意力后按射线取平均写回单元。

## 5. 注意力残差是否参与速度预测

是。两种注意力都在 U-Net 内部、速度输出之前：

- T1：`FlowUNet.forward` 在 `mid[1]`（瓶颈自注意力）之后 `h = self.xattn(h, *mem)`，再进 `mid[2]` 和整个上行路径 → `out`（速度）。
- XA：`XAUNet` 在下行路径第 2 级（16 × 32，192 通道）的第二个残差块之后 `h = self.xattn(h, *mem)`，再 `hs.append(h)`，所以增强后的特征同时进入 skip 连接和瓶颈。
- 两者的 `W_o` 零初始化，初始时速度与 T0 逐位相同（工程检查 `cond_equals_T0` / `velocity_equals_T0_at_init` 为 True）。训练后 `W_o` 非零，但 **`W_o` 的大小不能说明模块的有效贡献**，这正是 Task A1 / A2 要测的。
- 每个 Euler 步（10 步）、每个样本（K = 4）、每个候选都经过注意力；记忆 token（T1 的 `SceneMemory`，XA 的 G / L）按状态算一次、按候选选取。

## 6. 室外训练数据与采样分布

分片 `work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100/<scene>/<traj>.npz`（`heldout = False` 为训练，`True` 为 S-c）。本次从分片直接统计（`CLS_UNKNOWN = 0`；有标签 = UNKNOWN ∧ b ≠ 0）：

| 场景 | 训练轨迹 | 状态 | 视角 | UNKNOWN 射线比例 | 有标签射线 / 视角 | 命中标签占比 | 占全部有标签射线 |
|---|---|---|---|---|---|---|---|
| bannerman | 22 | 2,200 | 15,893 | 30.4% | 2,167 | 17.9% | 20.7% |
| dunnottar | 24 | 2,400 | 17,391 | 34.5% | 2,426 | 15.8% | 25.4% |
| fushimi | 22 | 2,200 | 16,170 | 15.0% | 1,015 | 17.0% | 9.9% |
| liberty | 24 | 2,400 | 16,410 | 22.7% | 1,643 | 10.5% | 16.2% |
| pisa | 24 | 2,400 | 17,517 | 21.9% | 1,527 | 15.8% | 16.1% |
| redeemer | 22 | 2,200 | 15,506 | 17.6% | 1,268 | 7.0% | 11.8% |
| **合计** | **138** | **13,800** | **98,887** | | | | 100% |

S-c（选模）：每场景 6–8 条留出轨迹，共 42 条、31,027 个视角。

- 每个“状态”= 一条轨迹的一个时刻，平均约 7.2 个候选视角（每状态最多抽 8 个）。
- **采样**（`V3Views.sample_batch`）：每步每 rank 室外抽 1 个状态、8 个候选视角（全局 4 rank = 32 视角），状态按权重抽（t ≤ 20 的状态权重 ×2），轨迹按 rank 分片（`sorted(out_tr)[rank::world]`）。40,000 步 × 4 rank = 160,000 个室外状态抽样，相对 13,800 个状态约 11.6 轮（CT0 记录 12.9 轮，口径略不同：按视角）。
- **注意**：T1 / XA 运行清单里 `views.out_train = 25,225` 是 **rank 0 的分片**（138 / 4 ≈ 35 条轨迹），不是全集；全集是 98,887 个视角。`data_counts.out_train = 138` 是全集。

## 7. 室内 / 室外的真实采样比例

- 视角数 1 : 1（每 rank 8 + 8，全局 32 + 32），状态数 1 : 1（每 rank 1 + 1）。
- 有标签射线数不是 1 : 1：`L_FM` 按全局有标签射线总数归一化，室内一个视角的有标签射线约 2,700 条、室外约 1,000–2,400 条（上表），CT0 的 rank-0 日志给出室内占有标签射线的 **0.54**。
- `L_recon`、KL 按状态平均（室内外各半）。
- 室内：228 条训练轨迹（114 层、59 房），VALSEL 9 层；清单里 `views.in_train = 66,803` 同样是 rank-0 分片。

## 8. XA 六组的一致性

来自六个运行清单与 `results/<id>/outdoor.json`：

| 项 | 值 |
|---|---|
| 训练步数 | 全部 40,000（早停关闭；T1 原本也未触发） |
| 种子 | 全部 seed 0，4 rank，公共模块初始化哈希全部 `6a9d6071…`（= T1 / T0） |
| 数据清单哈希 | 全部 `9f432c3f…`（= T1） |
| 第 1 步损失 | 全部 3.674514578066773（= T1 的第 1 步，同一样本流、零初始化） |
| 参数量 | 23,720,486 – 23,771,750（T1 23,730,986；差别全部在 `unet.xattn`） |
| 推理预算 | K = 4、Euler 10、R10、D1（深度 2、λ = 0）、同一噪声键；注意力 token 对数 S 262,144 / M 270,336 每视角（T1 65,536） |
| 选中的 checkpoint | 全部 step 40,000 |
| 闭环 | 全部 70 / 70 + 6 / 6，重放校验全部通过，规划器加载的 sha 与选中一致 |
| 训练时间 / 显存 | 16 h 08 – 16 h 45 min；S 系列 33.3–33.4 GB，M 系列 36.6–36.7 GB |

一致性没有问题；唯一的非对称是 M 系列慢 1–3%。

## 9. D1 规划器与官方覆盖评估协议（核对）

- 候选：模拟器的合法列表（6 移动 + 4 旋转的设置网格邻居 → 未访问优先 → GT 线段碰撞，模拟器侧）。
- 一步价值 `value_cov`：K = 4 个样本解码的命中点中，离所有 S10 观测点 > 1.0 m 的，按 1.0 m 格子去重，权重 `min((z/z0)², 1)`。
- D1：对每个候选展开设置网格邻居（未访问优先，穿过已知表面带的线段丢弃），路径价值 = 每样本并集的均值（λ = 0），候选分数 = 其所有两步路径的最大值；`argmax`，并列取最小下标。
- 覆盖：模拟器记录的 MAGICIAN 官方覆盖（有上限、可下降、不取 cummax）；`t25_out_cov.py` 重放校验逐位一致；`AUC@k = trapz(C[0..k]) / k`。
- Table 1 记录 `V3-D1-H3-NORISK`：All-14 0.425 / 0.654 / 0.638 / 0.765 / 0.910（AUC@20 / Cov@20 / AUC@50 / AUC@100 / Final）。

## 10. 已知事实（供 Task A–D 参考）

- 验证集（VALSEL / S-c）40k fair ES：T0 6.37 / 6.44，T1 6.42 / 6.48，XA-SR 6.37 / 6.43，XA-MI 6.24 / 6.42，XA-MR 5.92 / 6.18。
- 训练子集 vs 留出子集的流匹配损失差距：室外 0–2%（S-c 是同场景的留出轨迹），室内 10–12%（VALSEL 是不同房子）。
- 固定状态 T1@40K（CT1）：室外训练场景 Top-1 0.504 / 漏报率 0.32，Dev-5 Top-1 0.531 / 漏报率 0.29，室内 0.467 / 0.31；误报率 0.05–0.13。
- 反事实（CT3，T1@40K）：补上漏报命中把覆盖损失从 0.155 降到 0.115（室内）、0.211 降到 0.167（室外 Dev，部分）；去掉误报几乎无效；可表达的完美预测 0.060 / 0.114。
- 既有的 P1 多孔性记录：BAND 只沿 10% 采样射线写入（S10）；D100 把 BAND 补到全部像素，但 free 雕刻在 `z − trunc` 处停下。

## 11. 本审计未核对的项

- 训练时每个场景、每条轨迹的实际抽样次数（采样器不记录；Task C 会在训练后用相同 RNG 规则重放统计）。
- 室内分片的逐场景统计（本轮只审计室外；室内只做 sanity）。
