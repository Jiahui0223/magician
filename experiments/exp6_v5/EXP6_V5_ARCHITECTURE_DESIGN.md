# Seen2Scene · V5 正式架构设计

**版本**：V5 Design v1.0（2026-11-01）  
**状态**：用户已经确认四项核心决策；本文是可实施的设计规范，不是已经完成的代码或训练结果。  
**完整名称**：**V5 — Geometry-Aware Ray-to-Scene Memory with Multi-Surface Belief and 2D Rectified Flow**。  
**代码目标目录（建议）**：`experiments/exp6_v5/`；共用模型实现在实际仓库的 `seen2scene/futureview/` 合适位置，按源码审计确定。

## 0. 已确认的决策（不要再次征询）

| 决策 | 固定版本 |
|---|---|
| Q1 几何信念 | **显式 Multi-Surface Belief**：可表达墙 A 及墙 A 后的墙 B；训练期争取获得 Train-6 场景 Mesh/多视角几何监督 |
| Q2 场景查询 | **Geometry-Guided Local + Coarse Memory**，避免对所有 3D Token 做稠密 Attention |
| Q3 初始化 | **V4 Outdoor-only EMA@40k Warm-start**，保留已学 3D U-Net 与 2D Flow，并让新中间模块逐渐接管 |
| Q4 Flow Conditioning | **Scene-Conditioned Ray Latent + Multi-Surface Belief Summary** 共同形成二维条件特征 |
| 模型任务 | 输入候选合法 Camera → 预测 K=4 Future Depth/HIT；不预测候选 Camera Pose |
| Planner | 冻结当前 Candidate Generator、S10 稀疏 DistP1、K=4、`n_win=5`、距离权重、G3 环境，不加 Exploration Memory |
| 训练域 | 初始阶段 Outdoor-only Macarons++ Train-6；MP3D 仅 Zero-Shot G1/G2/G3 评价，**不作训练/选模** |

**一句话**：V5 只重设计 Camera-Conditioned 3D→2D Decoder，并加一个明确受监督的 Full-Ray Surface Belief；3D U-Net、2D Rectified Flow、输出 Depth/HIT 和主 Planner 保持原接口。

## 1. 为什么现在做 V5：证据和边界

既有 V4（D100 网络条件，S10 DistP1）：Macarons++ AUC@100 / Final / Cov@20 = **0.710 / 0.878 / 0.556**；MP3D Zero-Shot AUC@150 / Final / Cov@20 = **0.327 / 0.495 / 0.187**。V3.1 对比 0.697 / 0.865 / 0.551（Outdoor）、0.335 / 0.499 / 0.185（MP3D）。历史 B2 Outdoor 0.730 / 0.906、Indoor 0.334 / 0.530；E2-D100 Outdoor 0.711 / 0.880、Indoor 0.364 / 0.553。历史训练预算不同，不能只用这些数值推断架构因果。

Experiment A 固定同一 V4 和 Future Depth，只把评分 P1 由 S10 换成 D100：Macarons++ AUC **0.710→0.666**、Final **0.878→0.841**（明显退化）；MP3D AUC **0.327→0.345**、Final **0.495→0.503**（小幅提升）。因此 V5 主口径选择 **D100 模型输入 + S10 评分**，不把评分修改与网络修改混合。

已有诊断指向三个研究问题（目前仍是假设，不能冒称已验证）：
1. **尺度不匹配**：V4 每条射线固定 32 个 3D 特征采样点。对于不同物理长度的 Indoor/Outdoor Ray，这可能出现近处过密、远处错过细小几何的风险；实际是否均匀采样须查源码。
2. **聚合冗余**：原 Conv1d 与 Ray-Aligned X-Attn 都处理本条 Ray 已采到的特征，新增信息来源有限。仅因“换成 Attention”不能保证提升。
3. **可见/遮挡后几何的区分**：Current Depth 只有 First-Hit，但长走廊、门、桌子后方的潜在结构仍可作为有用的条件证据。模型应学到多表面几何信念；**不能**把遮挡后表面当成当前相机已可观测 Coverage。

V5 目标是验证：**动态几何记忆检索 + 显式多表面信念**是否能够更好地为原 2D Flow 提供跨尺度、空间相关的生成条件，并改善 G1/G2/G3。它不是保证会超过 B2 的预言。

## 2. 三模块结构和物理语义

```text
真实已执行 Camera 历史深度（不包含候选未来观测）
        ↓ D100 融合
Partial TSDF P1: EMPTY / BAND / UNKNOWN
        ↓
[Module I]   原 3D U-Net（从 V4@40k 继承）
             F0 / F1 多尺度 3D Features + Token 3D Position + Obs State
                          │
候选 Camera Pose + K → 每个 (u,v) 的 Ray: o + t d
                          │
                          ↓
[Module II]  Geometry-Aware Ray-to-Memory Decoder（V5 主改变）
       Ray Query q_r → Local+Coarse 3D Memory Retrieval
                         → Geometry-Biased Cross-Attention → z_r
                         ├→ Multi-Surface Belief b_r(t)（辅助监督）
                         └→ belief summary + z_r → H_C [B,C,H,W]
                          ↓
[Module III] 原 2D Rectified Flow（从 V4@40k 继承）
             K=4 stochastic Future Depth/HIT samples
                          ↓
             原 DistP1(score_p1=S10) + 原 Candidate Set
                          ↓
             选取下一 Camera → 获取真实 Depth → 更新历史 P1
```

**术语约束**：
- `3D Scene Memory` = 当前 U-Net 输出的、空间可定位的特征 Token；**不是**长期 `Exploration Memory`，后者这次不做。
- Ray `r(t)=o+t d` 是确定几何对象；`z_r` 是确定性（给定 P1 和 Camera）的条件 Latent，**不是 VAE 的高斯采样变量**。
- `b_r(t)` 是深度位置附近存在表面的一种条件概率/信念，**不是 First-Hit Depth 的概率密度**，也不是可以对所有深度相加得到 Coverage 的奖励。
- 由 `2D Flow` 提供 K=4 随机 Future Observation；V5 **不新增**第二套独立随机 VAE 或完整三维生成器。
- V5 并不因此自动保证跨候选 Camera 的世界一致性；共享场景特征只是有益条件，不能冒充采样了共享完整 3D 世界。

## 3. Module I：3D U-Net 与历史 P1

保持 V4 的 U-Net 架构与权重、EMA、特征格式及 P1 编码协议。历史参考：F0 通道 32、下采样 8 倍；F1 通道 64、下采样 16 倍；请由当前仓库实际模型运行确认，不硬编码为真。**禁止**把已预测或 GT 候选未来 Depth 融入历史 P1。D100 由历史已执行真实 Depth 构建，区域外 / 未观察位置仍 UNKNOWN。

每个 Memory Token 包含：

\[
M_j^{(\ell)}=(f_j^{(\ell)},x_j^{world},s_\ell,m_j),\quad \ell\in\{F0,F1\},
\]

`f` 是可训练 U-Net 特征，`x` 是体素中心的世界或同一局部参考系物理坐标，`s_l` 为该级特征体素的物理尺度，`m` 为该特征支持区域里的已观测 EMPTY / BAND / UNKNOWN 比例或明确几何标记（由当前历史 P1 下采样，不从未来 GT 引入）。**UNKNOWN 不是 invalid mask**；在区域内的 UNKNOWN Token 可以提供场景先验。

MP3D 的局部窗口每 10 步重建的现有逻辑不变。Memory Token 要随着窗口移动正确更新坐标，不能将不同局部窗口坐标相加或复用旧 Key/Value。Outdoor 也使用实际场景网格的坐标转换与物理体素尺寸，避免跨尺度错配。

## 4. Module II-A：Ray Query（不作固定 32-point Feature Sampling）

每个候选 Camera、每个像素 `(u,v)` 使用 Pose 与 K 定义几何射线：

\[
r_{C,u,v}(t)=o_C+t d_{C,u,v},\quad t\in[t_{near},t_{far}].
\]

`d` 是归一化世界/局部坐标方向；近远裁剪沿用现有传感器、模拟器与 Ray 几何协议。**不要用 GT Depth 规定 `t_far`**。

\[
q_r=\mathrm{RayMLP}(\mathrm{PE}(o_C,d_{C,u,v},K, t_{near},t_{far})),\quad q_r\in\mathbb R^{d}.
\]

建议 Decoder hidden dimension 192（可按真实 Flow 接口调整）；必须保留真实米制尺度 `t` 或 `log(1+t/s0)`，同时包含归一化深度信息，不能让 8m/80m 场景失去绝对尺度。建议显式编码相机位置、方向、视角 FOV、近远裁剪与 ray 与局部网格的相对信息。现有 `Geometry.view` 中合法的解析射线状态可作为 Query 附加输入，但**不是**重新运行 32 个 U-Net Feature Samples。

## 5. Module II-B：Geometry-Guided Local + Coarse Memory 检索

对于 Memory Token `x_j`：

\[
\tau_j=(x_j-o_C)\cdot d_r,\qquad
\delta_j=\|(x_j-o_C)-\tau_j d_r\|.
\]

- `tau`：Token 沿 Ray 的前后位置；必须在合法 `[near,far]` 前方。
- `delta`：Token 到 Ray 的垂直距离。
- 局部几何候选由 `delta <= rho_l(tau)` 产生；`rho_l` 应体现 feature voxel 的物理尺寸与相机视锥，而不是固定一个同时适用于室内外的米数。

初始建议：

\[
\rho_l(\tau)=\operatorname{clip}(a s_l+\tau\tan(\theta_{pixel/patch}),\rho_{min,l},\rho_{max,l}),
\]

其中 `theta` 与 K/特征尺度相关，`a`、上限是工程超参，应在审计物理单位后冻结。远处更宽并不是无限变大；真实几何可见性不能靠简单的最近邻距离代替。

Memory 分两组：

1. **Local Geometry Tokens**：F0 和 F1 当前 Ray 附近的多尺度 Token，候选可变数量，按相对几何关系做确定性 top-k 或分层选择；建议初始上限 `F0 ≤ 24`、`F1 ≤ 12`，数值只用于显存上限，**不意味着沿 Ray 预定 36 个采样深度点**。
2. **Coarse Context Tokens**：从 F1 / 场景低分辨率空间聚合得到少量全球或窗口级场景上下文，建议 ≤8 个；用于 UNKNOWN 近邻缺乏直接证据时的语义/结构先验。它们必须来自 P1/U-Net，不得来自 GT Mesh。

变长 Token 使用 `valid mask`，无 Local Token 时使用 Coarse 或合法的 learned null token。区域外 Token 必须 mask；**区域内 UNKNOWN 不应 mask 掉**。检索候选的建索引和 gather 不能为每条 Ray 构造完整 `R×全部 3D Token` 的 Dense 距离矩阵；应利用 3D grid、空间分桶/稀疏索引或 chunked 查询。若有物理视锥/可达窗口边界，必须尊重。

动态选出的 Local Tokens `24+12` 加少量 Coarse Tokens 虽有数量上限，但与原先固定 32 个**射线深度采样位置**不同：Token 是根据实际空间几何动态检索的，而非事先选择等距深度进行 `grid_sample`。

## 6. Module II-C：Geometry-Aware Cross-Attention

选定 Memory 后：

\[
\begin{aligned}
 e_{rj}&=\phi(\tau_j,\delta_j,s_l,\text{ObsState}_j,K,\text{relative coord}),\\
 a_{rj}&=q_r^T k_j/\sqrt{d}+\operatorname{GeoBias}(e_{rj}),\\
 z_r&=\operatorname{CrossAttention}(q_r,\{k_j,v_j,e_{rj}\})+\operatorname{QueryResidual}(q_r).
\end{aligned}
\]

- 推荐 4 heads、d=192 起步；实测 VRAM/吞吐再确定。
- `tau`、`delta`、`voxel scale` 不只是位置编码，也应直接参与 Attention logit bias 或 token embedding。
- 单 Query per Ray 首版输出一个 `z_r`，与用户确认的表示一致；不引入额外 Scene VAE。
- 做可见性相关的几何编码，但不要把 `UNKNOWN` 当 `EMPTY`，也不应因为已知 BAND 前面有遮挡便直接**删掉**所有更深的 Token：V5 要预测 Multi-Surface Belief，墙后仍须允许保留潜在表面信息。
- 此模块输出张量 `[B,R,d]`，其中 `R=H_view×W_view`，历史视角约 `64×114`，以真实代码为准。查询可在 `R_chunk=256/512` 分块实现，避免大显存中间张量。

## 7. Module II-D：Full-Ray Multi-Surface Belief

### 7.1 定义和概率语义

对每条 Ray 输出条件化、可按深度查询的表面信念：

\[
b_r(t)=\sigma\bigl(\mathrm{SurfaceMLP}[z_r,\mathrm{PE}(\log(1+t/s_0)),t/t_{far}]\bigr).
\]

`b_r(t)` 是 **深度 t 附近一个有限厚度区间存在表面**的概率（数值区间厚度 `Δ` 与体素物理尺寸一致，并在训练/评估固定）；严格的点事件概率为零，不能误称为“t 精确点存在表面概率”。它不是归一化到 1 的 First-Hit PDF：在墙 A 和 B 附近可以同时高。场景不确定度需要校准；不能默认概率自然准确。

**这里允许在任意深度 `t` 查询 MLP，这和旧版沿 Ray 固定 32 个位置从 F0/F1 `grid_sample` 有根本区别**。几何显存访问发生在第 5 节的空间相关 Token 检索。Belief 监督和 Summary 所需的数值深度查询，是对一个已汇总的 `z_r` 评价函数，不是旧的 32-point 3D Feature Sampling。

首版建议使用上面的**隐式查询型 SurfaceMLP**，而非同时引入完整 3D TSDF、沿线另一个 VAE/Flow、多个生成世界或全局 Occupancy Volume。关于“单个 `z_r` 是否足以表达多表面”的容量问题要做诊断；如果严格 Multi-Surface 测试不通过，报告失败，不要悄悄将 Multi-Surface 退回 First-Hit-only。

### 7.2 Belief Summary 接入 2D Flow

`H_C` 要同时包含 `z_r` 与 Belief Summary。建议从 `b_r(t)` 在 Ray 合法深度区间上计算少量统计：例如不同深度区段的局部表面概率、有效分布的均值/离散度、远近多峰信息，并显式编码深度区段的物理尺度。首版可在 **16 个对数/分层深度查询位置上评价 SurfaceMLP**、再用小 MLP 或 Attention Pooling 压缩得到 `[B,R,d_s]`；这 **16 个是 Belief Function 的数值探针，绝不是 16 个 U-Net 3D Feature 采样点**。如经审计已有更优稳定积分实现，可以替代，但必须记录差异。

\[
H_C=\mathrm{CondAdapter}\bigl([z_r,\mathrm{Summary}(b_r)]\bigr)
\in\mathbb R^{B\times C_{old}\times H\times W}.
\]

`C_old` 严格取 V4 Flow 的真实条件接口通道数。选择 `concat + small projection + LayerNorm / scale adapter`，保留空间网格布局和可用的原相机条件通道，不修改 GT Depth/HIT 含义。Belief Summary 对 Flow 输入是否有效，需要通过梯度和破坏性检查确认；不能只放一个从未影响 Flow 的辅助头。

### 7.3 训练标签：明确由训练场景几何提供

已确认 **显式 Multi-Surface**，所以必须提供墙 A 后墙 B 的真实监督来源：

**首选**：Macarons++ **Train-6** 场景的训练期 Mesh/可访问的多表面射线求交（包括遮挡后的多重表面），只用于生成训练监督缓存。真实 Mesh 内容**绝对不得**进入 V5 推理时的 Feature、Memory、Query、Planner，不能使用 Dev-3/Test-3 或 MP3D Eval 的 GT Mesh 训练。

对每条训练 Ray，在合法 `[near,far]` 中获得排序后的表面交点 `t_1,...,t_m`（双面处理、重复交点去重、掠射/数值容差、网格外 mask 等必须审计）。以训练期固定几何厚度 `Δ` 构造位置目标：

\[
y_r(t)\approx\mathbf1[\min_k |t-t_k|\le \Delta/2],
\]

或者使用归一化软核以减小体素离散噪声。`t>t_1` 的合法后方交点必须为正例，不能只标第一面墙。若 Mesh 不完整、超出合法场景范围或部分射线没有可信完整几何证据，相应深度区间标为 **unsupervised**，**不能默认负例**。目标是“表面邻域信念”，不是非封闭 Mesh 的实体体素 Occupancy。

若 Train-6 的完整 Mesh 求交功能不可得，应先审计训练可用多视角证据能否为墙后提供**有明确可观测支持**的标签。不能用任意一张 First-Hit Depth 把其后全当作空闲或“无表面”，也不能由模型自身预测制造假 GT。确实没有可靠 Multi-Surface 监督，则**报告关键阻塞并请求用户决定是否调整已确认的 Q1**；不得暗中降级。

推荐离线缓存训练射线与稀疏标签：例如每训练状态抽若干目标视角/有效 Rays，每 Ray 在交点附近、空白区间和遮挡后区域分层采样正负深度，用实际物理尺寸定义正例邻域；标签需按有效几何覆盖 mask。不要在每个 40k/12k 步训练时对所有 7296 Ray × 完整 Mesh 实时求交。场景 ID、姿态 hash、相机 K、近远范围、交点 hash、Mesh SHA、P1 历史观测时刻必须可追溯。

### 7.4 不能误用的“可见性”

`b_r(t)` 表达多表面存在的**边缘概率**，不能仅靠 `b` 在不同深度简单累乘，就声称获得真实联合 First-Hit 分布。位置之间的几何相关性不可凭空获得。首版由 2D Flow 负责生成当前视角 First-Hit Depth/HIT；它可以把 `b` 作为带有遮挡后信息的条件先验。

因此：
- 有 GT 首次 Hit 时可监控 Belief 是否在附近高概率，但不可强迫所有更深 `b` = 0。
- No-HIT 与超出有效 Range 不等于后方所有空间 EMPTY。
- 一条 Ray 的遮挡后几何预测**不计入**当前相机可见 Coverage，不写入 P1，不直接加到 DistP1。
- 未来探索潜力、Deferred Exploration / Exploration Memory 暂不属于 V5 主实验。

## 8. Module III：保留原 2D Rectified Flow 的科学理由

将所有 `H_C` 输入**原 2D Flow**，仍然预测 `Depth/HIT` 的 K=4 随机样本。这里的两层表示分工：

1. `b_r(t)`：给每条 Ray 明确的多表面几何候选，包含遮挡后的候选结构；它不负责独立抽样整张 Depth。
2. `Rectified Flow`：学习相邻 Rays 的**联合** Future Depth/HIT 分布与 First-Hit 观测结果，让一面门、墙在整个 2D 图上尽可能连贯。

因此不新增 VAE 采样、第二个 Ray Flow、独立随机完整 TSDF 生成。**保留 Flow 并不自动构成论文新颖性**；可能的贡献是 Partial-TSDF-conditioned Ray Memory Query + Multi-Surface Prior 如何改善 Flow 的生成条件与 NBV。

原 Flow Time Sampling、Rectification Target、Euler steps、深度/HIT 编码、mask、输出范围、K=4 与随机种子接口不变；除非源码审计发现严重错误并在偏差记录中报告。

## 9. Warm-start：解决“原 V4 Decoder 与新 V5 Decoder 不兼容”

**必须明确：不能直接删掉旧 Ray Encoder 后把随机新 Query 接给预训练 Flow，否则 Flow 的条件分布突变，Warm-start 几乎失去意义。**

### 9.1 Checkpoint

使用 V4 Outdoor-only EMA@40k `work_dirs/exp6_v4/checkpoints/final.pt`，本地 SHA 以仓库核对（历史前缀 `1b31afff...`）。只拷贝权重，不覆写原文件，注明所有继承和随机初始化层：

- 继承：V4 3D U-Net、原 2D Rectified Flow、所有可兼容归一化与输出层。
- 新建：RayEmbedding、Local/Coarse Memory Retriever、Geometry-Attention、SurfaceMLP、BeliefSummary、CondAdapter。
- 旧版 Conv1d Ray Encoder、V4 Ray-Aligned X-Attn：**只允许作为临时冻结 Teacher 的训练辅助**。正式 V5 推理路径不能继续调用这两个模块或旧 32-point 3D Sampling。新模块输出 `C_old` 条件通道与相机像素布局完全匹配。

### 9.2 Teacher-assisted feature migration（训练辅助，不是最终架构）

把旧 V4 固定为 Teacher，针对**同一个 Train-6 历史 P1 与目标 Camera**获得旧条件特征 `H_old`；V5 新 Query 得 `H_new`。初始期先让两者特征分布可兼容：

\[
\mathcal L_{bridge}=\|\mathrm{normalize}(H_{new})-\mathrm{normalize}(H_{old})\|_2^2
\]

需监控原始未归一化 feature scale，防止仅匹配归一化特征但 Flow 接收到不兼容幅值。建议初期将 3D U-Net / Flow 冻结，新 Decoder 接受 `L_bridge` + Multi-Surface Loss；之后衰减 `L_bridge` 至零，逐渐解冻 Flow 与 U-Net，让 V5 不被旧 Decoder 上限锁死。若 Teacher 有系统性错误，不能把其错误 Depth 当作 Multi-Surface GT；Teacher **只用于条件特征迁移**。

**V5 最终部署不依赖 Teacher；不存在伪装为新架构而实际仍暗中调用旧 32-point Ray Encoder 的情况。**

### 9.3 推荐初始训练预算（预注册、可因资源审计调整）

- `V5-Warm-Bridge`: 2k steps，冻结 U-Net/Flow，仅训练新中间模块与几何头；早期 `lambda_bridge` 高、末期下降；`lambda_surf` 非零；训练期有完整 Train-6 多表面标签。
- `V5-Warm-Joint`: 10k steps，适当降低旧 U-Net/Flow 的 LR，新 Decoder/Belief Head LR 较高；开启原 Flow Loss，`lambda_surf` 适度加权，`lambda_bridge=0` 或逐渐退为 0；整个路径必须获得有效梯度。
- 全阶段共 **12k 新训练 steps**。这是**推荐的初始预算，不是已经实验确认的最优数**。旧 V4 同 checkpoint 继续相同 12k、相同 Outdoor 数据顺序和预算作为 `V4-Continue12k` 时间/优化对照；这能减少“只是训练更久”的混杂。V5 多了 Mesh 辅助监督，因此对照只能用于比较整个系统，不单独证明 Geometry Query 的纯架构效应。
- 实际学习率、Batch、EMA、Scheduler 请按真实 V4 工程和显存审计后冻结；建议新模块 LR 约 `1e-4`，已训练 U-Net/Flow `1e-5` 量级（非强制常数），禁止沿用 V4 已退火到零的旧 Scheduler 状态而不记录。

## 10. Loss 组合与归一化

\[
\mathcal L=\mathcal L_{\mathrm{Flow/V4}}+\lambda_{surf}\mathcal L_{surf}+\lambda_{bridge}(s)\mathcal L_{bridge}.
\]

- `L_Flow/V4`：原完整 Flow Velocity / HIT 相关原损失组合，以真实代码为准（不重复加独立 GT HIT 损失）。
- `L_surf`：Multi-Surface 目标上的 masked BCE 或带正负例平衡的 BCE（可用 focal 但不要同时引入多种新权重）；按照**有效训练 Rays 与有效深度查询数**归一化。首个与后方表面应分别报告、避免首个表面数量主导；空/满不平衡需日志监测。
- `L_bridge`：仅新 Decoder 训练早期，按条件像素有效 mask 归一化。必须有清晰消退规则。
- 建议从 `lambda_surf` 的小量值开始，根据各项**梯度尺度**而非原始 Loss 数值设定；避免它压倒 Flow 主目标。权重/超参要在主对比前冻结，不根据 Test-3/MP3D G3 反复选择。
- 监督数据：`L_surf` 可以用 **Train-6** Mesh，`L_Flow` 用 Train-6 真实目标 Depth/HIT，`L_bridge` 用同一历史状态的冻结 V4 特征。MP3D Zero-shot、Mac++ Eval Mesh 一律不可用于反向传播。

## 11. 风险、检验和失败定义

| 风险 | 必须做的检验 |
|---|---|
| Multi-Surface Label 不可信 | 全部训练 Mesh 的多重求交与 Depth 第一命中抽样一致性；去重、开放网格、射线截断和可信 mask；展示墙后第二个表面标签 |
| 仍然隐式调用 32-point Sampling | 推理 Profile/调用图中不得调用旧 `RayEncoder` / 旧 ray `grid_sample(S=32)`；仅训练 Bridge Teacher 可使用 |
| `z_r` 将不同表面过度压缩 | 检查 GT 有≥2 surfaces 的 Ray 的后方 Surface Recall、Belief Brier/ECE、按深度分组结果，不只 First-Hit MAE |
| 未知区域被当空 | UNKNOWN Token 被保留；无可靠 Mesh 区间 Mask 不作负例 |
| Attention 忽略三维位置 | 对 Pose/Ray 更改测试、打乱 token 位置的破坏性测试、相对距离偏置统计 |
| Flow 接到错误幅值的条件特征 | `H_old/H_new` 均值、标准差、余弦、下游 loss、NaN/Grad；训练 Bridge 与 Joint 的连续性 |
| 只学会抄旧 Teacher | `lambda_bridge→0`；Joint 后 Multi-Surface 得到独立改进；G1-G3 实测 |
| 带来不可承受推理延迟 | 同步比较 Encoder/Query/SurfaceMLP/Flow/DistP1 的分段 walltime、VRAM、GPU-h；维持真实候选数和 K4 |
| Surface 好、NBV 差 | 固定相同模型条件/评分协议，G1、G2 真选中增益、G3 AUC/Final/Cov@20 全部报告，不能只靠 geometry head 证明可用 |

## 12. 评估协议（不因 V5 改变）

**G0 正确性**：加载和冻结旧 V4 权重的继承清单；旧 V4 reference replay；输入/输出 shape、坐标、mask/UNKNOWN、Pose dependence、梯度、内存、No-GT-leak、Teacher 推理零依赖。对 Surface Label 做几何可视化与遮挡后的 Positive 检查。

**G1 生成**：Original Flow Loss / HIT Brier / Fair ES / Known、Unknown、Truly-new Depth MAE / K4 diversity / 校准，按同一固定状态同一 P1 条件、有效 Ray mask、Seed 评估；Belief Extra：GT first/second/third surface recall、precision、Brier/ECE、depth scale buckets、遮挡后多表面可视化。Belief 指标不要拿独立样本和不同 Ray mask 生硬比较。

**G2 排序**：冻结 `n_win=5`、DistP1=S10、同一 candidate pool、同一历史状态，报告 **真正被选 Camera 的 one-step GT Gain（pp）**、Top-1、候选排列相关，Regret 可作附录。区别训练中已有 surface GT 与评价中 GT Gain：测试 GT 只能在评价端读。

**G3 全闭环**：Macarons++ 6 scenes ×5 starts ×100 steps = 30 EP；MP3D 5 eval floors ×2 starts ×150 steps =10 EP。主表 `AUC@100/@150, Final Coverage, Cov@20`、MP3D stagnation、按场景/楼层的 paired bootstrap 95% CI、同种子同动作规则；基线 MAG-G / MAG-B / V2 Original / B0 / B2 / E2-D100 / V3-T0 / V3-T1 / V3.1 / V4-A0 和（单列诊断）V4-A1。V4 continued 12k 若执行成功也列作增量预算对照。**Test-3 与 MP3D 楼层已在历次研究中使用过，不得称盲测**。

**模型选择**：使用训练前冻结的最后 EMA/checkpoint 规则；S-c 仅监控且要查它与 Train-6 是否场景隔离，不能对 Test-3/MP3D G3 反复调参数。若需要真正未见场景的论文最终测试，另设后续独立 blind benchmark，不把这次预言成盲测结论。

## 13. 创新性定位和不在 V5 范围

V5 当前能提出的研究假说：
1. 不依赖固定 32 点深度采样的**尺度感知 Camera-conditioned Scene-Memory Retrieval**，更适于 Indoor/Outdoor 不同空间长度与复杂 UNKNOWN 分段。
2. **Multi-Surface Geometry Belief** 表达可见墙后潜在结构，并与 `Scene-Conditioned Ray Latent` 共同增强原二维 Depth Flow 的条件。
3. 保留标准 Depth/HIT → DistP1 → NBV 闭环，观察几何生成收益能否转换为探索收益。

上述创新性是**待证假设**：Ray Query/Attention、隐式几何场、Rectified Flow 都有前人工作。不能宣称组合本身必然 CVPR novelty；需要严格现有方法对比、效率和消融证据。

**不在本版本**：探索记忆 / Deferred Exploration、长期规划/Beam、候选 Camera 生成网络、Full 3D TSDF 完成、共享随机世界、去掉 2D Flow、全局 Dense Attention、P1 的 Sparse/Dense Scoring 新搜索、n_win 修改、MP3D Fine-tuning、追加 V3.1 训练。

## 14. 完成判据

1. `V5` 真实推理时以 `D100 P1` → U-Net → **新 Query** → `b_r`/H → **原 Flow** → Depth/HIT；不调用旧 32-point U-Net Feature Sampler；K4 与 `DistP1=S10` 不变。
2. 训练期可靠 Multi-Surface 标签来源与遮挡后正例已审计。缺少可靠标签不得伪装多表面监督。
3. 新条件适配与旧 Flow Warm-start 有可复现实测；无 GT Future Geometry 泄漏；G0 通过。
4. Outdoor-only Train-6 完成真实训练、冻结 checkpoint；对 MP3D Zero-Shot 和 Macarons++ 双域 G1/G2/G3 提交**真实**表格与逐场景统计。
5. 给出 V5 是否超越 V4/B2/E2 的证据边界、成本、风险和代码 SHA；不存在事后选最佳测试 checkpoint 的行为。

**本文件描述设计与验收，不意味着已有远端仓库修改或训练完成。**
