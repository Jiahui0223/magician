# EXP6 V5.2 —— 实现架构（Idea 2：Ray Belief → Camera Utility）

代码：`seen2scene/futureview/v52_model.py`（模型）、`seen2scene/futureview/v52_score.py`（Camera Score）、`experiments/exp6_v5_2/scripts/v52_core.py`（规划器挂点）。V5 / V4 代码与权重不改。

## 1. 数据流
```
真实已执行观测 → P1_D100（网络输入、结构 EMPTY / UNKNOWN / BAND、粗网格）+ P1_S10（band_dil 新颖性）
P1_D100 粗网格 → V5 3D U-Net（继承）→ F0 / F1 场景记忆
候选相机（原生成器 / 合法性）→ 7,296 条射线 o, d, tcap, dz, 11 投影通道
  → V5RayDecoder（继承：Ray Query + Local / Coarse 检索 + 交叉注意力）→ z_r
  → 每条射线 64 个有序区间 I_i = [t_i^-, t_i^+)，log1p(t / 16 VS) 均匀，t ∈ [tcap · 0.5 / 70, tcap]
  → SurfaceMLP（继承）b_r(t_i)：多表面边际（Δ = 8 VS 邻域存在表面）
  → HazardHead（新）ρ_ri = softplus(MLP[W_z z_r + W_f f_ri])，f = [PE(log t), t / tcap, log Δt, b（截断梯度）, logit b,
       此前 b 的最大值, 此前 b 质量, 区间中点的 D100 粗格 (EMPTY, BAND, 域内) 比例, 在网格内]
  → h_ri = 1 − exp(−ρ_ri Δt_i / S0)，log T_ri = −Σ_{j<i} ρ_rj Δt_j / S0，p_first = T h，p_nohit = Π(1 − h)（log 空间，和恒为 1）
  → S_visible / P_surf / P_space（world 体素 / 粗格，去重）→ Score → 规划器原选点规则 → 执行真实观测 → 更新 P1
```
没有 Flow 分支、没有 K = 4 采样、没有深度图、没有 DistP1 调用（守卫计数，G0 / G3 中均为 0）；预测从不写回 P1。

## 2. 语义
- b 不是首中 PMF、1 − b 不是 FREE；b 以截断梯度进入 hazard，首中损失不能改变"墙后存在"的 b。
- h 是条件命中概率；p_first 是首中 PMF（+ p_nohit = 1）；T 是到达区间前未被挡住的概率。

## 3. Camera Score（`v52_score.py`）
- **S_visible**：每区间的首中质量均分到 8 个子点 → 细体素（VS 网格，与 DistP1 相同）；排除域外、S10 `band_dil`（DistP1 新颖性规则）、D100 已确认 FREE（q_valid，被排除的质量单独报告）；权重 = DistP1 的 min(1, (z / z0)²)；同一 (射线, 体素) 内求和、跨射线取 max（去重），再对体素求和。不乘像素面积，避免重复距离补偿。δ 首中极限下等于 DistP1（单测）。
- **P_surf**：b · (1 − T) · min(1, Δt / Δ) · w_dist · κ，κ = exp(−max(0, t − t̄) / 128 VS)（t̄ = 期望首中深度）；同样的域 / 新颖 / FREE 掩码，同体素取 max。启发式联合代理，不是精确 P(表面 ∧ 被遮挡)，**不计入可见覆盖**。
- **P_space**：D100 粗格（8 VS）中 UNKNOWN 格（域内 ≥ 0.5、已观测 < 0.5）；frontier（26 邻域有 ≥ 25 % 已确认 FREE 的格）权重 1，其余 UNKNOWN 权重 0.1，已观测 / 域外 0；值 = 区间中点生存概率 T_mid × 权重，同格取 max。不用 1 − b；被预测遮挡的格自然 → 0；只用已执行观测的 FREE，不做未来可达性推断（保守近似）。
- **归一化**：每项除以 max(候选集最大值, floor_k)，floor_k = 0.05 × Train-6 S-c 每状态最大值的中位数（训练后标定一次，冻结）。
- **Score** = Ŝ_vis + λ (0.5 P̂_surf + 0.5 P̂_space)，λ ∈ {0, 0.1}，同一权重。

## 4. 训练（`train_v5_2.py`，`configs/train_v52.json`）
L = L_surface（V5 多表面 BCE，三组均衡）+ β L_firsthit（−log p_first[i*] 或 −log p_nohit；t* 来自分片），β = 1；Stage H 2k 只训 HazardHead，Stage J 4k 小 lr 解冻 encoder + decoder；固定 EMA@6000。
