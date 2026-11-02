# V5.2-M 架构（Idea 2，完全从零训练）

代码位置：
- 模型：`seen2scene/futureview/v52m_model.py`
- 打分适配：`seen2scene/futureview/v52m_score.py`
- 配置：`configs/model_v52m.json`
- 参数实测：`PARAMETER_AUDIT.md`（总计 26,999,143）

```
D100 coarse grid (EMPTY, BAND, dom ×255)  ──► 3D U-Net encoder (V5 Encoder3D 结构, 32/64/96, 随机初始化)  ──► 多尺度 memory F0 / F1
                                                                                                                │
每条射线 (o, d, tcap, 11 维 D100 投影通道) ──► Ray-to-Memory Query（沿射线 24 + 12 个 memory token，4 头注意力）──► z_r ∈ R^128
                                                                                                                │
整幅视角 64 × 114 射线网格：[Linear(z_r)→64 ⊕ 11 通道 ⊕ valid] ──► 2D Spatial U-Net [96,192,384,384]×2 ──► c_r ∈ R^64
                                                                                                                │
每条射线 64 个深度区间（16 vox 起，近细远粗；与旧 V5.2 相同）的区间特征 f_{r,i} (29 维)：
   深度 Fourier PE、t/tcap、log dt、D100 解析结构 8 维、coarse 观测 4 维、射线类别 one-hot 3 维
[z_r ⊕ c_r] ─Linear→ 192 （逐射线广播） + Linear(f_{r,i}) ─► 6 × (LN → depthwise Conv1d k5 沿深度 → 残差; LN → FFN×4 → 残差)
                                                                                                                │
                                                     ┌──────────── head: 表面 logit b_{r,i}  +  原始 hazard raw_{r,i}
                                                     ▼
raw += softplus(known_bias) · 1[区间含可信 TSDF 零交叉 t_known]      （BAND / 已知表面：软约束）
ρ = softplus(raw);   a_{r,i} = ρ · Δt · (1 − free_frac_{r,i}) / S0  （D100 FREE：在透射率之前去掉）
T_i = exp(−Σ_{j<i} a_j);  p_first_i = T_i (1 − e^{−a_i});  p_nohit = T_N    （截断：t_max 之外的质量归入 no-hit）
                                                     ▼
S_visible / P_surf / P_space（V5.2 打分器原样：世界体素去重、DistP1 距离权重、S10 band_dil 新颖性、D100 EMPTY q_valid）
score = S_vis/floor + λ · P_potential（λ ∈ {0, 0.1}，η = 0.5，floor 由 Train-6 S-c 校准）──► B2 规划器选 NBV（K=4、n_win=5、R10）
```

## 物理约束怎样进入网络
- **FREE（D100 已证实的空）**：每个区间计算 `free_frac`，并直接乘进光学深度，所以 FREE 内**不可能出现首中质量**，也不会“吃掉”后面的质量。
  - 取值：区间与 `[t_near, min(d_lo, t_known, t_max))` 的重叠比例，再乘以区间在 domain 内的比例（4 个子点处的 coarse dom 分数）。
  - 对照：若先算透射率、再删除 FREE 区间，FREE 中的虚假墙会吃掉后面墙的首中质量。G0 测试把这两种顺序做了对比：正确顺序下墙 A 拿到解析值 0.956，错误顺序只剩 0.050。
- **BAND / 已知表面**：只在 t_known 所在区间加一个可学习的非负偏置（初值 softplus⁻¹ 后约 3）。
  - 不强迫每个 BAND 体素都命中，网络可以学到“穿透”。
  - 是否违反由 holdout 的 KnownPassThrough 监测，并计入选择分数。
- **UNKNOWN**：没有任何强制，完全由 3D memory、射线 token 和 2D 上下文预测。
- **截断**：`t_max = min(出网格, tcap)`，之后的区间进入 `beyond` 特征，质量自然流向 no-hit。

## 相对旧 V5.2（2.67M）的改变（联合干预，不做单变量归因）
| 方面 | 旧 V5.2 pilot | V5.2-M |
|---|---|---|
| 初始化 | V5@12k warm-start + hazard 6k | 全部随机，从零训练 70k |
| 2D 上下文 | 无（射线独立） | 2D Spatial U-Net 覆盖整幅 64×114 射线网格（22.5M） |
| 深度轴 | 逐区间 MLP hazard head（13,953 参数） | 6 层 depthwise-conv + FFN 的 1D decoder（1.83M），沿深度传递信息 |
| FREE | 透射率前去除 | 同左；另加 domain 子点比例和 t_max 截断 |
| 已知交叉 | — | 软偏置 known_bias（可学习） |
| 训练分布 | label 射线 | label 射线 + 128 UNKNOWN 补采样 + 128 均匀射线；分层份额记录；校准只用均匀射线 |
| 选择 | 末权重 | Train-6 S-c：NLL_UNK + 2·HCFalseNoHit_UNK + 2·KnownPassThrough |

## 推理成本
推理时需要整幅视角的射线网格，2D U-Net 依赖相邻射线。因此每个候选视角先由 `view_belief` 完整解码，再由回放适配器把 V5.2 打分器的分块调用逐块喂回。G0 测试验证这样聚合与 V5.2 原生聚合逐值一致。

实测打分开销（固定状态 smoke）：每状态约 0.8–1.1 s（6–10 个候选）。规划器 smoke 中每步约 0.8–1.3 s。

## 禁止项与守卫
- 不调用 Flow / `M.sample`、`G.decode`（深度图）、DistP1 的 `new_set_from_depth` / `new_mask_from_depth`，也不调用 `V2Predictor.cond / sample`。
- 规划器中的守卫一旦发现上述调用就抛错并计数。G0 规划器 smoke 的计数全部为 0。
