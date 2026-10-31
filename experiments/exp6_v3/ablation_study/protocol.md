# EXP6 V3 Ablation — 协议 v1.0（2026-10-31 09:55 冻结；在看任何新计算结果之前）

离线诊断，不训练、不改模型、不改 DistP1 / Greedy、不改任何冻结的历史数据与评价目标。GT 只在评估侧（oracle 增益、t\*、标签）出现，且在该状态全部规划侧分数写出之后读取。所有对冻结规则的偏离记入 `reports/V3_ABLATION_DEVIATIONS.md`，本文件不改。

## 0. 模型（只加载，sha256 在运行 manifest 中核对）
| 名称 | checkpoint | 输入地图 | 打分图 | 来源 |
|---|---|---|---|---|
| **B2** | V2-Mix `W21-1_V2Mix_best/ema_017500.pt`（`343f41ab…`） | D100（P1FusionFull，全部有效像素） | 稀疏 band_dil（10% 采样 P1） | Task 3 B2 规划器 |
| **T1** | V3-VAE-XAttn `task1_xattn/checkpoints/final.pt`（`639c9d3a…`） | D100 → 4× 聚合 → VAE μ | 稀疏 | EXP6 |
| T0 | V3-VAE `task0_vae/checkpoints/final.pt`（`7ed0ddd7…`） | 同上 | 稀疏 | EXP6 |
| E2D | E2-D100 `e2/training/d100/ema_005000.pt` | D100 | 稀疏 | E2 pilot |
| V2M-sparse（= B0）、V2O | `historical EXP5`：`v2.json / v2.npz` | 稀疏 | 稀疏 | EXP5 A1，不重算 |
| GTD | 真实深度 t\* 经同一 DistP1 | — | 稀疏 | EXP5 S1 / S2（Pool CL 已有；Pool FX 补算） |
主分析 = **B2、T1**；T0、E2D 为对照；MAG-G / MAG-B 只在 G3 对照表中出现（不能输出 K=4 深度）。

## 1. 固定状态池（任何模型的分数都在同一 `(dataset, scene, trajectory, step, candidate set, P1)` 上）
- **Pool CL**（主池，闭环相关性）：EXP5 A1 / A2 清单 `work_dirs/exp5_future_view/ablation_study/manifests/states_{indoor,outdoor}.json`，MP3D 368 / Macarons++ 1,093 个去重状态；来源轨迹 = Task 2 闭环（V2O、V2M、MAG-G、MAG-B）；时间点 MP3D {0,5,20,25,50,55,100,105,140,145}、Macarons++ {0,5,10,15,20,25,50,55,80,85}；候选集 = EXP5 §2.5（MP3D prefer-unvisited 合法集，Macarons++ 合法列表）；噪声 = EXP5 A1 种子（室内 `exp5-V2|task0cl|<tag>|t<r>|k<k>`，室外 `closed_loop_v1.seed_key(scene,start,0,r,k)`，r = t − t mod 10）。
- **Pool FX**（EXP6 G2 的状态）：VALSEL 90（9 ep，E2 轨迹；噪声 `exp5-T3E2|valsel|<tag>|t<r>|k<k>`）+ Dev-3 B2 层 125（15 ep，Task 3 B2 轨迹）+ Dev-3 B0 层 150（⊂ Pool CL，只作回归）。候选集 = 合法列表。
- 相位：Pool CL 用 EXP5 `t_bin`（MP3D early ≤ 25 / mid ≤ 55 / late；Macarons++ early ≤ 15 / mid ≤ 55 / late）；Pool FX 用 EXP6 `phase_of`（< 0.2 T early，≥ 0.8 T late）。表头注明，不混用。
- 规划器状态重建：MP3D 区域 n_win = 5 以刷新步 r 的相机为中心（Task 2 卡片），P1 = 记录 0..t，编码器输入 = r 时刻的地图；Macarons++ 整场景一个网格。D100 融合与稀疏融合都从同一 svc 记录重放。

## 2. Experiment A — 三条评分路径（同状态、同候选集）
- **A1 Pred+DistP1**：`score_pred(s,a) = mean_k DistP1(P1_s^{sparse}, D_a^k)`，K = 4，`PC.score_state(core, geo_D100, h_r, legal_R, legal_T, eps, z0, keep_hits=True)` 得到各样本命中深度，再在稀疏几何上 `distp1_k`（Task 3 / EXP6 冻结规则）。保存 `t [n_legal,4,7296]`、`hit_u` 原始样本。
- **A2 GTDepth+DistP1**：对同一候选的 t\*（`oracle.npz:tstar`，64 × 114 射线，jitter 0），`tt = t* if t* ≤ tmax else ∞`，`SCO.new_set_from_depth(tt, o, d, dz, k_M, dims, domain, band_dil, z0)` 的权重和（EXP5 S2 规则，逐位相同）。K = 1，因为确定性传感器下真实观测唯一；不人为复制。
- **A3 OracleGain**：`oracle.npz:gain`（MP3D C_obs 规则；Macarons++ 单调覆盖，`gain_official` 次口径），EXP5 S1 / E2 / E1.2 的冻结评估器，不重算。
- 选择规则（与线上相同）：MP3D `v2nv_planner.nv_choice`（argmax，并列取最小索引，全部 ≤ 0 时按种子随机）；Macarons++ `m_score.choose`。
- 指标（EXP5 §3.3）：Spearman(score, G)（候选 < 2 或常数时无定义，单独计数）；Top-1（并列容差 τ = 1e-4 覆盖率）与严格 Top-1；pairwise accuracy（|ΔG| > τ 的候选对）；regret = max G − G(选中)（百分点）与归一化 regret（max G > τ 的状态）；一致率。
- 拆分：`G*(s) = max_a G(s,a)`，`r_pred = G* − G(a_pred)`，`r_gt = G* − G(a_gt)`，`Δ = r_pred − r_gt`（可为负，不解释为非负可加的"纯预测误差"）；预测保真度 Spearman(S_pred, S_GTD)，评分保真度 Spearman(S_GTD, G)。
- 汇总：状态 → episode（场景 × 起点 × 第一来源）→ 场景；均值 = 场景均值的平均；CI = 场景级 bootstrap（MP3D 5 楼层、Macarons++ 6 场景；10,000 次，seed 0）；配对差按场景配对；另报 Test-3 / Dev-3 各 3 场景均值（不给 CI）。
- **MP3D 中期停滞分析**（预先定义）："停滞状态" = 来源轨迹在 [t − 30, t] 的覆盖增长 < 0.01 且 t ≥ 30（用存储的 episode 曲线，模型无关）。报告：停滞状态数、G\* 分布与 G\* > τ 的比例（是否存在合法高收益候选）、各模型 r_pred / r_gt、oracle 最优候选在规划区域外的增益占比（`a1_region_diag` 的独立实现，T0.12-C 拆法）。
- **Macarons++ 高 regret 分析**（预先定义）：对 B2 / T1 各自 r_pred ≥ 0.5 pp 的状态，统计被选候选的 GT 新表面射线中与评估点集相关 / 有用的比例（`a1_relevance_diag` 规则）与距离权重的作用（GTDn 对照）；"虚假遮挡"用 EXP5 A2 的幻象遮挡率定义。区域 / 评分改动只做 counterfactual，不改 A1 / A2 / A3。
- 必过检查：(1) Pool CL 的 Dev-3 B0 层 150 个状态：新重放的 B2 / T0 / T1 `distp1_k` 与 EXP6 行、E2D 与 E2 行 **逐位相同**（失配先原样重跑一次，再失配 → STOP）；(2) Pool FX 的 B2 / T0 / T1 / E2D 分数与 EXP6 / E2 行逐位相同；(3) 每个状态 `legal_idx` 与 `oracle.npz` 相同；(4) FX 的 GTD 在 Dev-3 B0 子集上与 EXP5 `v2.json` 逐位相同；(5) 候选顺序置换不改变逐候选分数（单测，1 个状态）；(6) GT 文件在规划侧分数写出之后读取（代码顺序 + 单测）。
- 交付：`experiment_A_gtdepth_oracle/report/{A_G0_VALIDATION, A_FIXED_STATE, A_FAILURE_CASES}.md`，逐状态 / 逐候选 CSV，图。

## 3. Experiment B — Depth Uncertainty Planner（只改评分器）
- 输入：A1 保存的 K = 4 原始样本（`t, hit_u`）与稀疏几何的射线字段（`valid, cls, n_u, d_lo, d_hi, t_known, tcap`），同一候选相机的同一像素 / 射线、不同 k；不跨相机比较；不读 GT。
- 定义（射线 p，有效 = `valid`；M_unknown(p) = `n_u > 0`，即射线在到达已知表面 / 离开域之前穿过 P1 的 UNKNOWN 体素，几何边界与当前 P1 相同；完全已知或空的射线 U ≡ 0）：
  - HIT_k(p) = `hit_u[k,p]`（样本 k 在 UNKNOWN 区解码出表面）；NOHIT 否则（含到达已知表面或无表面）；无效射线排除。
  - `U_hit(p) = H_2(q)`，q = K 个样本中 HIT 的比例，二元熵以 log2 归一化到 [0,1]。
  - `U_depth(p) = clip(std_k(t_k | HIT_k) / span(p), 0, 1)`，仅当 ≥ 2 个样本 HIT；span(p) = `d_hi − d_lo`（该射线 UNKNOWN 段的首末采样距离，模型单位）；span ≤ 0 → U_depth = 0。NOHIT 不以 far 代入。
  - `S_depthU(a) = (1/|valid|) Σ_p M_unknown(p)·[α U_hit(p) + (1−α) U_depth(p)]`，**α = 0.5（冻结）**。
  - P0 = `mean_k DistP1`（线上基线）；P1 = `S_depthU`；P2：状态内对候选集做 min-max 归一化（常数 → 0），`S_P2 = n(S_cov) + λ·n(S_depthU)`，**λ = 1.0（主，冻结）**，λ ∈ {0.5, 2.0} 只作敏感性报告，不据测试选择；RiskAware（探索性）= `mean_k DistP1 − 0.5·std_k DistP1`。
  - 并列 → 最小索引（同线上）。DepthU 是 predictive disagreement proxy，不是校准的认知不确定性或信息增益。
- 固定状态 G2：模型 B2、T1（T0 / E2D 若原始样本可用则加入），同状态 / 相机 / 种子 / K；指标 regret / Top-1 / Spearman，Early / Middle / Late，逐场景与配对 CI（同 §2）；DepthU 与 GT 未来深度误差、实际 oracle 增益的关系（重点标出"很不确定但无收益"的候选）；K = 4 选点稳定性用 EXP5 `kabl.npz` 的历史独立种子集（仅 V2M）作参照，不扩大 K。
- **预注册 Gate（决定是否进入 G3）**，dev 集 = MP3D Pool FX VALSEL、Macarons++ Pool CL 的 Dev-3 状态；对每个数据集、每个模型 m ∈ {B2, T1} 分别判定：P1 或 P2 相对 P0 满足 (a) 整体 regret 或 Early regret 的场景配对差 95% CI 上界 < 0，或 (b) 相对改善 ≥ 10% 且 MP3D ≥ 4/5 楼层、Macarons++ ≥ 2/3 Dev-3 场景改善；且 (c) 没有任何场景的 regret 相对恶化 > 25%。通过 → 对该数据集、该模型、该评分器按冻结 G3 协议跑完整闭环（MP3D 10 ep × 150 步 / Macarons++ 30 ep × 100 步，同一合法候选生成、真实观测更新 P1、同一起点），报告 AUC / Final / Cov@k / 逐场景 / 配对 CI / 选择分歧案例；不通过 → 报告阴性结果，最多 2 个标记 exploratory 的闭环。不调 λ、α、场景、时间点或 K 来让 Gate 通过。
- 交付：`experiment_B_depth_uncertainty/report/{B_SCORE_AUDIT, B_FIXED_STATE, B_CLOSED_LOOP, B_DECISION}.md`。

## 4. 资源与执行
- 环境：seen2scene env（`LD_PRELOAD=<env>/lib/libtbb.so`），`torch.use_deterministic_algorithms(True, warn_only=True)`，每卡 1 个进程，`expandable_segments`。
- 并发：按 (pool, dataset, 来源 episode) 分配，原子 claim 文件，原子写出 `<sid>.npz + .json`，断点续跑跳过已完成状态；两节点共享 claims。
- 成本预估（运行后以实测替换）：Pool CL 160 个来源 episode 的 D100 + 稀疏重放 + 4 模型 × ≤ 10 候选 × K=4 ≈ 2–3 GPU-h；Pool FX 24 个 episode < 0.5 GPU-h；分析 CPU。
- 资源核实（09:30）：gpu117 交互作业 3429774（4 × L40S，11-01 14:50 到期，空闲）、gpu115 sbatch 3435033（4 × L40S，11-01 11:09 到期，空闲）均为本任务此前申请、现已空闲，用于本轮评估；3435072（排队）不新增使用。无正式训练。
