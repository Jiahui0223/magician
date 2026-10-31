# EXP6 V3 Task 2 — 协议 v1.0（Coverage-Aware NBV Planning；2026-10-31 17:30 起草，Phase 0 / 单测通过后冻结，正式运行前不再修改）

不训练、不改模型、不改任何冻结的历史结果（V3 Task0 / Task1、Ablation A / B / C）。GT 只在评估侧（oracle 增益、t\* 诊断、离线 counterfactual、失败案例、G3 覆盖率）。所有偏离记入 `reports/TASK2_DEVIATIONS.md`，本文件不改。

## 0. 状态池、模型、缓存（Phase 0 审计）
| 项 | 来源 | 复用 |
|---|---|---|
| Pool CL | EXP5 A1 / A2 的 1,461 个 Task 2 闭环固定状态（MP3D 368 / Macarons++ 1,093；`work_dirs/exp5_future_view/ablation_study/`）| oracle 增益、t\*、候选集、候选采样点 逐位复用 |
| Pool FX | EXP6 G2 的 365 个状态（VALSEL 90、Dev-3 B2 层 125、Dev-3 B0 层 150 ⊂ CL）| 同上（E2 / E1.2 的 oracle）|
| K = 4 预测深度 | Experiment A 缓存 `work_dirs/exp6_v3/ablation_study/cache/<pool>/<dataset>/<sid>.npz`：B2 / E2D / T0 / T1 每候选每样本的解码首次命中距离 `t`（D100 几何解码，冻结噪声种子）| 逐位复用，**不再运行 Flow** |
| 几何重放 | `ablation_study/shared/ab_score.py` 的 Episode / 融合重放（稀疏 + D100，Task 2 区域 / R10 规则）| 代码复用 |
| 模型 | B2 = V2-Mix `343f41ab…`（D100 输入、稀疏打分）；T1 = V3-VAE-XAttn `639c9d3a…`；T0、E2D 对照 | 只在 G3 加载 |
| 历史 G3 | MAG-G / MAG-B / B0 / B2 / E2D / T0 / T1（EXP6 `reports/g3.json`）、C1 / C2（Experiment C）| 不重跑 |
| 重放检查 | Experiment A：1,460 / 1,460 数值等同；Experiment C G0 #8：B2 闭环逐位；本 Task：S0 在 t\* 上 = Experiment A `gtd_distp1` 逐位，S0 在缓存样本上 = 缓存 `distp1_k` 逐位（每状态断言 + 分析汇总）| |

**dev / eval 划分（选择规则只看 dev）**：MP3D dev = Pool FX VALSEL（9 楼层，6 个非评估房屋）；MP3D eval = Pool CL 的 5 个评估楼层（= G3 楼层，非盲）。Macarons++ dev = Dev-3（barts / pantheon / colosseum；CL 547 状态 + FX 125）；eval = Test-3（非盲，Task 2 / EXP5 / EXP6 / C 已用）。全部结论标 exploratory。

## 1. 原 DistP1 的审计结论（`exp1_coverage_scoring/report/E1_DISTP1_AUDIT.md`）
`scoring.new_set_from_depth`：命中点 q = o + d t（区域出口 / 射程 70 之外 = 无命中）→ 体素 floor(q/VS)（VS = 0.011 模型单位 ≈ 1–1.8 cm）→ 只计域内、且不在**稀疏 10% P1 BAND 的 3×3×3 膨胀**内的体素 → 唯一体素，权重 = 该体素上各射线 min(1, (z/z0)²) 的最大值，z = 运行时 z 深度，z0 = MAGICIAN scene_scale / 2（MP3D 楼层 32–91 运行单位 = 4.7–13 m；Macarons++ 15–50 = 射程的 22–71%）→ 求和；K = 4 样本取均值。覆盖率口径：MP3D C_obs = 可见 GT 点（300k 面积加权）落入任一历史 **10% 采样点** eps 邻域的比例，eps = 0.08 模型单位 = 0.55–0.90 运行单位（7.8–12.8 cm）；Macarons++ 单调覆盖 eps = 1.0 运行单位（3–8 体素）。四处口径差：(i) 新颖性尺度 1 体素膨胀 vs eps ≈ 5–13 体素，且 BAND 只沿 10% 采样射线存在；(ii) 权重上限 z0 vs 覆盖率在射程内 ≈ 面积比例；(iii) 无入射角（1/cosθ）补偿；(iv) 区域裁剪（MP3D）。EXP5 已证明去掉距离权重（纯计数 ∝ 1/z²）更差，因此 S1 只能替换为有物理解释的面积项，不能删权重。

## 2. 评分器（`common/t2_scorers.py`；全部为每射线权重，视图分数 = 唯一命中体素上最大权重之和 = 原 DistP1 去重规则；K 样本取均值；预先登记，不再增删）
| 名称 | 每射线权重 | 说明 |
|---|---|---|
| **S0** | min(1,(z/z0)²)，只计不在稀疏 band_dil 内的命中 | 原 DistP1（逐位复现，`SCO.new_set_from_depth`）|
| S1a | Â = (αz)²·dz/cosθ ÷ (αz0)²，同 S0 的新颖性 | 一条打分射线的精确表面足迹面积（α = 8 tan30°/255；cosθ = \|n·d\|，n 由 64×114 命中网格的中心差分估计，深度跳变 > 15% 或缺邻居 → cosθ = 1；cosθ ≥ 1/4）；无上限 |
| S1b | min(1,(z/z0)²)·dz/cosθ | 只加入射角补偿（隔离 cosθ 的作用）|
| S1c | (z/z0)² | 只去掉上限（隔离 z0 上限的作用）|
| S1d | min(Â, Â_sat)，Â_sat = 1.6π eps²/(αz0)² | 用"每 4×4 块约 1.6 个覆盖采样点最多能覆盖的面积"替代 z0 上限（采样饱和，参数来自覆盖率定义）|
| **S2** | min(1,(z/z0)²)·N_lin，N_lin = clip(d_obs/(2 eps), 0, 1)，**不再做 band_dil 排除** | d_obs = 命中点到**历史 10% 采样点**（记录 0..t，= 覆盖率的参照集）的距离；N_lin = eps 圆盘未被覆盖部分的线性近似；τ_lo = 0、τ_hi = 2 eps 由覆盖率定义给出，不调参 |
| S2h | ·[d_obs > eps] | 覆盖率尺度上的硬新颖性 |
| S2d | ·N_disc（两 eps 圆盘的精确未覆盖面积比）| 参数无关的替代 |
| **S3** | S0 权重 × V，V = 0 若命中在 D100 稠密已知表面前 < TRUNC（3 体素）或在其之后 | 稠密 P1 = B2 / T1 的模型输入（真实观测）；"UNKNOWN ≠ FREE"：只用已知表面与 EMPTY，不惩罚未知区 |
| 组合 | S0S3（= S3）、S2S3、S1aS2、S1bS2、S1dS2、S1aS2S3、S1dS2S3 | 权重相乘，同一去重 |
| 全局（仅 GT 诊断，R1 桥）| G_hard / G_soft / G_area：t\* 只以射程封顶、不做区域裁剪与域检查，新颖性由采样点树给出 | 预测深度无法得到区域外命中，不进入 E1-Pred |
固定常数：VS 0.011、TRUNC 3 VS、α、1.6 采样 / 块、cosθ 下限 0.25、法线跳变阈 0.15；eps 与 z0 来自各数据集冻结定义；无自由参数。

## 3. Experiment 1 阶段与指标
- **E1-GT**：每状态每候选，t\* 经各评分器 → 与 oracle 增益比较：Spearman、Top-1（τ = 1e-4 覆盖率）、pairwise、regret pp、归一化 regret；状态 → episode → 场景均值，场景级 bootstrap 95% CI（10,000 次，seed 0）；按相位（CL：EXP5 t_bin；FX：EXP6 phase_of）；校准：分数 vs 增益的 Spearman（跨候选、跨状态）与线性拟合；失败案例（S0 错 / S 新对，反之）。
- **E1-Pred**：同一状态、同一候选、同一缓存 K = 4 样本，B2 / T1（主）、T0 / E2D（对照）各评分器的 K 均值 → 同上指标；"仅在 GT 上有效"= GT 改善而预测退化 → 记为对预测误差敏感。
- **G2 Screening（冻结；dev 集；对每个 (模型 m ∈ {B2, T1}, 数据集) 分别判定）**：候选评分器 S 通过当且仅当 (a) GT 阶段 S 的 dev regret ≤ S0 的 dev regret（场景配对均值）；(b) 预测阶段 S − S0 的场景配对 Δregret 95% CI 上界 < 0，或相对改善 ≥ 10% 且 MP3D ≥ 6/9 楼层、Macarons++ ≥ 2/3 场景改善；(c) 没有任何 dev 场景的 regret 相对恶化 > 25%。**选择规则**：在两个数据集、两个模型都通过的变体中，取 dev 相对改善（四个 (m, 数据集) 的平均）最大者为 G3 评分器；若只有一个数据集通过，则只在该数据集跑 G3 并如实报告；若无通过者 → No-Go，不跑 G3。eval 场景的结果只报告，不参与选择。
- **E1-G3**：B2 + S、T1 + S（冻结 G3 协议：MP3D 10 × 150、Macarons++ 30 × 100；同一候选、噪声、记录、覆盖率），对照 B2、T1、C1、C2 及历史组；AUC / Final / Cov@20 / 停滞 / 逐场景配对 CI（Task 3 §4 判定）；每候选评分耗时与显存。只有 Early 改善 → 报告为 Early-only。

## 4. Experiment 2（MP3D 优先）
- **R0**：B2 回放一致（Experiment C G0 #8 已逐位；本 Task 的 G3 运行器再做 5 步回放）。
- **R1 评分区域（仅打分侧）**：离线 counterfactual：在 Pool CL / FX 的 MP3D 状态上，用 t\*（全网格）与 S2 类点新颖性计算 n_win ∈ {5, 7, 9} 区域裁剪下的 GTD 分数与无裁剪（G_\*）分数 → 与 oracle 增益（全层）的 regret / Top-1；并报告候选的 oracle 增益在各区域内的占比。预测深度不能超出模型域 → 不做模型版 R1；如需改模型输入另设实验。
- **R2 Frontier-aware（无 GT）**：从真实记录构建全层 2D 可通行 / 未知图（相机高度切片：EMPTY 体素投影 = 可通行，与之相邻的 UNKNOWN = frontier；网格 = T0.12 相机网格 1 m）；FrontierProgress(C) = 当前相机到最近可达 frontier 的路径距离 − 候选相机到最近可达 frontier 的路径距离（路径只经过已知可通行格；不可达 frontier 不计）；两种模式：固定 bonus S = Gain + λ·FP（λ = dev 楼层（VALSEL，Experiment A 缓存）上 B2 非零 DistP1 的中位数 = **119.1**；只此一个预设值）；触发模式：仅当模型预计 max Gain < θ（θ = VALSEL 上 B2 每状态最大 DistP1 的 20% 分位 = **75.3**）时按 FP 选、并列按 Gain；访问计数惩罚防往返（已访问格 FP 减半）。FP ∈ {−1, 0, +1}（一格移动）；无可达 frontier → FP = 0。先在 Pool CL 停滞状态上离线评估（用 oracle 增益看 FP 导向动作的真实增益与后续 5 步可达增益），再 G3（MP3D 10 条）。
- **R3 候选空间 counterfactual（离线，GT）**：原合法集 vs 扩展集（两步可达的网格位姿，经原始线段合法性检查；记录候选数与移动预算的变化）的最大真实增益、frontier 路径进展、是否能看到新有效表面；只分析，不改在线候选生成。
- **R4 组合**：仅组合有独立正向证据的模块；一次只改一个变量的对照。
- Experiment 2 验收：停滞持续时间、有效远距离推进、Final 与 AUC（MP3D 10 条，楼层聚类 bootstrap），不以一步 regret 为据。

## 5. 失败案例规则
E1：S0 选错且 S 新选对的 regret 最大 3 例、反之 3 例、两者皆错 3 例（各数据集）。E2：Experiment A / C 的 MP3D 停滞 episode（C1 / C2 / B2 的最长停滞窗口）各 3 例，输出 §8 要求的 10 项（俯视 P1、候选、预测 / 真实增益、frontier、新旧选择、后续曲线）。五类问题分别标注。

## 6. 资源
既有分配 gpu117 3429774（11-01 14:50）、gpu115 3435033（11-01 10:09），每卡 1 进程；固定状态打分 ≈ 1 GPU-h（无模型推理）；G3 每组 40 条 ≈ 4–5 GPU-h；不取消任何作业；实测写入 `reports/TASK2_RESOURCES.md`。
