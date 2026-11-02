# EXP6 V5.2 冻结协议 v1.0（2026-11-02，G0 通过后、正式训练前冻结）

依据：`EXP6_V5_2_ARCHITECTURE_AND_PILOT.md`（本目录）。冻结后的改动只记入 `reports/V52_DEVIATIONS.md`；不看任何评估场景的结果改模型、λ、η 或区域定义。

## 1. 模型与训练（唯一一次训练）
| 项 | 值 |
|---|---|
| 父权重 | `work_dirs/exp6_v5/checkpoints/v5_final.pt`（V5 EMA@12000，sha `5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543`）；只复制 encoder + V5RayDecoder（126 张量），Flow U-Net 不加载 |
| 新模块 | HazardHead 13,953 参数（`configs/model_v52.json`；64 个区间，log1p(t / 16 VS) 均匀，t ∈ [tcap · 0.5 / 70, tcap]）；V5 解码器中 Flow 条件部分（BeliefSummary、CondAdapter、out_scale，14,944 参数）冻结不用 |
| 损失 | L = L_surface + β L_firsthit，β = 1；不加 Flow / Coverage / Ranking / KL 损失 |
| β 规则（G0 预先确定） | 联合设置下 20 步共享主干梯度中位数：L_surface 0.536，L_firsthit 1.631，比值 3.04 ∈ [0.2, 5] → 保持 β = 1（规则：比值越界才改为 g_surf / g_fh） |
| 首中标签 | 分片中的 T1.1 首中 t*（训练射线的标签相机）；t* < tcap → 所在区间，否则（含 inf）→ no-hit；t* < 近裁剪 → 不监督 |
| 射线 | 每视角 256 条标签缓存射线（表面 + 首中）+ 256 条均匀射线（首中） |
| 阶段 | Stage H 0–1,999：encoder / decoder / surface head 冻结，只训 HazardHead（lr 5e-4，100 步 warmup，余弦到 0 @6000）；Stage J 2,000–5,999：encoder + decoder 解冻（lr 2e-5，300 步 warmup，余弦到 0 @6000），hazard 继续 |
| Batch / EMA | 全局 64 视角 / 8 状态 / 步，4 rank；AdamW β (0.9, 0.99)，wd 0.01，裁剪 1.0，bf16；EMA 0.999（初值 = 父 EMA + hazard 初始化） |
| 数据 | Train-6 去 S-c 138 轨迹（E2 D100 分片）+ Train-6 多表面标签 `work_dirs/exp6_v5/labels/v1`；MP3D 0 |
| **最终权重** | **固定 EMA@6000**（训练前冻结；不按 S-c / Dev-3 / Test-3 / MP3D 选择）。若 Stage J 中 S-c 第二+ 表面召回崩溃（< 0.5 × 起点），报告并回看 EMA@2000 / 4000，不静默继续 G3 |

## 2. Camera Score（只在推理；`configs/score_v52.json`）
- S_visible、P_surf、P_space 定义见 `ARCHITECTURE.md` §3（`seen2scene/futureview/v52_score.py`）；新颖性 = S10 `band_dil`（与 DistP1 相同），q_valid 排除 D100 已确认 FREE 的体素（被排除的质量单独报告）；距离权重 = DistP1 的 min(1, (z / z0)²)，z0 = 每场景 / 楼层冻结值。
- 参数：每区间 8 个子点；阈值 vis 1e-3、surf 1e-2、space 1e-2；Δ = 8 VS；远遮挡折扣 κ 的尺度 128 VS；非 frontier 的 UNKNOWN 格权重 ε = 0.1；frontier = 26 邻域内有 ≥ 25 % 已确认 FREE 的格；UNKNOWN 格 = 域内比例 ≥ 0.5 且已观测比例 < 0.5。
- 归一化：每项除以 max(候选集中该项最大值, floor_k)。**floor_k 规则（训练后、任何 G2 / G3 前执行一次）**：在 Train-6 S-c 留出轨迹（每场景 1 条，t = 5, 15, …, 95 的 60 个状态，真实记录回放、B2 几何）上用最终权重计算每状态各项的候选最大值，floor_k = 0.05 × 其中位数。
- **λ ∈ {0, 0.1}，η = 0.5**，同一权重，两个打分分支；λ 不参与训练。

## 3. 评估
| 门 | 内容 |
|---|---|
| G1 | S-c（训练监控）、Macarons++ Dev-3（B0 150 + B2 125 态）、MP3D VALSEL（90 态）：多表面召回 / precision / Brier / ECE（第一、第二+，按间隔含 ≥ 128 VS）——评估端网格 GT；首中 NLL、最高区间准确率、no-hit 校准、众数 / 期望深度误差（oracle t*，只作指标）；打分延迟、显存 |
| G2 | 同 365 态、同候选、同选点规则（`ab_load.pick_of`，含 MP3D 的 prefer-unvisited）：被选候选真实一步增益（pp）、Top-1、Spearman、与 V5 / V4C / B2 / E2-D100 选择不同的比例；按阶段、域、UNKNOWN 射线比例分组；记录 S_vis、P_surf、P_space、Pot%、λ 翻转 |
| G3 smoke（预先冻结） | Macarons++ Test-3：alhambra s0 / s1、bridge s0 / s1（Test-3 按字母序前两个场景 × 前两个起点），100 步；MP3D VALSEL：8194nk5LbLH L0 起点 10、EU6Fwq7SyZv L0 起点 67（VALSEL episode 列表前两个），150 步；λ = 0 与 0.1；VALSEL 上同时跑 V5 与 V4C 作配对参照（Macarons++ 用已有的 V5 / V4C / B2 / E2-D100 / MAGICIAN 闭环） |
| 全量 G3 的放行规则（预先冻结） | G0 通过，且下列之一：(a) G2 中某个 λ 的被选真实增益（全部状态）在某一域 ≥ 0.9 × V5；(b) smoke 中某个 λ 在某一域的平均 AUC ≥ V5 同 episode 平均 − 0.01。满足 → 同一冻结权重、两个 λ 跑 Macarons++ 30 × 100 与 MP3D 10 × 150，场景 / 楼层级配对 bootstrap CI；不满足 → 停在 smoke，报告 |
| 规划器（不变） | 候选生成 / 合法性、`n_win = 5`、R10 刷新、真实 D100 + S10 历史融合（p1_full b2）、覆盖率评估器；V5.2 只替换 `v2_plan_core.score_state` 的打分（`scripts/v52_core.py`），Flow / 深度解码 / DistP1 调用由守卫计数（必须为 0） |
测试场景（Test-3、MP3D 评估楼层）与 VALSEL 均被历次实验使用，**非盲测**；不用其 GT 调 λ / η 或改模型。

## 4. 停止条件
需要评估 GT 才能打分、概率不归一 / NaN、预测写回 P1、同体素重复计分、空间潜力退化为 1 − b、调用了 Flow、标签缺失、D100 / S10 时间历史用错 → 停止并报告。完成 G0 → 单次训练 → G1 / G2 → G3 smoke →（满足规则则）全量 G3 → 报告后停止；不启动 U0、室内微调、Exploration Memory 或其他训练。
