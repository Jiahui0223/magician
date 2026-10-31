# EXP6 V3 Ablation — Experiment C 协议 v1.0（GT-Depth Closed-Loop Evaluation；2026-10-31 15:40 起草，G0 通过后冻结，正式 40 条 episode 开始前不再修改）

诊断实验。不训练、不改模型、不改 DistP1 / Greedy / 候选生成 / 覆盖率定义，不改 A / B 的任何数据与结果。V4 不启动。所有偏离记入 `../reports/V3_ABLATION_DEVIATIONS.md`，本文件不改。

## 1. 两个 GT 诊断 Planner（只有评分来源不同）
| 组 | 评分 | 其余 |
|---|---|---|
| **C1 GT depth + DistP1** | 每个合法候选 **一张**真实首次命中深度 t\*（64 × 114 打分射线，jitter 0，`t11_labels.Renderer` = EXP5 S1 渲染器）→ `tt = t* if t* ≤ tmax else ∞`（tmax = 区域出口 / 传感器射程 70）→ `scoring.new_set_from_depth(tt, o, d, dz, k_M, dims, domain, band_dil, z0)` 的权重和（距离权重 min(1, (z/z0)²)，体素去重；**与 Experiment A 的 GTD 逐行相同**），K = 1，不复制、不加噪 | Task 2 / Task 3 B2 规划器的全部其余部分：稀疏 10% P1（`m_p1.P1Fusion`，= B2 的打分图 band_dil）、室内 n_win = 5 区域 + R10 刷新、室外整场景网格、合法候选 = 环境列表、选择规则（室内 `v2nv_planner.nv_choice` prefer-unvisited + argmax + 最小索引 + 全 ≤ 0 时种子随机；室外 `m_score.choose` + `m_common.fallback_choice`）、真实观测融合、Coverage |
| **C2 GT gain greedy** | 每个合法候选的**真实单步覆盖增益** `G(a) = Cov(S_t ∪ O_a) − Cov(S_t)`：MP3D = 冻结 C_obs 规则（候选位姿用冻结传感器渲染、以模拟器第 t+1 帧的 "sample" 种子取 10% 采样、新覆盖的可见 GT 点数 / 可见 GT 点数；= T0.12-C / EXP5 S1 室内评估器）；Macarons++ = Task 2 **单调覆盖**（`t25_out_cov` 规则；正式室外主指标；T7a 快照 / 恢复评估器 `m_gain_eval`，种子 t+1），官方 MAGICIAN 增益只记录不用于选点 | 同上；同一选择规则作用于增益向量（并列 → 最小索引；全 ≤ 0 → 同一 fallback）。**候选集单步 oracle greedy，不是轨迹最优，不保证 Final ≥ C1** |

GT 旁路：模拟器在写出第 t 步服务记录后、执行动作前，为**当前合法列表**渲染 / 评估并写 `<episode>/cq/cq_t<t>.npz`（键只允许 `t, legal_idx, tstar`（C1）或 `t, legal_idx, gain`（C2）），路径随服务记录一起发给子进程；子进程读取前断言 `t` 与 `legal_idx` 与记录一致。每次 GT 查询前后比较模拟器状态哈希（MP3D：`Coverage.covered` + 覆盖值；Macarons++：位姿 key、历史长度、官方覆盖值、单调覆盖数组），不同即 STOP。未选中候选的 GT 不进入 P1、覆盖、历史或下一步合法集。子进程的 LeakGuard 保留（Task 2 / Task 1.4 规则 + 禁止 Experiment C 工作目录下除本 episode svc / cq / 规划日志以外的一切、禁止 EXP6 / Task 2 评估输出），cq 读取次数写入日志（`cq_reads`）。

## 2. 协议（= EXP6 V3 G3 = Task 2 / Task 3 冻结闭环）
- **MP3D**：5 个 val 楼层 × 2 起点（`task2/configs/eval_indoor_v1.json`），C0 + 150 步；T0.12 1.0 m 网格、官方邻居、官方 GT 线段合法性、起点吸附；`t23_in_sim.py` 的循环体逐行复制（`c_in_sim.py`），新增的只有 GT 查询与诊断；AUC@150、Final@150、Cov@{10,20,50,100}。
- **Macarons++**：Test-3（alhambra、bridge、neuschwanstein）+ Dev-3（barts、pantheon、colosseum）× 5 起点（`eval_outdoor_v1.json`），C0 + 100 步；`v2_sim.run` 不改（官方网格、`get_valid_neighbors` → GT 线段、悬停、种子观测、官方覆盖、逐步 checkpoint / 逐位 resume）；`EpisodeSim` 子类只增加在线单调覆盖与合法列表缓存；覆盖率用 `t25_out_cov.py` 不改（`c_out_cov.py`）：官方曲线回放逐位相等否则 STOP，单调覆盖为主指标；AUC@100、Final@100、Cov@{10,20,50}。Test-3 已在 Task 2 / EXP5 / EXP6 使用，**非盲测**，本轮为诊断性实验。
- 相同起点、相同初始观测、相同动作空间 / 移动规则 / 合法性、相同覆盖评估器；不同 planner 从第一步起产生不同轨迹，不共享未来状态。随机性只在 fallback（种子键：室内 `exp5-T0.10|NV|T2|<C1|C2>|<tag>|<t>`，室外 `m_common.fallback_choice(scene,start,t,n)`，与线上形式相同）。
- 历史参照组直接复用正式 G3 结果（不重跑）：MAG-G、MAG-B（Task 2 共享协议闭环）、B0、B2、E2-D100（Task 3 / E2）、V3-T0、V3-T1（EXP6）。它们与 C1 / C2 的协议相同（同一模拟器代码、同一 episode 定义）；每步时间不可比（不同负载）。

## 3. 每步诊断（评估侧，子进程答复之后计算，从不发给子进程）
MP3D：全部合法候选的真实增益（C1 现算；C2 复用其查询）、oracle 最优 / 被执行动作的 regret / Top-1（τ = 1e-4）、剩余可见 GT 点比例、其中落在当前规划区域（刷新步相机为中心的 n_win = 5 区域，T0.12-C 拆法）的比例、被 GT 线段阻挡的邻居（不在合法集）的真实增益（候选空间诊断）。Macarons++：全部合法候选的单调 / 官方增益、regret / Top-1、剩余（单调）比例、剩余点中位于某合法候选相机 70 单位射程内的比例、在线单调覆盖。执行一致性检查（每步）：被执行候选预先算出的 10% 采样 = 下一帧采样（逐位），其增益 = 覆盖增量（|Δ| ≤ 1e-12；室外另要求官方覆盖逐位）；C_obs / 单调覆盖不下降。诊断使 C1 的模拟器耗时约等于 C2，不影响决策。

## 4. 统计
- 每条 episode 的完整曲线、Final、AUC、Cov@k 保留；场景（MP3D 楼层）内取均值，再做场景级 bootstrap（10,000 次，seed 0）95% CI；配对差按同一 (scene, start) 配对后按场景聚合（`t34_analyze.summarize` 规则）。时间步不作为独立样本。
- 配对：**C1 − B2、C1 − T1、C1 − E2D**（R1）；**C2 − C1**（R2）；C2 − B2、C2 − T1；C1 − MAG-B、C2 − MAG-B；另报 C1 − T0、C1 − B0。判定沿用 Task 3 §4："有意义" = ΔAUC ≥ +0.010 且 ΔFinal ≥ +0.010 且胜 ≥ 4 场景且无场景 < −0.030；"退化" = ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010。Macarons++ 另报 Test-3 / Dev-3 各自均值（3 场景，不给 CI）。
- MP3D 停滞：`e2_g3_analyze.stall` 规则（t ∈ [30, 120] 内 ≥ 30 步覆盖增长 < 0.01 的窗口）；停滞窗口内的诊断（候选数、G\*、剩余、区域内比例、阻挡邻居 G\*）按 §3 汇总。
- C1 / C2 的闭环内单步 regret（§3）与 Experiment A 的固定状态 regret **分表**，不换算、不相加。

## 5. G0（正式运行前必过；结果 `report/C_G0_VALIDATION.md`）
1. Single-view GT：每个候选一张 t\*，形状 [n, 7296]，有限值 > 0，命中率 ∈ (0, 1)；t=0 状态的 t\* 与 EXP5 Pool CL `oracle.npz:tstar` 逐位相同（同起点 → 同状态）。
2. GT-DistP1 回归：C1 子进程在 t=0 的分数与 Experiment A 缓存 `gtd_distp1`（同 sid）相对差 ≤ 1e-9 且 argmax 相同（偏差 #1 的"数值等同"标准；逐位计数同时报告）。
3. Oracle Gain：C2 在 t=0 的增益与 `oracle.npz:gain` 逐位相同；被执行候选的采样 = 下一帧（逐位）、增益 = 覆盖增量（≤ 1e-12）。
4. No-Leakage：每次 GT 查询前后模拟器状态哈希相同；子进程 P1 哈希在 GT 打分前后相同；cq 文件只含允许的键；重复查询同一状态结果逐位相同。
5. Legality：pick ∈ 合法列表且 ∈ prefer-unvisited 候选集（室内）；由模拟器与子进程双重断言。
6. Monotonicity：C_obs 与单调覆盖不下降（断言）；官方室外覆盖允许下降（记录 `official_max_drop`）。
7. Determinism：同一配置跑两次 → 动作、分数、覆盖曲线逐位相同（室内 C1 5 步、室外 C2 3 步）。
8. Baseline replay：用原 B2 子进程（`t32_in_planner.py` / `t32_out_planner.py`，T3_P1_MODE=b2）跑新循环 5 / 3 步 → 与存储的 B2 episode 覆盖、动作、分数逐位相同。
9. Smoke：室内 / 室外 C1 / C2 各 5 / 3 步完整状态更新，无 NaN / 异常退出；室外在线单调覆盖 = `t25_out_cov` 回放。
任一项失败 → 修复、记录、重跑 G0，不开始正式评估。

## 6. 执行与资源
- 顺序：G0 → **C1 全部 40 条** → 阶段回复 → C2 全部 40 条 → 室外覆盖率 → 配对统计 → 报告。队列 `run_queue_c.sh`（C1 作业在前，C2 在后；原子 claim，跨节点共享）。
- 资源：既有分配 gpu117 3429774（4 × L40S，11-01 14:50 到期）、gpu115 3435033（4 × L40S，11-01 10:09 到期），每卡 1 个 worker；3435072 不使用，不取消任何作业。预估：C1 ≈ 40 × 6 min、C2 ≈ 40 × 7 min（含诊断）≈ 9 GPU-h → 8 卡约 1.5 h 墙钟；实测写入 `../reports/V3_ABLATION_RESOURCES.md`。
- 输出：`work_dirs/exp6_v3/ablation_study/experiment_C/eval/{closed_loop/{indoor,outdoor}/{C1,C2}, metrics/outdoor/{C1,C2}, logs, claims}`；G0 在 `.../experiment_C/g0/`；代码 / 协议 sha256 写入 `configs/FROZEN_SHA256.json`。
