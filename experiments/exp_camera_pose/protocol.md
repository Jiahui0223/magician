# Camera Pose Phase 1 —— 预注册协议 v1.0（2026-11-02，在任何 C1 结果出来之前冻结）

依据：任务书（本目录 README 引用）和 `reports/AUDIT.md`。之后的改动一律记入文末的偏差日志。评估场景是此前用过的**非盲**场景。

## 1. 模型（不训练、不改结构）
| 模型 | checkpoint（软链接，运行前核对 SHA） | 推理配置 | 本阶段 |
|---|---|---|---|
| B2 | `work_dirs/checkpoints/exp6_v2/B2/v2mix_ema17500.pt`（343f41ab…） | P1 模式 b2：D100 输入 / S10 DistP1，K = 4，10 步 Euler，R10，室内 n_win = 5 | Stage B / C 必跑，D 条件执行 |
| E2-D100 | `work_dirs/checkpoints/exp6_v2/E2/e2d100_ema5000.pt`（61439612…） | 同上（T3_CKPT 覆盖） | 同上 |
| V4 | `work_dirs/checkpoints/exp6_v4/V4/v4_ema40000.pt`（1b31afff…） | exp6_v4 冻结的 D100 / S10 | Stage B（历史分数），闭环可选 |
| V5 | `work_dirs/checkpoints/exp6_v5/V5/v5_ema12000.pt`（5ef30ac4…） | exp6_v5 冻结的 D100 / S10 | Stage B（历史分数），闭环可选 |

3D U-Net、Ray Encoder、2D Flow、K、Euler 步数和 DistP1 一律不改。

## 2. 方法定义
### C0 —— 原始邻居基线
官方邻居 → 合法过滤 → 模型 → 未来深度 → S10 DistP1 → 原选择规则：
- 室外：`m_score.choose`；
- 室内：`nv_choice`，在未执行过的候选中取 argmax，全为 0 时用带种子的 fallback。

C0 的完整 G3 结果**直接复用历史结果**。新入口在 λ = 0 时直接返回原始 DistP1（恒等），因此与 C0 逐位一致。Stage A 用一个完整 episode 做回归检查。

### C1 —— Frontier-Aware Pose Selection（同一动作集合内重新排序，不扩展候选）
**FrontierPrior F_j**：对候选 j，规划器用它自己的 64×114 射线网格，读取视图射线统计（D100 规划几何）。
- 只统计 `cls == UNKNOWN` 且 `d_lo < tcap` 的射线。这类射线先穿过已知 FREE，然后在传感量程内进入域内 UNKNOWN。
- 进入点 `p = o + d·d_lo` 按粗格（8 VS）量化。F_j = 这些进入点所在**不同粗格的个数**（视图内去重）。
- 先遇到已知表面（TSDF 交叉）的射线不计入。只计 FREE→UNKNOWN 的边界，不把 UNKNOWN 体积当作可见。
- 不读取 GT 网格或未来真实深度。

**分数**：`Score_C1 = D_j / max_k D_k + λ · F_j / max_k F_k`。
- 分母取全部合法候选中的最大值；分母为 0 时该项记 0。
- λ = 0 时**直接使用原始 DistP1**（与 C0 恒等）。
- 选择规则与 C0 完全相同（同一个 `choose` / `nv_choice`）。

λ > 0 会改变一种情况：DistP1 全为 0 时，C0 会触发 fallback，C1 则由 F 决定。这种情况的比例记作 fallback 率并报告。

**λ 网格**：{0, 0.1, 0.3}，不增加其他值。

**λ 的选择（只用开发集固定状态）**：
- 开发集：Macarons++ Dev-3 275 态、MP3D VALSEL 90 态。VALSEL 的房屋与 G3 的 MP3D 房屋不重叠。
- 对每个模型，取使“两个域的 mean selected true one-step gain 的平均值”最大的 λ*；并列时取较小的 λ。
- λ* = 0 表示 C1 在固定状态上没有收益。

### C2 —— 多尺度 / 由粗到细候选
依据 AUDIT §5–6：
- **室外 NOT_SUPPORTED**：官方位姿空间、碰撞和覆盖率都绑定在固定网格上。
- **室内只能作为新动作协议**：需要新网格下的 C0 重跑、网格外候选的新 oracle 和路径成本记账。

Phase 1 **不跑 C2 闭环**。只有 C1 全部完成且时间允许时，才做室内固定状态可行性探针，并单独报告；否则标为 NOT_RUN。

## 3. 阶段与门槛
**Stage A（Preflight）**
- 核对 4 个 checkpoint 的链接目标和 SHA。
- 在固定状态上检查：F 的计算无 NaN、无空候选；C1 的候选集合与 C0 完全相同（Oracle Best Gain 必须相等）；λ > 0 时确实会改变部分选择。
- 闭环回归：用 C1 入口、λ = 0 跑 B2 Outdoor barts s0 完整 100 步，必须与历史 C0 逐步一致（选择 / 覆盖率）。

**Stage B（固定状态分析）**
- 4 个模型 × λ ∈ {0, 0.1, 0.3}，在 Dev-3 275 态和 VALSEL 90 态上，使用历史逐候选 DistP1（同一状态、同一候选、同一种子）加新的 F。
- 报告：候选数、Oracle Best Gain、Selected Real Gain、Regret（只在附录）、Top-1、Spearman、改变选择的比例、F 的计算耗时。
- 按 §2 规则确定 λ*。

**Stage C（短闭环 Pilot，只在开发场景，配对）**
- 模型：B2、E2-D100。λ_pilot = λ*（若 λ* = 0，则用 0.1，只检验“一步指标看不到的长程收益”，不再调参）。
- Outdoor：Dev-3 的 barts / pantheon / colosseum × 起点 0、1，共 6 个 episode，各 100 步。C0 用历史结果。
- Indoor：VALSEL 的 8194nk5LbLH L0 S10、EU6Fwq7SyZv L0 S67、X7HyMhZNoso L0 S30、pLe4wQe7qrG L0 S22，共 4 个 episode，各 150 步。C0 与 C1 都新跑，起点和种子相同。
- 指标：AUC、Final、Cov@20、覆盖曲线，以及 F / D 的诊断量。

**Stage D 门槛**（按模型 × 域分别判定）：
- 条件：pilot 的 mean ΔAUC（C1 − C0）≥ +0.005，且没有任何 episode 的 ΔAUC < −0.03，且 mean ΔFinal ≥ −0.01。
- 通过才跑该组合的完整 G3。优先级：B2 Outdoor → E2 Indoor → B2 Indoor → E2 Outdoor。V4 / V5 只在 B2 和 E2 都有通过项且资源有余时考虑。

**Stage D（完整 G3）**
- 范围：MP3D 10 episode × 150 步；Macarons++ Dev-3 + Test-3（6 场景，30 episode × 100 步）。
- 与历史 C0 的同一 episode 配对比较，按场景 / 楼层做 bootstrap CI。
- Dev-3 的起点 0、1 参与过 pilot，因此另外单列 Test-3（未参与选择）的结果。

## 4. 结论规则
- **GO**：至少一个必跑的模型 × 域在完整 G3 上 ΔAUC > 0 且 CI 不含 0（或 ΔAUC ≥ +0.01 且场景胜数 ≥ 4/6），并且其他已运行组合没有显著退化。
- **MIXED**：部分组合提升，另一些不显著或退化。
- **NO-GO**：所有必跑组合的 pilot 门槛都不通过，或者完整 G3 无改善 / 系统性退化。
- **INCOMPLETE**：必跑部分没有完成。

pilot 结果不得写成完整 G3；候选 oracle 的变化也不得写成模型提升（C1 与 C0 的 oracle 相同）。

## 5. 资源与可恢复性
- 运行产物放在 `work_dirs/exp_camera_pose/`，按 episode 写 done 文件和 claim 文件，中断后可续跑，不重复已完成的 episode。
- 记录 Git commit、checkpoint SHA 和种子（沿用历史种子键）。
- 优先使用 gpu115 的 4 卡。需要时提交 4 卡 sbatch，受每用户 8 卡 QOS 限制。不抢占、不取消他人作业；节点故障时保留已有结果，不无限重试。

## 6. 偏差日志
- 无协议偏差。下面是工程修正，不影响任何结果：
  - `analyze_closed.py` 中室外 episode 字段名写错（`pick` → `action_key`），已修正，没有重跑实验；
  - `run_queue.sh` 的覆盖率 done 文件路径少了一层 `V2M/`，在覆盖率阶段开始前用原子替换修正。
- 发现（不是改动）：FrontierPrior 的近场性质——候选在未观测区内时 F ≈ 1——在 pilot 之后才在分析中确认。按协议没有修改定义。
