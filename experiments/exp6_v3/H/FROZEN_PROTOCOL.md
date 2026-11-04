# FROZEN_PROTOCOL — EXP6 V3 / H：Coverage-Aware Decision & Short-Horizon Planning（protocol H-v1）

- **冻结**：2026-11-04，在任何 H1–H3 正式闭环之前。
- **机器可读版**：`FROZEN_CONFIG.yaml`（sha256 记录在每个规划器日志的 `H_POLICY` 行，以及 episode 的 `planner_hello.planner.h_policy`）。
- **任务书**：`../EXP6_V3_H_PLANNER_AI_TASK.md`。
- **修改规则**：任何改动都要新建 protocol 版本，并记入 `reports/DEVIATIONS.md`。

## 1. 唯一冻结模型
- **模型**：V3-T1（V3-VAE 自训练稠密 3D VAE + Scene X-Attn + 2D rectified flow）。
  - checkpoint：`work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt`，sha256 `639c9d3a…`，EMA@40k；
  - 与 CVPR Table 1 的 V3 行是同一文件；
  - 本实验**不训练**。
- **推理**：与 Table 1 V3 行完全相同。
  - P1：D100 部分 TSDF 4× 网格作为模型输入，S10 稀疏孪生地图的 `band_dil` 用于 DistP1（b2 模式）；
  - R10，K = 4，Euler 10；
  - 噪声键为 `seed_key(scene, start, rep, t_refresh, k)`，一步内所有候选共用；
  - 456 × 256 渲染，64 × 114 条评分射线；
  - 候选：模拟器的合法列表（get_neighboring_poses → get_valid_neighbors（未访问优先）→ GT 线段碰撞）；
  - 选择：`m_score.choose`（取最大、并列取最小索引、全零走带种子的回退）；
  - 100 步。

## 2. 四种策略（逐级嵌套；代码 `scripts/h_policy.py`，只在规划器进程内替换 `v2_plan_core.score_state` 的打分）
| 策略 | 每样本价值 g_k(c) | K 聚合 | 时域 |
|---|---|---|---|
| **H0**（= Table 1 V3） | 冻结 DistP1：域内命中、在 S10 `band_dil`（±1 P1 体素）之外，按 P1 体素（0.011 模型单位）去重，权重取 max w(z) | 均值 | 1 |
| **H1** | 域内预测命中（判定与 DistP1 相同）转到运行时坐标；与**所有已执行帧的 S10 采样点**都相距 > eps_cov = 1.0（官方覆盖容差）才算新；在边长 1.0 的格子上去重，权重取 max w(z)（w 与 z0 和 DistP1 相同） | 均值 | 1 |
| **H2** | 同 H1 | mean − 0.25 · std（ddof 0） | 1 |
| **H3** | 同 H1 | 两步：每个样本取 c1 与 c2 新格子的并集（每格取 max 权重），Q = mean − 0.25 · std；c1 的分数取各 c2 中 Q 的最大值；没有 c2 时用 H2 分数 | 2，只执行 c1，逐步重规划 |

**H1 的依据**（审计数字见 `SCORING_AUDIT.md`）：
- H0 的“已见”判定半径只有 ±1 个 P1 体素；
- 官方覆盖容差换算成 P1 体素后，在 eiffel 为 2.7 个、在 sestino_museum 为 20.5 个；
- 所以 H0 会把与已观测表面相距不足一个覆盖容差的预测体素也计为新的，并且在同一个覆盖单元内重复计数。
- H1 只使用智能体自己的 S10 观测点（服务记录的 `sample_points`，已执行帧 0..t）；不使用 GT mesh、评价采样点或未来深度。

**H3 第二层的构造**：
- **c2 的来源**：c1 在设置网格上的邻居（MAGICIAN `get_neighboring_poses`：6 个平移 + 4 个旋转，边界截断，方位角取模），按未访问优先（c1 计为已访问），排除 c1 自身。
- **阻挡判定**：c1 → c2 的线段若穿过当前 D100 P1 的已知表面带则丢弃；未知空间视为可通行。
- **不读取 GT 碰撞**。
- **位姿表**：c2 的 (R, T, 中心) 来自 `build_pose_tables.py` 用 MAGICIAN 自己的相机代码从 settings.json 生成的位姿表，已与历史服务记录逐位核对。
- **预测**：c2 复用当前缓存的 V3 特征和同一组 K 个噪声图（开环、P1 冻结）；跨视角的样本 k 配对只是蒙特卡洛近似，不是联合世界后验。
- **搜索**：第一层是全部合法候选，第二层穷举（不剪枝）。

**回归变体**（不进主表）：
- `h1deg`：H1 代码路径加 H0 价值 → 必须复现 H0；
- `h3d1`：H3 代码路径、深度 1 → 必须复现 H2；
- λ = 0：H2 退化为 H1（代码恒等，并在记录中检查）。

## 3. 评估
- **场景与起点**：Table 1 的 All-14（Train-6、Dev-5、Test-3；sofia_church 为 PENDING）× 起点 0–4 × 100 步。
- **顺序**：先 Dev-5（每个策略 25 个 episode），再 Train-6 与 Test-3。冻结后不再按任何结果修改配置。
- **运行方式**：由 CVPR Table 1 自己的队列 `t1_queue.sh` 运行（模型 ID V3-H1 / V3-H2 / V3-H3），经 `t1_out_cov` 官方覆盖重放（必须与存档曲线逐位一致）后导入规范记录。
- **H0**：Table 1 V3 行的 70 个 episode。H 运行器以 h0 策略重跑 barts_s0，必须与之逐位一致。
- **覆盖口径**：MAGICIAN 官方 S10 为主，单调 S10 为辅，分开列出。
- **AUC 与 Final**：AUC@k = trapz(C[0..k]) / k；Final = C[100]。
- **统计**：同一场景的 5 个起点先求均值，场景等权；场景级配对 bootstrap 95% CI（10k，seed 0），报告胜 / 负场景数。
- **配对**：
  - H1 − H0、H2 − H1、H3 − H2；
  - H1 / H2 / H3 − H0；
  - H0–H3 − B0 / B2 / MAG-B。
- **G1**：共享的 V3-T1 预测，引用已有的 V3 固定状态结果，不重复计算。
- **G2**：Dev 固定状态上的单步诊断，只作解释，**不是 G3 的门槛**。
