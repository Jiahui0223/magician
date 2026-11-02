# Seen2Scene / EXP6 — V5.2 Architecture & Focused Pilot

**版本**：V5.2 · Camera-Conditioned Ray Geometry Belief for Direct NBV Scoring  
**状态**：已按用户最近讨论形成初版可执行设计；这是给拥有 `seen2scene` 代码与 BSU HPC 的代码 AI 的实施说明，不代表训练已执行。  
**参考**：现有 `experiments/exp6_v5/` 的 `FINAL_REPORT.md`、`ARCHITECTURE.md`、`FROZEN_PROTOCOL.md`、`LABEL_QUALITY.md`、`CHECKPOINT_LINEAGE.md`，以及先前 `EXP6_V5_ARCHITECTURE_DESIGN.md`。请以真实仓库和冻结协议为准。

## 0. 一句话目标和不可改变的边界

**训练** Camera-Conditioned Multi-Surface Geometry Belief；**推理**直接从 Belief 计算：

\[
\boxed{S_{V5.2}(C)=S_{visible}(C)+\lambda S_{potential}(C)}
\]

其中 `visible` 是当前候选 Camera 的**预计可见新增表面**，`potential` 是该方向关联的**遮挡后潜在新表面 + 未探索潜在空间**。**不生成 Future Depth 图、不调用 2D Rectified Flow、不运行旧 DistP1 打分**。为计算当前可见价值，内部可以计算可见性/First-Hit **概率分布**，但不得又退回“先输出 Depth 图，再调用 DistP1”。

- V5.2 属于 **Idea 2：Ray Belief → Camera Utility**；V5 保留为 **Idea 1：Belief-Conditioned 2D Flow → Future Depth → DistP1** 的对照。
- 输入是由**已经执行的真实观测**融合的 `P1_D100`，继承 V5 3D U-Net、Ray Query、Geometry-Guided Local/Coarse Retrieval、Cross-Attention、Multi-Surface Belief。独立的 **Exploration Memory 暂不做**。
- 同一历史轨迹产生的 `P1_S10` 保持可用，用于 **novelty known-band mask** 与原 V4/V5 可比；`P1_D100` 用于网络输入及 FREE/UNKNOWN/可达边界几何。两者不可由已融合 Dense BAND 简单下采样替代。
- 未来真实 GT Mesh/GT Coverage **只能用于训练标签、离线评估和配对结果计算**，不可用于推理的 Score、候选生成或 P1 历史更新。
- **不增加** Final Coverage Loss、Camera Score Loss、Ranking Loss、Flow Matching Loss；仅训练多表面和 First-Hit 两个明确目标。Planner 仍使用原可执行候选集/运动合法性、`n_win=5`、真实深度融合、Coverage evaluator，不替换全套规划器。
- 目标是**小规模验证**，不是一次性训练完整新系列，不做 U0、Indoor FT、V5 从零训练或多组新 Attention。

## 1. 已有证据（不能把预期当作事实）

V5 的完整测试：Outdoor AUC 0.711、Final 0.891；Indoor Zero-Shot AUC 0.333、Final 0.501；与 V4C 的闭环结果不可区分。V5 `Surface Belief` 的第二及后续表面召回：训练场景留出轨迹 0.44、Outdoor 0.31–0.32、MP3D 0.28；深遮挡更弱（约 0.11–0.16）。因此 `potential` **必须作为弱、封顶的探索提示**，不能把所有墙后预测直接加到当前 Coverage。此前 V4 的 Dense-score Experiment A 使 Outdoor 严重退化，故评分定义/历史 known-mask 必须明确隔离。

V5 已验证：推理不使用旧 32-pt feature `grid_sample`，合法 Train-6 Mesh 射线多交点监督可用，继承网络可由 `work_dirs/exp6_v5/checkpoints/v5_final.pt` 迁移。**实际读取和 SHA/EMA 核对优先**，不要信任手写路径推测。

## 2. 模型数据流（V5.2 正式架构）

```text
P1_D100 (actual executed history)
     |
V5 3D U-Net -> F0/F1 spatial memory
     |                              Camera Pose + K + pixel rays
     +------------> V5 Ray-to-Memory Geometry Query
                                 |
                          z_r (per-ray latent)
                                 |
              Multi-Surface Belief b_r(t), 0..1
                                 |
            Conditional Hit Hazard h_r(t)   [NEW]
                                 |
              Ordered Soft Visibility / First-Hit PMF
                                 |
        +------------------------+-----------------------+
        |                                                |
  S_visible(C)                                S_potential(C)
  expected newly visible                      occluded surface +
  surfaces in WORLD GRID                      direction-linked UNKNOWN space
        |                                                |
        +------------------------+-----------------------+
                                 |
                   S_visible + lambda*S_potential
                                 |
                      select legal NBV camera
                                 |
                    execute REAL observation
                                 |
                    update history P1_D100/S10
```

没有 Flow branch 的前向调用，没有生成式 K=4，不直接把预想的 Surface/UNKNOWN 当成真实地图。

## 3. 数学语义：必须明确区分三种量

### 3.1 Multi-Surface marginal `b_r(t)`

V5 的 `b_r(t)` 表示沿射线**固定物理容差的表面邻域**存在的边际概率；同一 Ray 上墙 A 和 B 都可能有高值。**`b` 既不是 First-Hit PMF，也不是自由空间占据概率；`1-b` 不等于 FREE，尤其在遮挡物后。**

### 3.2 Conditional first-hit hazard：作为 Visible 的内部概率工具

在每条 Ray 的合法近远量程生成有序的深度**积分区间** `I_i=[t_i^-,t_i^+)`，宽度为 `Δt_i` 米。`t_i` 可采用尺度自适应分层或对数区间，并在近表面增强精度。**这是对低成本 Ray Belief/小型 Head 的查询，不是恢复 V4 的 32-pt 3D U-Net grid sampling**。

建立轻量 `HazardHead`，输入 `z_r`、当前 `b_r(t_i)`、`log1p(t_i/s_voxel)`、`Δt_i`、射线近远、局部观测状态：

\[
\rho_{ri}=\mathrm{softplus}(g_\phi(z_r,b_r(t_i),t_i,\Delta t_i,\ldots)),\qquad
h_{ri}=1-\exp(-\rho_{ri}\Delta t_i).
\]

\[
T_{r1}=1,\qquad T_{ri}=\prod_{j<i}(1-h_{rj}),\qquad
p^{first}_{ri}=T_{ri}h_{ri},\qquad p^{nohit}_{r}=\prod_j(1-h_{rj}).
\]

- `h` 的含义是“前面没有命中时，该区间发生第一个命中”的**条件**概率，而非 `b` 的原始边际表面概率。
- 强制 `sum_i p_first_i + p_nohit = 1`（在有效量程内）、浮点稳定（`log1p`/累计 log survival、float32 累积、clip 极值），相机 pose/尺度换算正确。
- 不要把 `p_first` 压缩成期望深度并反投影。它**直接为 World-Voxel 的概率新表面收益提供质量权重**。
- `b` 继续输出并监督所有可信多交点，包括第一次命中后；在 Visibility 内发生的降权**不能反向修改**“墙后存在概率”的语义。
- 如选择自适应区间数量，上限由性能审计决定（建议先 48–64 个、分块计算，必要时近表面细化），记录相对表面宽度、速度和体素精度；不可在未审计的前提下将 16 个很粗 probe 直接当作高精度 First-Hit Score。

### 3.3 Visible：概率化的新增表面**世界空间**覆盖

将每个 `(r,i)` 的世界位置 `x_{ri}=o_C+t_i d_r` 映射到既有评价对齐的 3D 体素。排除有效场景边界之外的点和历史已知 BAND 邻域；novelty mask 首版沿用原 DistP1 的 `P1_S10` `band_dil` 规则，同时用 `P1_D100` 已确认 FREE 检查物理冲突（若引入额外掩码，单列报告，不能暗改）。

重要：**不能把每条 Ray 的概率简单求和**，因为许多 Ray 可能对应同一世界体素。首版保守做法：同一 voxel 的候选命中概率取 `max`，而非重复计分；如源码已有可靠面积或概率 union 估计，可提供对照并说明其对共享场景相关性的假设。`max` 是评分代理，不等于精确的多射线联合可见概率。

\[
S^{raw}_{visible}(C)=\sum_{x\in \mathrm{novel\_voxels}}\max_{(r,i)\mapsto x}
\left[p^{first}_{ri}\,w_{dist}(t_i)\,q_{valid}(r,i)\right].
\]

**距离权重首版沿用**冻结原 DistP1 的 `w_dist(z)=min(1,(z/z0)^2)`（`z0` 必须从仓库审计，不凭空指定）。这样对原先离散 Ray-Hit 计分仍有可比较的物理尺度补偿/远处偏好。**不要同时将每个命中体素乘一次完整像素面积 `z²/(fx fy)`，造成距离面积补偿重复。** 后期若改成完整真实表面积测量再重审权重。注意这仍是启发式的 Coverage 代理，不能声称是严格的真实物理面积期望。

### 3.4 Potential：两种信息来源，只有一个最终 Potential 分数

\[
P_{potential}(C)=\eta\,\widehat P_{surf}(C)+(1-\eta)\,\widehat P_{space}(C),\qquad\eta=0.5\text{ (Pilot default)}.
\]

**A. Occluded surface potential**：在历史 P1 中尚未确认的潜在表面，利用 `b_r(t_i)` 表示其存在信念；用 `1-T_{ri}` 标识它相对当前 Ray 可能位于先前遮挡后。但 `b_i(1-T_i)` 仅是**启发式联合代理**，并非精确 `P(surface AND occluded)`（相关性未知）。同一体素去重，保留相对于当前候选 Ray 的方向关联，并对远遮挡表面低精度施加距离/置信度折扣。**不得把该项计为当前可见 Coverage**。

**B. Unexplored-space potential**：从**历史 D100 P1 的 UNKNOWN** 出发，计算与候选 Camera 朝向相关、并与**当前真实观测 FREE 的可达边界/frontier** 有关联的未知体素/空间范围（去重，按体素体积或固定值计数）。在候选方向前方，经 `T` 表明较可能通畅的区域优先；被模型预测遮挡且已确认无合法可达路径的区域计 0；可达性**尚未确认**且有观测支持的边缘区域允许小的非零权重。只使用合法相机位置与已执行观测的 FREE 连通信息，不求未来 GT、不声称进行了长时域路径优化。**不能把 `1-b` 当作自由空间预测**。

这两项物理单位不同（表面体素 vs 空间体素/体积），必须先分别归一化到可比较范围：从 **Train-6 或同一待决策候选集合的可观测预测分数** 估计尺度，不用 Test-3/MP3D 的 GT 做标定；对异常大的 Potential clip，并保存原始两项值。

最终建议：

\[
\boxed{S_{V5.2}(C)=\widehat S_{visible}(C)+\lambda\left[0.5\widehat P_{surf}(C)+0.5\widehat P_{space}(C)\right]}
\]

`lambda = 0`（Visible Only，核心可靠性对照）及 `lambda = 0.10`（Full V5.2，**同一模型权重，只调整推理评分**）。`lambda` 不是 Loss 权重，不能用每个测试场景的 GT 来挑。**λ 小只有在两部分已经尺度归一化后才成立。** 严格记录加权潜力占总分比例、候选分数接近时的翻转率、潜力驱动的新增停滞/回访行为。

> 边界：即使增加 Potential，这仍是**不带 Exploration Memory 的局部启发式 NBV**，不能保证长走廊被保留并在几十步后重新访问。空间潜力不是当前一步真实 Coverage。失败场景必须报告虚假墙后收益和无法到达的路径。

## 4. 训练目标：两项，仅两项

从 V5 已有稳定多表面标签缓存复用 `Train-6 mesh depth-peeling`（16 层、最多 8 个可信表面求交；实际数量遵循报告）。

\[
\boxed{\mathcal L=\mathcal L_{surface}+\beta\mathcal L_{firsthit}}
\]

- `L_surface`：沿 Ray 多表面存在性，在真实 Mesh 完整可信的 interval 上监督（包括第一次命中后）。沿用 V5 合法标签 Mask、正例重加权/采样，不得把未观测且 Mesh 不完整的区间自动标负。
- `L_firsthit`：对真实首次可见命中 interval `i*` 最小化 `-log(p_first[i*]+eps)`；若**可信的有效量程内真实 No-Hit**，监督 `-log(p_nohit+eps)`；无效/截断/透明/外部深度没有确证时 Mask。真实 GT 仅用于 loss，不作为网络条件。
- `β=1` 作为量纲归一化后的初始化目标；先记录两项样本平均和梯度范数，若比例严重失衡，**只根据 Train-6/S-c 训练监控决定一次合理重标定并冻结**。避免增加 `L_coverage/L_rank/L_flow/L_KL`。
- `SurfaceHead`、`HazardHead`、Ray Query、3D U-Net 确保梯度有效。保留 V5 多表面能力是必要的门槛，不可只训练 Hazard 让后方 belief collapse。

## 5. Warm-start 与最小 Pilot（无需同时开展多条完整训练）

**父权重**：`work_dirs/exp6_v5/checkpoints/v5_final.pt`，先核对 SHA 与 EMA，V5 已训练 12k。载入并继续训练：3D U-Net + Ray Query + Multi-Surface Head；**不加载或调用 Flow 的预测分支**。新增 `HazardHead` 权重初始化，保留原 checkpoint 独立不可覆盖。

建议预算（仅初始建议，源码审计可报告必要变化）：

1. **Stage H：2k steps**。冻结 V5 Scene Encoder、Ray Query、SurfaceHead，仅训练 `HazardHead`，检查 First-Hit NLL、No-Hit 校准、输出概率和数值稳定性。
2. **Stage J：4k steps**。小 LR 解冻 U-Net/Ray Query/SurfaceHead，联合训练 `L_surface + β L_firsthit`，跟踪 S-c 第二表面召回、First-Hit NLL、假阳性与梯度。加载父 EMA 时明确新 Head 的初始化与 EMA 规则； optimizer 状态不相容时不要伪称精确 resume。
3. **只训练一次 V5.2**。`lambda=0` 和 `lambda=.1` 完全共享这个 checkpoint，作为**免重训的评分分支对照**，切勿训练两遍。

初始训练总计 6k steps。不能仅凭 train loss 变低就声称 NBV 改善。若旧 V5 的 Surface Belief 在 Stage J 明显崩溃，应报告、回看中间 checkpoint，不得静默继续高成本全量 G3。

## 6. 实验与停止门槛（快速出结论）

### G0：正确性（必须过）

- checkpoint 父权重/SHA、EMA、Train-6 Mesh label cache、P1_D100/P1_S10 分离，世界/相机坐标与 near/far，GT 不泄漏。
- **两堵墙 3m/9m**、无 Hit、单 Hit、多 Hit、已知 BAND、已确认 FREE、多 Ray 命中同一 voxel、近远尺度变化、EMPTY/UNKNOWN、超量程/无效 Depth、没有 Local Memory Token，各造合成测试。
- 有序可见性：若前方确定不透明墙 A，墙 B 不计 Visible，但可在 Multi-Surface 与低权重 Potential 中保留；若 A 的概率降低，墙 B 的可见概率可合理增大。
- `p_first.sum()+p_nohit == 1`，无 NaN/Inf；变更数值积分 bins 时分布与评分不应随意大变。
- **证明 G3 用 V5.2 时 Flow forward/sample 次数为 0、旧 DistP1 调用为 0**，不生成 Depth；不改变真实环境状态更新与合法动作数。原 V4/V5 重新加载原评估框架可回放若干固定状态用于平台确认。
- 禁止大量 GPU 大跑前未保存 G0 测试证据。

### G1：Belief / Visibility（保留两个任务的评估）

- 多表面 Recall/Precision/Brier/ECE（训练 S-c、Macarons++ Dev-3、MP3D VALSEL），包括第二表面、距离第一表面 ≥128 voxel 的远遮挡表面；不可只展示总体 loss。
- First-Hit PMF NLL、Top depth bin accuracy、No-Hit Calibration、适用的 Depth interval error；**不把这理解为和 Flow 的 MAE 完全同一接口**。
- Query/Surface Head/Hazard Head 延迟、GPU RAM 和吞吐。

### G2：配对固定状态 Camera Selection（优先）

- 同一真实历史 P1、同一候选列表、同一坐标约定：V5.2 `lambda=0`、`lambda=.1` vs V5/S10、V4C/S10、B2/E2 已有固定状态可引用的严格对应集。
- Spearman / Top1 / 真实 selected one-step Coverage Gain（pp）、与基线选择不同动作比例；按长 Ray/近 Ray、UNKNOWN、多层遮挡、Outdoor/Indoor 拆分；`S_visible`,`P_surf`,`P_space`, weighted Pot%、lambda score 翻转记录。
- 真实增益 GT **仅作指标**，不用于评分。

### G3：轻量闭环，再决定要不要完整评估

- 快速 smoke：Macarons++ Test-3 中预先固定 **2 场景×2 初始位姿**（4 episodes，100 steps）和 MP3D VALSEL 中 **2 楼层×1 初始位姿**（2 episodes，150 steps），两种 λ、同种子/候选协议。提前冻结选用的 ID，不根据结果选有利场景；此处为探索诊断、不是正式统计结论。
- 报告每个 episode AUC、Final、Cov20、停滞、Score 分量和改变动作原因；对 λ=0、.1 的各类失败轨迹进行检查。
- 若 G0/G1 合理且 G2/小 G3 有可用信号，再在**相同冻结 checkpoint**做标准 Macarons++ 6 scenes×5 starts×100 与 MP3D 5 floors×2 starts×150 全量配对 G3，计算 scene/floor 级 bootstrap CI。不因跑完以后失望就静默改 λ/η/区域定义重跑同一评估集。
- 即使没有提升也交付完整 G2 + 小 G3 报告；必须明确未完成的全量 G3，**不能将 smoke 解释为正式 SOTA 结果**。

### Baseline / 科学比较

- V4@40k、V4C、V5、B2、E2-D100 和 MAGICIAN 的现有 G3 是历史比较，训练来源不同；不可将 V5.2 的 `lambda=.1` 优势全部归因于新的 Geometry Query，因为 V5.2 同时更换了训练目标/打分函数/是否使用 Flow。
- `V5.2 λ=0` vs `V5.2 λ=.1`：隔离 Potential 的**推理阶段作用**。
- `V5.2 λ=0` vs `V5/S10`：比较直接几何可见评分与 Flow+DistP1 **整套系统**，并非单纯模型变化消融。
- 既有 test scenes/MP3D 曾被多次使用，**非盲测**；不做测试场景超参搜索；仍需真正独立场景确认泛化。

## 7. 文件结构与交付要求

```text
experiments/exp6_v5_2/
  README.md
  ARCHITECTURE.md
  AUDIT.md
  FROZEN_PROTOCOL.md
  CHECKPOINT_LINEAGE.md
  configs/
    train_hazard_pilot.yaml
    train_joint_pilot.yaml
    score_visible_only.yaml
    score_visible_potential.yaml
    eval_fixed.yaml
    eval_smoke_g3.yaml
  scripts/
    audit.py
    train_v5_2.py
    eval_g1_g2.py
    eval_g3_smoke.py
  tests/
    test_firsthit_distribution.py
    test_visibility_and_occlusion.py
    test_voxel_dedup_and_distance.py
    test_space_potential.py
    test_no_gt_leak_no_flow.py
  reports/
    G0_VALIDATION.md
    G1_BELIEF_AND_FIRSTHIT.md
    G2_FIXED_STATE.md
    G3_SMOKE.md
    FINAL_REPORT.md
  figures/
work_dirs/exp6_v5_2/
  checkpoints/
  runs/
  eval_fixed/
  eval_smoke/
  logs/
```

可复用现有仓库公共模块，不要为满足示意目录强制复制代码。最终提交：Git SHA、训练/评估命令、父权重 SHA、模型参数量、GPU-h、Ray scoring 耗时、每阶段 loss/召回/NLL、G2 所有候选排名结果、G3 trajectory/coverage 曲线、失败场景分析。不要覆盖 `exp6_v5/` 或 `exp6_v4/`。

## 8. 给代码 AI 的最终执行原则

**先完成源码审计与 G0、再训练单条 6k V5.2、再离线评分 G2、最后轻量 G3；不要跳过关键概率语义和 P1 去重校验。** 优先复用 Train-6 已验证多表面标签和 V5 父权重。只有 `L_surface+L_firsthit` 两项训练目标；`lambda` 只在推理。若原实现不能提供计算 `P_space` 所需的合法运动/连通信息，可先用“真实 D100 FREE 邻接 UNKNOWN frontier + 当前候选 Ray 几何关联”的保守近似并完整记录，不能声称做了全局探索或未来可达性学习。

**停止并报告阻塞的情况**：需要使用 GT eval mesh 才能得到 Score、概率不归一化/NaN、模型预测写回历史 P1、同体素重复计分无法控制、所谓空间潜力实际仅为 `1-b`、使用了已删除的 Flow、缺少可信训练/评估标签、错误使用 D100/S10 时间历史。对于普通实现参数差异，记录后采用合理可复现默认值即可，不必为了小的工程细节反复询问用户。
