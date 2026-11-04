# EXP6 / V3 / H — Coverage-Aware Decision & Short-Horizon Planning

**完整 AI Coding Agent 执行任务书｜版本 H-v1 / 2026-11-04**

> **任务核心：冻结已训练的 V3-T1（3D VAE + Scene Cross-Attention + 2D Rectified Flow）模型，比较 H0/H1/H2/H3 四种 NBV 决策策略，完成与既有 CVPR Results/Table 1 完全统一的 Macarons++ Train/Dev/Test/All-14 全场景闭环评估。** 本轮默认不训练、不修改 VAE、Flow、P1 更新和历史模型权重，只调整必要的评分/规划逻辑。
>
> **用户明确需求：不要把实验设计成 G1/G2 一旦不理想便跳过 G3 的 GO/NO-GO 流程。** 四个策略都要拿到真实的完整覆盖曲线与 AUC@20、AUC@100、Final Coverage；合法运行中遇到的失败/资源问题须重试、续跑，不能捏造结果、不能用短程曲线预测 Final。
>
> **数据优先级：官方 MAGICIAN S10 Coverage 评价指标为主；统一训练/推理/评分/评估的定义，尤其 V3-T1 的 D100 推理输入与 S10 band_dil 评分不得混淆。** 下文所述“物理对齐”只能使用推理时真实可得的 P1、真实历史观测、已知相机内外参、合法动作图及公开不泄漏未来场景的元信息；严禁推理时调用 GT Mesh、未来真深度、Oracle Coverage、评价器内部未观测真值。

---

## 0. 开始前的项目审计，不做推断式实现

### 0.1 位置和命名

**实验代码、结果、报告：**

```text
experiments/exp6_v3/H/
```

请保留历史 `exp6_v3/T0`、`T1`、`exp6_v3.1`、`exp6_v5/F1`，不修改旧模型。新实验名为：

- **H0** — `V3T1-DistP1-Greedy`：官方历史 V3-T1 推理/评分/选择行为（原样复现）。
- **H1** — `V3T1-CoverageAligned-Greedy`：在相同 V3 预测基础上，用推理可用的几何信息改进新表面价值估计，仍为一步 Greedy。
- **H2** — `V3T1-UncertaintyAware-Greedy`：以 H1 的几何价值为基础，利用 K=4 的**跨样本风险/一致性**，而非重复原来的求均值。
- **H3** — `V3T1-ShortHorizon2`：保留 H2 的单视角价值定义，在合法动作图上增加**两步、空间去重、每次真实观测后重规划**的规划机制。

**重要：这三个修改是逐级嵌套的：H1 vs H0 看 coverage proxy；H2 vs H1 看多样本风险；H3 vs H2 看短时域规划。** 在报告中不能错误归因为“ H3 仅比 H0 多了 Beam ”。如需额外组合消融，必须单独命名，不得污染 H0–H3。

CVPR 主评估位置按实际仓库审计为准，历史项目预期为：

```text
experiments/cvpr_results/table1/
```

用户也称其 `CVPR-Results`。**必须先 `find` / 查配置确认真实存在的路径，而不是新建另一个大小写不同的目录。** 不允许覆盖或手动改动既有 `TABLE1.md`。接入已有 model registry、episode runner、coverage evaluator、aggregate、paired/bootstrap、plot 脚本，让它们自动产出新增行。

### 0.2 先确认四条事实

1. **唯一主模型 = 历史 V3-T1 的已选定 checkpoint**，不是 V3-T0，不是 outdoor-only V3.1，也不是 B2/V5/F1。实地核对完整权重路径、checkpoint SHA256、训练步数、model config、VAE 类型、2D Flow 类型和加载方式。`V3-T1` 历史是 Mixed scratch 40k + VAE/Scene Cross-Attention；**不要重新选择 checkpoint**。
2. **共同主推理协议**（须按 CVPR Table 1 现有冻结配置审计）：D100 P1 用于 V3 的 3D 场景编码/推理，S10 band_dil 用于 DistP1/其他评分，R10 更新 3D 缓存，K=4 Flow 样本，Euler 10 steps，图像渲染 456×256，Ray Grid 64×114，Greedy-NV 原合法/未访问规则，Outdoor 100 步。禁止根据模型/策略动态改这些条件。
3. **Coverage 主指标**：由 CVPR Table 1 的 MAGICIAN 官方 S10 coverage 评估器，按官方定义计算，所有模型同一 evaluator；次级可计算项目现有 monotonic S10 coverage，但**分开列、不混合**。S10 是评价采样规则，不等于 D100 的 P1 输入密度。
4. **当前数据覆盖**：历史 Table 1 为 Train-6 (6×5=30 episodes)、Dev-5 (5×5=25)、Test-3 (3×5=15)，总计 **All-14 = 70 episodes**。`sofia_church` 尚缺失，因此不能输出完整 All-15；若以后恢复，按同一冻结协议补齐并新生成 All-15。

### 0.3 历史 V3-T1 对齐参考（只作为回归参考，不允许硬编码成果）

旧 `TABLE1.md` (MAGICIAN 官方 S10 主协议)：

| Scope | AUC@20 | AUC@100 | Final |
|---|---:|---:|---:|
| V3-T1 / Dev-5 | 0.384 | 0.748 | 0.897 |
| V3-T1 / All-14 | 0.384 | 0.744 | 0.903 |

上述只用于对照历史报告。实际 H0 必须由真实自动脚本或逐位可复用的**同配置、同 checkpoint、同起点**轨迹确认；如存在差异先排查 evaluator、版本、seed、覆盖口径，而不是改指标或忽略差异。

### 0.4 必须审计 K=4 聚合，避免 H2 无意义

历史 `ABLATION_SUMMARY.md` 已验证：**“K=4 均值 = 四个样本分数的均值”**，并指出“样本分数离散度与预测误差相关”。

因此：

- 原 H0 若已实现 `mean_k(DistP1(D_k))`，H2 不能再把 `mean_k(score(D_k))` 称为新方法。
- 逐行审计当前 V3-T1 `Core.sample`、`score_state`、`new_set_from_depth`、K 维 reduce 次序、soft/hard hit、score reduce、固定 RNG 键；提供定位路径和代码行号。
- H2 要引入**确实不同于 H0/H1 的随机决策项**，如基于样本级新表面增益的风险校准 `mean − λ·std`（下文给出冻结默认值），并可分析 3D voxel 共识，但不能声称这与纯“期望 Coverage”完全相同。
- 单独跑一个自动化恒等测试：`lambda=0` 时 H2 对 H1 的样本级平均应逐位/数值严格一致（若 H1 是样本级均值定义）；`lambda>0` 应在不一致状态产生可追踪的分数变化。

---

## 1. 四个方法的精确定义

### H0：冻结基线，必须逐位复现

- 调用原 V3-T1 推理，保持目前的 D100 特征图 / S10 评分图双分支与原来所有默认规则。
- 使用原始 DistP1，原已访问/合法候选过滤，选最大分数。
- 记录每个 step 的候选、4 个 sample 的 Depth/Hit/分数、最终选择、真实新观测及官方 Coverage。
- 如果 CVPR Table 1 已经有完全相同的 V3-T1 episode（checkpoint/config/seed/场景/起点/步数/评估器均一致），通过内容 hash、schema 及逐位回归检查后**复用**，不重复跑；不一致时重跑并单独记录原因。
- **H0 的参考行必须能被 Table 1 自动回归重现**，否则不得继续进行对照结论。

### H1：Coverage-Aligned Scoring（几何新表面价值，不用 GT）

目的：修复“DistP1 的预测新体素 ≠ 真实地图有效新增覆盖”的代理偏差，而不是简单删除有效的距离权重。

对每个候选 c、Flow 样本 k：

1. 通过已有 V3 解码，获得该视角预测的 first-hit/no-hit 和深度 `D[c,k]`。**不要对无命中点伪造深度**。
2. 投影到当前 P1 的三维坐标系，保留原来的合法 domain、深度范围、视角内有效性和物理可见性检查。
3. 由目前**真实已观测**的 S10 表面点/BAND 构造覆盖代理 `S_seen`；不要使用 GT 全场景 Mesh、evaluation sampling 顶点、未来深度、Oracle。
4. 对预测新表面三维体素去重。保留原来的距离加权 `w(z)=min(1,(z/z0)^2)`；采用与 P1 voxel size / 官方可见覆盖容差相容、事前固定的观测邻域尺度，用**软或硬的几何新颖性** `novelty(v,S_seen)` 判断预测命中是否与历史真实观测重复。
5. 如需排除不属于目标有效覆盖域的区域，必须使用所有策略同样可见的**既有合法评价域的公开元信息**（例如原始规划 domain / 裁剪规则），不能调用 GT Mesh 标签、GT 表面 ID 或按照 Test/Train 结果人工手画遮罩。若元信息本身包含未观测 GT，则不得使用，保留原 domain。
6. 推荐主公式（仅供实现目标，具体高效函数需按代码实际审计确定）：

   `G_H1(c,k) = Σ_{v in unique predicted hits} w(z_v) · novelty(v, S_seen) · valid(v)`

   其中 `novelty∈[0,1]`，`valid` 仅来源于合法、真实可用观测状态，重复体素只计一次；若使用 per-voxel max weighting，需与 H0 维持一致。
7. 用与 H0 **相同的 K=4 样本、同样的 sample-mean 聚合** 得到 `G_H1(c)`，选最大。此处 H1 改的只有**每样本的几何评分定义**，不改变聚合规则。

**尤其注意**：不要再次乘一个无解释的 `z^2` “表面积系数”，导致在原 `w(z)` 已补偿的基础上重复加权；如要讨论面元投影和法向，可另做诊断，不作为主 H1 的隐形变量。

**H1 科学定位**：这是从推理可获得的几何观测构造的 *coverage-aligned proxy*，不是精确的真实 expected coverage。实现/论文都不许把代理数值与 GT Coverage Gain 混称。

**应有回归：** 当 `novelty` 特化为当前 `band_dil` 硬排除、`valid` 用原 domain、其他规则退化为 H0 时，H1 应复现 H0 的候选 score/ranking；若无法复现，应先修复坐标或判重等 bug。

### H2：Uncertainty-Aware Expected Gain（多样本均值 + 可靠性）

**重要区别：H0 和 H1 已使用了 4 样本分数均值。H2 的新意不在“再平均四个 Depth 分数”，而是在利用四个假想世界之间的差异，避免被高不确定度的幻象新表面误导。**

- **完全复用 H1 每样本的表面价值 `g_k(c) = G_H1(c,k)`**；不改变 Depth、候选集合、P1、距离权重、样本数 K、噪声。
- 定义 `mu(c)=mean_k(g_k(c))`，`sigma(c)=std_k(g_k(c))`（明确 `ddof=0`，数值精度一致）。
- 冻结一个主要版本：

  `S_H2(c) = mu(c) - lambda_risk · sigma(c)`

  例如可将 `lambda_risk=0.25` 作为**预注册试验初值而非已证明最优**；若沿用项目其它评分量纲，应先用同一固定状态分布审计尺度，不通过 Test/All-14 调参。仅允许事前固定小范围开发集敏感性（例如 λ∈{0,0.25,0.5}），**主表 H2 只保留冻结 λ 对应的单一模型行**。
- 可额外计算三维 voxel 共识指标：4 次生成中某个候选新表面 voxel 出现次数、分歧率、no-hit 率，用来解释 `sigma` 与错误候选的关系；**不能把对 GT 的真实正确性当成推理时共识标签**。
- 如果历史 H0 的 K samples 并不是四个独立合理世界，而强相关，则如实报告样本相关性，不要夸称 epistemic uncertainty 或 calibrated probability。
- **H2 与 H1 的唯一主差异就是风险敏感决策**。测试 `λ=0` 复现 H1；`λ>0` 只改变评分，不得改模型预测及几何集合构造。
- 预先考虑 H2 可能过度保守：需要展示选择变化率、分数 margin、按 t≤20 与 t>20 的收益，勿假定一定提高 early AUC。

### H3：Short-Horizon2，重规划、合法动作、三维去重

目标：在 H2 的多世界几何价值基础上，让 Planner 允许一个**立即收益较小但下一步更有潜力**的动作。不应简单将两个相同墙面的 DistP1 分数相加。

**主版本建议：两步有限搜索（receding horizon = 2），不是完整 MAGICIAN 10×10 Beam，也不是声称已实现真实 belief-space posterior updating。**

- **第 1 层候选**：当前状态下与 H0/H1/H2 完全一致的合法候选 Camera 集合（不许扩大合法动作集、跨障碍 teleport、改变站点邻接图），默认候选个数沿用仓库真实逻辑，不能自行写死 4；如通常约 4，可记录实测数量。
- **第 2 层候选**：从每个假设第 1 步位姿沿**已知合法图**能到达的相邻位姿；禁止使用未来真实深度/GT 来判断哪个动作最优或可见。
- **两步覆盖集合**：令预测的新表面体素集合为 `U(c1,k)` 和 `U(c2,k)`，在同一空间尺度和体素坐标下计算 `U1 ∪ U2`，相同表面只算一次；第一步预测的已“发现”体素在第二步的新增价值中应扣除，不允许同一墙面重复领奖励。
- **预测模型状态：主实现为 frozen-P1 open-loop two-step proxy**：两步均可使用当前真实 P1 的已缓存 V3 编码和候选 Camera 预测；搜索时只在评分集合中对 `U1` 做虚拟去重，不向 V3 的 D100 P1 伪造高分辨率真实观测，不执行不具备真实样本依据的虚假 D100 融合。每执行第一个真实动作后，获取真实 depth、正常更新 D100/S10 双地图并重新规划下一步。
- **物理一致性限制：** K=4 在不同相机的噪声种子不代表同一真实世界。跨 Camera 同索引样本配对仅是蒙特卡洛近似，不能在论文中宣称学习了全球一致 joint 3D world posterior。记录跨视角几何冲突/重复比例；遇到强矛盾采用合理 fallback。禁止把预测的虚拟表面写入真实地图。
- **主评分**（与 H2 同样的风险项、同样的 novelty / w / valid）：

  `G_union(c1,c2,k) = gain_of_unique_3D_union(U(c1,k),U(c2,k), S_seen)`

  `Q(c1,c2) = mean_k(G_union) - lambda_risk · std_k(G_union)`

  `c1* = first_camera(argmax_{legal (c1,c2)} Q(c1,c2))`

  若图上没有合法 c2，退回 `S_H2(c1)`；同分时采用原稳定 tie-break。每一步只**执行 c1**，真实观测后重规划。
- **固定计算预算和可追溯性**：主 H3 允许的二层拓展策略、宽度/深度、计算上限和是否第一层 pruning 必须在运行前写入 `FROZEN_CONFIG.yaml`。若为节约 GPU 使用第一层 Top-2 短名单，则如实命名 `top2 × depth2`，不要写成穷举；优先复用/caching 当前状态的相机预测结果，批量预测相同位姿，并对吞吐、峰值显存和额外时间负责。
- 不得将未实现的 **真正“将想象深度写回 P1 后重编码再预测未来”** 与本版本混同。真正 belief updating 如将来实现，必须是另一个独立编号；不能悄悄改变 H3 定义。

---

## 2. 永久冻结表：四种策略唯一允许差异

| 设置 | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| V3-T1 checkpoint SHA / VAE / XAttn / Flow | 完全相同 | 相同 | 相同 | 相同 |
| 是否训练模型 / 更换 checkpoint | 否 | 否 | 否 | 否 |
| 真实 D100 P1 场景编码 | 相同 | 相同 | 相同 | 相同 |
| 真实 S10 band/已观测表面及合法规则 | 相同 | 相同 | 相同 | 相同 |
| K / Euler / R | 4 / 10 / 10 | 相同 | 相同 | 相同 |
| 评估分辨率、合法相机、起点、步数 | 相同 | 相同 | 相同 | 相同 |
| 主要 Coverage evaluator | 相同 | 相同 | 相同 | 相同 |
| 每样本评分 | 传统 DistP1 | Coverage-aligned | 与 H1 相同 | 与 H1 相同 |
| K 样本汇总 | 原始均值 | 同样均值 | 均值 − λ·风险 | 风险校正的 2-step union |
| 决策时域 | 1 | 1 | 1 | 2（replan each step） |
| 预测世界状态写回真实 P1 | 否 | 否 | 否 | 否 |

**关于“变量全部相同”的准确含义：** 除研究对象（每样本几何评分、跨样本汇总、搜索时域）外，其他系统变量完全冻结。四条闭环会走向不同位置，所以它们后续遇到的实际 P1 和候选集合不可能逐步完全相同；“相同候选”的严格配对只适用于同一固定状态 G2，不适用于各自独立闭环的所有后续步骤。

## 3. 完整评测计划：先 Dev，再 Train + Test，最后统一 All-14

### 3.1 G0/G1/G2：只读诊断，不是阻止 G3 的门槛

- **G0 回归**：H0 同一 checkpoint、同一候选集合、同一 RNG 的 score、selected action、closed-loop 逐位复现；D100/S10 dual-P1、坐标系、voxel map / K4 sampling、缓存、合法图的断言。
- **G1**：冻结 V3-T1 的 First-Hit、No-Hit、Depth 误差及概率/几何诊断（H0–H3 预测模型完全相同，原则上 G1 不需要四份重复跑，只需证明是共享预测并引用同一组结果）。
- **G2**：固定历史 Dev 状态上的 H0/H1/H2 单步评分及 H3 第一步选择结果：Top-1（并列容差一致）、oracle regret（pp）、Spearman、selected gain、分数 margin、四样本方差与错误关联、选择改变率；只在评估端使用 GT 作为 oracle label；主方法绝不读取 GT。
- **G3**：**四种策略必须获得真正完整的 100 steps episode / Coverage curves / Final**。结果不理想也完整保存。G2 只是解释，不能硬门槛跳过 G3。

### 3.2 先完成 Dev-5，全 5 起点 / 每策略 25 episodes

- **Dev 是开发与解释模型的主要工作集**：完整 `Dev-5 × 5 starts × 100 steps`，不要只跑历史 Dev-3 就声称 Dev-5。
- 先跑与历史固定状态一致的 Dev-3 / 早期 smoke 做回归，但无论结果好坏，在无有效基础设施故障的条件下完成全 Dev-5。
- 展示每个方法的 `AUC@10/20/40/50/60/80/100`、`Cov@10/20/40/50/60/80/100`、Final、per scene、per start、0–20 与 0–100 Coverage 曲线。
- 如果 H2 主 λ 或 H3 拓展宽度在研究中需要选择：**只允许依照预先制定、冻结的 Dev 决策规则选择一次**；选择记录必须包括所有备选结果，不能只保留最高的 Test AUC。Dev-5 场景数较少，结论探索性为主。

### 3.3 冻结全部配置后继续 Train-6 + Test-3

- 主实验必须覆盖 CVPR Table 1 已有**全部 14 个可运行场景**：

  - Train-6：`6 scenes × 5 starts = 30 episodes` / 策略。
  - Dev-5：`5 scenes × 5 starts = 25 episodes` / 策略。
  - Test-3：`3 scenes × 5 starts = 15 episodes` / 策略。
  - **All-14：70 episodes / 策略**；四策略总计 280 episode，其中 H0 若严格验证，可复用已有历史 70。
  - 若 `sofia_church` 将来能够合法下载/运行，在原配置上补齐 5 episode / 策略，再汇总 All-15；在此之前必须标 `All-14` 和 `All15-partial14of15`，不能冒充全 15。

- 所有策略使用同一份 Scene Registry / 实际 scene ID、Camera start IDs、动作图、物理尺度、official evaluator、100 步封顶、相同机器人运动预算。
- 对 Train / Dev / Test 分别先求每场景五起点均值，再对场景等权平均。Train 是 in-distribution，不把 Train 结果宣称为跨场景泛化；历史 Test 已在研究中使用，也不能宣称独立盲测。
- 如果某个策略某些 episode 失败，**不得省略失败片段却仍给它报同样全覆盖率**。列缺失 episode、原因、重试命令、真实最终执行步数；不插值补虚假的 100 步或 Final。

### 3.4 与 MAGICIAN Beam 的正式比较

CVPR Table 1 现有 Beam 基线（原版 10×10、自己的真实 planner）作为外部对照，**不需要把 MAGICIAN 换成我们的 H0–H3 评分**。所有方法采用共同场景、起点、观测预算和官方覆盖 evaluator。

配对汇总包括：

1. H1−H0、H2−H1、H3−H2（逐模块新增贡献）；
2. H1/H2/H3−H0（相对 V3-T1 原基线）；
3. H0/H1/H2/H3−B0、−B2、−MAG-B（完整系统性能，需注明不同架构/训练来源）；
4. 预定义主指标：**AUC@20、AUC@100、Final Coverage**，辅助 Cov@20、Cov@50、AUC@10 与规划耗时。

不允许 H1-H3 使用更密集的 GT S10 Coverage “评估采样”作为规划输入；Coverage evaluator 只能在真实观测/模拟器执行**之后**计分。

### 3.5 统计

- 每个 scene 5 starts 先求均值，然后跨 scene 等权。
- 对相同 scene+start 的真实覆盖曲线做**paired difference**，按 scene cluster bootstrap 95% CI，保持同场景配对关系；单独报告胜/负 scene 数。
- 同时展示逐 scene 的 AUC@20/AUC@100/Final、平均值与 CI，避免单个高方差 scene 支配结论。
- 主报告既显示 Dev-5、Test-3，也显示 All-14；All-14 包含训练场景，不应被叫作“未见过的场景表现”。
- 如果对很多 λ/宽度进行尝试，不要把开发集筛选出的最好数字当作无偏统计证据；在报告明确说明探索性筛选。

---

## 4. 必须加入 CVPR Results/Table 1 的方式

**重要：实验 H 产生的独立文档不够，必须实际接入 CVPR 结果流水线。**

1. 先只读检查 `experiments/cvpr_results/table1/` 内 `PROTOCOL.md`、`FROZEN_CONFIG.yaml`、`SCENE_REGISTRY.yaml`、`MODEL_REGISTRY.yaml`、`scripts/run_eval.py`、`scripts/aggregate.py`、`scripts/validate.py`（实际文件名不同则用真实名称）。
2. **不得编辑或重命名历史模型行**，不得直接通过文本追加 Markdown 表格。新增模型注册信息（H0 若原已有 V3-T1 行，用明确 alias 映射到原行，不创建另一条同义重复数值；H1–H3 新的 model_id 分别为 `V3-H1`, `V3-H2`, `V3-H3`）。
3. 实际在 Table 1 runner 中读取 H1/H2/H3 的 score/planner adapter（复用项目代码，不生成无法加载的伪模型 checkpoint）；注册的字段必须包含冻结 V3-T1 ckpt SHA、policy SHA、score config SHA、P1 D100 / S10、K/R/Euler、horizon、合法动作规则、seed、scenes、coverage evaluator 版本。
4. 把每个 episode 的原始轨迹、逐步真实 Coverage、动作序列、model hash、policy hash、score trace、runtime 通过已有验证器写入或导入 Table 1 数据源，保留 H 子项目的 trace 作为可溯源副本。
5. 调用已有聚合、bootstrap、画图、LaTeX 工具从单一数据源生成 `TABLE1.md` / `TABLE1_FULL.csv` / `TABLE1_BY_SCENE.md` / `TABLE1_BY_SPLIT.md` / 论文 LaTeX 表与图（具体文件存在与否按项目实现核对）。
6. **先自动验证 H0 与旧 V3-T1 的对应 70 episodes 及 aggregate 完全一致**，再加入其它三行；验证全部无误才允许称为“CVPR Table 1 新结果”。
7. 报告本次跑后新增/重用/缺失 episode 数，保证 Train、Dev、Test 分组都完整；没有完成的 scope 标记 `partial`，不可悄悄并入完整总表。
8. 如果当前 Table 1 pipeline 接口无法注册 planner policy，不允许在 H 文件夹造一套“看起来一样”的官方评估器代替；应加最小兼容 adapter，保留原 Coverage evaluator 原封不动。

---

## 5. 目录组织与可复现性

建议结构如下（以现有项目脚本风格适配）：

```text
experiments/exp6_v3/H/
  README.md
  FROZEN_PROTOCOL.md
  FROZEN_CONFIG.yaml
  MODEL_AUDIT.md
  SCORING_AUDIT.md
  DATASET_AUDIT.md
  CVPR_TABLE1_ALIGNMENT.md
  scripts/
    h0_replay.py
    score_h1.py
    score_h2.py
    planner_h3.py
    eval_fixed.py
    run_closed_loop.py
    validate.py
    register_table1.py
    aggregate_local.py
    plot.py
    submit_slurm.sh
  configs/
    h0.yaml
    h1.yaml
    h2.yaml
    h3.yaml
  results/
    per_episode.csv
    per_scene.csv
    by_split.csv
    coverage_curves.csv
    fixed_state_scores.csv
    decisions.csv
    pairwise.csv
    summary.json
  reports/
    IMPLEMENTATION_AUDIT.md
    G0_REGRESSION.md
    G2_FIXED_STATE.md
    G3_DEV.md
    G3_TRAIN.md
    G3_TEST.md
    G3_ALL14.md
    H1_ANALYSIS.md
    H2_ANALYSIS.md
    H3_ANALYSIS.md
    RUNTIME.md
    TABLE1_INTEGRATION.md
    FINAL_REPORT.md
    MISSING_EPISODES.md
  figures/
    coverage_dev_0_20.png
    coverage_dev_0_100.png
    coverage_test_0_20.png
    coverage_all14_0_100.png
    auc_horizon.png
    final_coverage_by_scene.png
    pairwise_early_auc.png
    runtime_vs_auc.png
  logs/
  slurm/
  table1_export/
```

所有输出从 CSV/JSON 原始实测数据自动生成。**不要填充占位数、伪造 episode、硬编码 summary。** markdown/CSV 可以记录运行尚未完成状态，但不能表现为已经完成。

---

## 6. 关键正确性与泄漏检查（必须通过）

- `V3-T1 H0` 原模型/score/trajectory regression；如旧历史缓存可严格复用，记录 sha/样本数/逐位核对结果。
- `P1`：模型输入 D100 与评分 S10 的结构存储相互独立，真实观测同步写入，不能把 D100 BAND 误当 S10 novel-surface mask；H3 假想集合也不能写回真实 P1。
- `K=4`：samples/seeds across H0/H1/H2/H3 from identical policy-independent deterministic key `(scene,start,t,candidate_pose,k,checkpoint_hash)`（若此举会改变原 H0，必须保留 H0 既有键并在 H1–H3复用，记录一致性；先前历史结果不得因重设 seed 而被破坏）。
- `Hit/No-Hit`：预测无命中不生成假 surface；有效 Depth 必须位于 UNKNOWN 合法区，且不穿透已知表面；对越界、nan、负深度有测试和记录。
- `Coordinates`：射线外参、世界坐标、相机空间、V3 TSDF 聚合坐标、voxel index 需统一；不同场景的尺度 `k_M` 不能错用。
- `Observed`：观测过的表面来自真实已执行动作的深度历史；不能将 GT mesh、全局“可见 GT 点集”混入 `novelty` 或 `valid`。
- `Geometry dedup`：相同三维体素/邻域内的多个 pixel 命中只记一次；H3 两步相同物理区域不能重复计分。
- `Legal graph`：H3 仅沿当前协议允许的姿态边延伸，不允许跨墙、跳跃或假设未验证通路；未来图结构若含 GT 可见性信息不得读取。
- `H2 zero-risk check`：`lambda=0` 与 H1 完全一致（除浮点排序误差并以固定 tie break 检查）；`lambda>0` 分数变化可解释。
- `H3 depth=1 check`：depth=1 + 与 H2 相同风险定义时，选点应复现 H2；无 c2 时回退 H2。
- `Official coverage`：只在执行真实动作之后使用统一 evaluator；逐步 Coverage 不能非法下降（若官方实现非单调则保留并单独解释，不能擅自修改为 monotonic）；官方 S10 与本地 secondary monotonic 输出隔离。
- `Trace completeness`：记录所有 scene/start/t，100-step 的每步都有真实动作、coverage、runtime、policy hash、错误状态；缺失按日志重跑，不用插值补齐。
- `No hidden tuning`：不能为了 H1/H2/H3 候选精度，在 Test-3 或 All-14 之后改变评分阈值、λ、top-B；任何后改需另开协议版本。

---

## 7. 运算、排队和恢复

- 使用 Boise State HPC SLURM；先查实际 4×L40 资源、现有 Table1/F1 作业，不要抢占用户正在运行的任务。
- 每个 H 策略支持按 `split/scene/start` 提交和 `--resume`。先在少量真实合法起点做快速稳定性 smoke，随后完成 Dev-5 全部、Train-6 全部、Test-3 全部。**Smoke 不是模型性能筛选门槛。**
- 能复用的 H0 历史 70 条轨迹严格复用；H1/H2/H3 不能随意复用旧 policy 的轨迹当作新 policy 闭环结果。
- 同一个 policy 的 episode 可跨 GPU/节点并行；K4 的新评分不再重复跑 VAE/Flow，如果深度样本可缓存应缓存。H3 可依据合法姿态二层缓存，避免计算量指数膨胀。
- 对物理设计导致的时间增长，必须实测 1/5/20/100 step runtime，报告每步延迟中位数、P90、总规划时长、峰值 VRAM、额外推理次数，与 H0 匹配硬件独占测量；不要假装 H3 与 Greedy 同算力。
- 对 OOM、SLURM 重启、异常 episode 失败进行 checkpoint / restart；独立中断、逐条输出可验证的 `MISSING_EPISODES.md`。不得将“跑得太慢”转写成“算法不适用”的无数据结论。

---

## 8. 实验成功与失败都必须提供的最终结果

### 主结果表（至少四行 H0–H3，Train/Dev/Test/All-14 都有）

| model_id | split | scenes | episodes | AUC@20 | Cov@20 | AUC@100 | Final | latency s/step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V3-T1 / H0 | train/dev/test/all14 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| V3-H1 | train/dev/test/all14 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| V3-H2 | train/dev/test/all14 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| V3-H3 | train/dev/test/all14 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |

以上只是**模板**，不能用文字“实测”作为最终数值，不可上传空结果表宣称已交付。

### 指标精度

- 每个时间点：`Cov@10/20/40/50/60/80/100` 和 `AUC@10/20/40/50/60/80/100`（AUC 公式、step0 约定、积分/归一化必须使用 Table1 官方 runner 的同一实现）。
- `Final = Cov@100`（完整同协议 100 动作，除非 evaluator 自身 step convention 另有定义，需写清）；展示每条 0…100 的真实覆盖曲线。
- Dev-5、Test-3、Train-6、All-14 各自完整 summary；每场景五起点均值 + cluster-bootstrap CI；每对与 H0 的 paired 差及胜负场景。
- 0–20 步放大图和 0–100 步覆盖曲线为必交，另有 late gain 20–100，Final、Latency、动作翻转率和重复访问次数。

### 结论不能只写“赢了/输了”

报告应分别回答：

1. H1 的几何去重/真实已观测点过滤是否比 H0 排序更对齐 true oracle gain？是否影响早期 AUC/Final？
2. H2 是否成功利用生成样本之间的不一致性？是在减轻幻想表面、还是过度惩罚真实不确定区域？
3. H3 是否真正使用了两步的空间互补，而不是多重复覆盖？对 Final、Early AUC 和成本的影响各是什么？
4. Dev 上的收益能否延续到 Test-3？Train 表现是否显著不一样？
5. 与 B0/B2/MAGICIAN Beam 的现有公开 Table 1 结果相比，在**同一 evaluator** 下真实 AUC@20/AUC@100/Final 的改变量是什么？
6. 是否存在“Oracle G2 变好，但 G3 AUC 变差”的例子？必须如实展示。
7. 哪些问题来自模型 Future Depth 错误，哪些是 Score/Planner 导致？不能把两者归因混淆。

---

## 9. 推荐执行顺序（包含代码改动与完整结果）

**P0 — Audit & Freeze**
- 真实目录与当前 CVPR Table 1 的 registries/evaluator/原始曲线 audit。
- 核对 V3-T1 checkpoint、SHA、D100/S10、K4、R10、Euler10、评分和随机种子；写 `FROZEN_PROTOCOL.md`、`FROZEN_CONFIG.yaml`。冻结后不根据 Dev/Test 成果修改。

**P1 — H0 + Scoring H1/H2**
- H0 replay/alias regression。
- 实现 H1 和 H2，无 GT 的单测；G2 fixed-state 只作为诊断；立即进入各自完整 Dev G3，而不以 G2 结果拦截。

**P2 — H3 physically consistent two-step union**
- 在相同 V3-T1 权重/样本/域/动作图下实现 H3；depth1 回归 H2、depth2 几何 union 去重；完成完整 Dev G3。

**P3 — Full CVPR Train + Test + All-14**
- 固定全部 H1/H2/H3 的主配置后，完整运行 Train-6、Test-3，与 Dev-5 合并为 All-14。
- 在既有 CVPR-Results pipeline 注册新增 H1/H2/H3，自动生成 Table 1 / 按场景表 / 按 Split 表 / LaTeX 和所有图。

**P4 — Independent verification & final report**
- 使用保存的轨迹独立复算官方覆盖，列所有失败/retry/缺失，生成 `FINAL_REPORT.md`。必须含主表、逐场景、AUC horizons、Final、成败分析、效率、训练/未训练、统计不确定性以及真实 Table 1 集成证据。

---

## 10. 给代码 Agent 的验收清单

- [ ] 真实创建 `experiments/exp6_v3/H/`；没有污染原 V3/T0/T1/V3.1 和 EXP6-V5/F1。
- [ ] V3-T1 checkpoint SHA 与历史冻结参数有实际可核验记录；模型**0 次重新训练**。
- [ ] H0 复现原 CVPR V3-T1，K4 平均分数与原流水线一致。
- [ ] H1 只改物理几何评分；没有偷用 GT Mesh/coverage labels。
- [ ] H2 不是原 K4 均值的重复实现；有 λ0 回归、风险分数/样本差异实测日志。
- [ ] H3 只走真实合法动作图；两步 3D novelty union 去重；不向真实 P1 写幻想观测；每步重规划。
- [ ] **H0/H1/H2/H3 Dev-5 全部 25 episodes/策略；Train-6 全部 30；Test-3 全部 15；All-14 每策略 70/70**，未完成如实标记 partial。
- [ ] 同一个 CVPR Table 1 官方 S10 Coverage 评价器、相同 start IDs、R10、K4、Euler10、P1 D100/S10、100-step 预算。
- [ ] AUC@10/20/40/50/60/80/100、Cov@10/20/40/50/60/80/100、Final、完整 0–100 曲线、paired 95% CI、每场景与每起点。
- [ ] 所有 H 结果注册并**通过官方 CVPR Result 聚合脚本自动更新**，非手工修改主表。
- [ ] `sofia_church` 缺失时准确写 All-14，不虚构 All-15。
- [ ] 提供 runtime/memory 对照、模型 checkpoint / policy SHA、复现命令、作业编号及不遗漏的失败记录。
- [ ] 即使某策略效果弱，也交付完整 G3 和 Final Coverage，不以 G1/G2 “未达门槛”中止。

**最终目标：弄清楚 V3-T1 的生成式未来视角预测是否能通过更物理合理的几何评分、跨样本不确定度处理、以及低成本两步规划，在 Early AUC、Overall AUC 和 Final Coverage 上带来可以复现的收益。研究过程与 CVPR Table 1 严格对齐，无论结果好坏，都产出可用于论文的完整、可信比较。**
