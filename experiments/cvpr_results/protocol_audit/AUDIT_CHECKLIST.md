# AUDIT_CHECKLIST — CVPR Results 协议审计清单（2026-11-04）

**状态**：PASS / PARTIAL / FAIL / UNKNOWN / UNRESOLVED。

**证据所在文件**：
- E = `MAGICIAN_EVAL_AUDIT.md`
- F = `AUC_COVERAGE_FORMULA_AUDIT.md`
- T = `MAGICIAN_TRAINING_AUDIT.md`
- D = `TRAINING_DATA_COMPARISON.md`

## A1 评估协议
| # | 项 | 状态 | 一句话证据 | 出处 |
|---|---|---|---|---|
| 1 | 15 个场景与论文表 6 / 7 对应 | PASS | 名称一一对应，资产 sha256 与冻结清单一致 | E §1.1, §1.5 |
| 2 | Train-6 / Dev-5 / Test-3 划分有据可查 | PASS | 冻结 manifest；与 MACARONS 的划分不同，已披露 | E §1.2 |
| 3 | sofia_church 缺失原因 | PASS（原因确认） | 冻结 fork 多出的 `torch.tensor()`，官方 2026-08-29 才修复；官方测试配置也只有 14 个场景 | E §1.3 |
| 4 | All-14 / All-15 严格区分 | PASS | 只输出 All-14；All-15 一律标为 partial | E §1.4 |
| 5 | 数据版本与论文一致 | UNKNOWN | 论文没有说明数据版本 | E §1.5 |
| 6 | 起点 = settings `start_positions[k]`（官方代码与我们相同） | PASS | OFF `magician_planning.py:834-835`；SIM `m_t6_sim.py:69-71` | E §2.1 |
| 7 | 起点与上游 MAGICIAN 实际运行一致 | PASS | LMDB 70 / 70 一致，最大 \|ΔX\| = 4.5e-5 | E §2.2，`INITIAL_POSES_COMPARISON.csv` |
| 8 | 所有模型使用相同的 5 个起点 | PASS | 26 个模型、1655 条 episode，第 0 步位姿全部一致 | E §2.3 |
| 9 | 没有重复、缺失、错位或坐标错误 | PASS | 5 个起点互不相同，顺序与 settings 一致 | E §2.4 |
| 10 | 没有挑选起点 | PASS | 只接受 5 个起点都完整的场景 | E §2.5 |
| 11 | 与论文实际使用的位姿相同 | UNKNOWN | 论文没有公布位姿；官方 README 说明 settings 中的就是“随机采样的 5 个起点” | E §2.6 |
| 12 | 种子 | PASS（已披露） | 官方不设种子；我们按（场景, 起点, 步）设种子，只影响评估噪声 | E §2.7 |
| 13 | 100 步，第 0 步计入，101 个值 | PASS | OFF :464 / :516；全部 1655 条 episode 都是 101 个值 | E §3.1 |
| 14 | 动作空间与合法规则 | PASS | 三个关键函数在 OFF 与 FORK 中逐字相同 | E §3.2 |
| 15 | 无合法动作时的处理 | PARTIAL | OFF 结束轨迹，我们原地停留一步；主表模型 0 次触发 | E §3.3 |
| 16 | 真实观测预算一致，beam 与 greedy 相同 | PASS | 每步只拍 1 帧；beam 10 × 10 与论文、官方配置一致 | E §3.4, §3.5 |
| 17 | 覆盖计算与官方发布代码一致 | PASS | GT 采样、填充、覆盖函数逐字相同 | E §4.1 |
| 18 | 覆盖计算与论文所述 GT 一致 | UNRESOLVED | 论文用渲染的可见点作 GT，发布的代码用 mesh 采样，论文的 GT 没有公开 | E §4.2 |
| 19 | 官方口径与单调口径不混用 | PASS | 主表只用官方口径 | E §4.4 |
| 20 | S10 / D100 不影响评估 | PASS | 评估采样在模拟器内完成，与规划器无关 | E §4.5 |
| 21 | 所有方法使用同一评估器 | PASS | 同一冻结模拟器；每条 episode 重放逐位一致 | E §4.6 |
| 22 | AUC 梯形积分，含两端点，除以 k | PASS | 5 个合成曲线测试都通过 | F §2 |
| 23 | Final = C[100]（不取最大值） | PASS（已披露） | 两者平均只差 0.002–0.005 | F §3 |
| 24 | 先场景内平均，再场景等权 | PASS | 与 70 条直接平均完全相同 | F §5 |
| 25 | 与论文的 AUC 写法一致 | UNKNOWN（差异有界） | 论文未说明；各种写法相差 ≤ 0.009 | E §5.5，F §4 |
| 26 | 论文公布值与复现值对齐 | UNRESOLVED | All-14：AUC −0.010（n.s.），Final −0.015（✱），集中在 4 个场景 | E §6，`SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv` |
| 27 | 论文表格内部一致 | FAIL（论文侧） | FisherRF 的 Final 平均写作 0.786，逐行均值是 0.772 | E §6 |

## A2 训练协议
| # | 项 | 状态 | 一句话证据 | 出处 |
|---|---|---|---|---|
| 28 | MAGICIAN 是否有自己的权重 | PASS（没有） | 只加载 MACARONS 的 `trained_macarons.pth`（sha 34c5247f，epoch 42）；测试全程 `no_grad` | T §A1 |
| 29 | MAGICIAN 占据模型先在 ShapeNet 预训练，再在场景上微调 | PASS（论文） / 推断（权重） | 论文第 4 页；配置 `pretrained: true`；ShapeNetCore v1（推断） | T §A1、§A2 |
| 30 | MACARONS 权重的训练场景与评估场景的重叠 | PASS（已披露） | 8 / 14 个评估场景是其训练场景（dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum） | T §A2、§A6 |
| 31 | MACARONS 训练起点与评估起点的重叠 | UNKNOWN（风险低） | 训练起点在全网格随机抽，不读 settings 起点；恰好重合的期望约 0.06 次（8 个场景合计）；训练轨迹中途是否经过评估起点无法核实 | T §A2、§A6 |
| 32 | MACARONS 是否用 val 场景选 checkpoint | UNKNOWN | 发布的训练器丢弃了 val loader，按训练损失保存；作者实际做法未知 | T §A2 |
| 33 | SCONE 的训练 | PASS（原方法） / UNKNOWN（论文所用） | 原方法：ShapeNet v1 的 8 类，不在场景上训练；MAGICIAN 论文中这一列实际用的权重未知 | T §A3 |
| 34 | FisherRF 的训练 | UNKNOWN | 原方法没有学习先验；MAGICIAN 改编版的代码与权重未公开 | T §A4 |
| 35 | Random Walk | PASS | 不训练 | T §A5 |
| 36 | 我们各模型的训练数据、步数、密度 | PASS | 室外数据只来自 Train-6 去掉 S-c（138 条轨迹）；混合模型另加 MP3D 228 条；V3-T1 与 Y50K 的运行记录已核对 | D §B1 |
| 37 | 我们的评估场景是否进入训练 | PASS（Dev / Test 没有） / 已披露（Train-6） | Dev-5、Test-3、sofia 没有任何训练轨迹；每个 Train-6 场景有 3 个官方起点是训练轨迹的起点，起点 4 从未使用 | D §B2，E §2.8 |
| 38 | 评估场景是否用于选择 | 已披露 | Y50K 与 D1 按 Dev-5 选出，F1-GQ 按 Dev-3 选出；Test-3 不是盲测 | D §B4 |
| 39 | 训练轨迹的生成策略 | 已披露 | train 中有 9 条由 MAGICIAN（MACARONS 权重）生成的轨迹，另有用 GT 选动作的 CG-ε 轨迹；标签都来自 GT 网格 | D §B3 |
| 40 | 已有审计文档的准确性 | FAIL（4 处更正） | S-c 是 42 条而非 44；“官方用完美深度训练”没有依据；“alhambra / neuschwanstein 用于选 checkpoint”没有依据；MP3D 是 59 个房屋而非 61 | T §D |
| 41 | MP3D 评估重叠（只影响 MP3D 比较，不影响 Table 1） | 已披露 | 我们的 MP3D 训练房屋包含 MAGICIAN 的 2 个 MP3D 评估场景 | D §B5 |
