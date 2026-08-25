# Hallucination-Aware Active Mapping — 项目状态总结

> 最后更新：2026-08-25 ｜ 作者：Jiahui Zhang（Boise State University）
> 基线：MAGICIAN（CVPR 2026, `shiyao-li/MAGICIAN`），author weights，perfect depth，beam 10×10，101 poses/traj。
>
> **本文是仓库里唯一的跨阶段权威总结。** 逐次实验的原始日志、LMDB、论文素材（约 7.8 GB）都在
> 本地 `jiahui/` 工作区里，不上传 GitHub；能长期复用的结论、代码改动清单和复现方式，全部浓缩在这里。

---

## 0. 一句话

`occ` 网络输出的是**信念**，MAGICIAN 却把它当**事实**用（硬阈值 `occ > 0.5`，同一批点既是覆盖收益的来源、又是碰撞墙）。
我们做了一套反事实审计，给每类地图错误定价，并给出一个免训练、免 GT 的规划器侧修法 **SCCov**：
**探索时不预判，看清后不纠缠。**

当前进度：**Stage 1（审计）完成 → Stage 2（信念阶梯 + SCCov）完成 → Workshop 论文写完 → Stage 3 Phase 0 诊断完成并出了否定结论 → 下一步 Phase 1（可学习置信度通道）。**

---

## 1. 仓库里有什么 / 没什么

| 路径 | 内容 | 是否上传 |
|---|---|---|
| `macarons/` | pipeline 本体。**我们所有的模型/规划器改动都在这里**，全部由环境变量门控，默认关闭时与上游逐位一致 | ✅ |
| `configs/` | 训练 / 测试 / 消融配置 | ✅ |
| `jiahui/*.md`, `jiahui/*.py` | 长期维护的文档与分析工具（环境、架构、训练、实验总览） | ✅ |
| `jiahui/stage_1/ stage_2/ stage_3/ paper/ log/` | 实验日志、LMDB、dump、论文源码与图素材，约 7.8 GB | ❌ 本地 |
| `data/ weights/ results/ work_dir/` | 数据集、权重、运行产物 | ❌ 本地 |
| `RaDe-GS/` | submodule（`diff_gaussian_rasterization` 的来源），见 `.gitmodules` | 以 submodule 指针形式上传 |

Remote 约定：`origin` = 我们自己的 `Jiahui0223/magician`；`upstream` = 原作者 `shiyao-li/MAGICIAN`（只读，用于同步上游更新）。

---

## 2. 代码改动清单

改动只落在两个文件，都是**加法**，默认关闭时零行为差异。

| 文件 | 改了什么 |
|---|---|
| `macarons/utility/macarons_utils.py` | 三路证据计数器 `proxy_n_surface / proxy_n_free / proxy_n_occluded`（声明、初始化、逐帧更新、memory 存取）；模块级 `BETA_ASSERT` 开关 |
| `macarons/testers/magician_planning.py` | SCCov 主方法；oracle 反事实（FP/FN 分解）；verification-gain 对照；`noimag` 档；深度噪声注入；Beta 后验与两个连续杠杆；gain probe 与 PLY/NPZ 诊断落盘 |

**关键不变式**：`n_surface + n_occluded == n_behind_depth`。`MAGICIAN_BETA_ASSERT=1` 时逐帧断言，6 次全程运行未触发，
即加计数器不改变任何既有行为。

### 环境变量总表

```
# ---- 主方法 SCCov ----
MAGICIAN_SC_MODE          设了就开启（总开关）
MAGICIAN_SC_K=2           成熟度门：进过视野 ≥K 次才有资格被审
MAGICIAN_SC_LAMBDA=1.0    gain 杠杆：suspect 的 opacity ×(1−λ)
MAGICIAN_SC_COLLIDE=1     collision 杠杆：0 = 只变透明不拆墙
MAGICIAN_SC_SIGNAL=unsupported   'ratio' 是已证明无效的负对照
MAGICIAN_SC_TAU=0.6       仅 ratio 信号使用
MAGICIAN_SC_TRACE=<csv>   逐步 trace

# ---- 阶梯上的其它档位与对照 ----
MAGICIAN_NOIMAG=1                     完全关掉想象
MAGICIAN_ORACLE_MODE / _DROP          GT oracle：删幻觉 / 补漏检
MAGICIAN_VERIFY_MODE / _LAMBDA / ...  verification-gain 对照
MAGICIAN_DEPTH_NOISE=0                深度噪声套件

# ---- Stage 3 Beta 后验（默认关闭）----
MAGICIAN_BETA=1           总开关
MAGICIAN_BETA_C=5.0       先验伪计数（等效样本量）；→∞ 精确退化成 base
MAGICIAN_BETA_WS / _WF / _WOCC=1.0/1.0/0.25   三路证据权重（w_occ=0 是负对照）
MAGICIAN_BETA_KAPPA=0.0   可信下界 μ − κσ
MAGICIAN_BETA_TAUCOL=0.5  碰撞杠杆的 LCB 阈值
MAGICIAN_BETA_GAIN / _COLLIDE=1       分别关掉两个杠杆
MAGICIAN_BETA_TRACE / _DUMP / _DUMP_STEPS / _ASSERT

# ---- 运行控制与诊断 ----
MAGICIAN_TEST_SCENES / _ONLY_START / _MAX_STEPS / _LMDB_DIR_NAME
MAGICIAN_DUMP_DIR / _DUMP_STEPS       占据场落盘
MAGICIAN_GAIN_PROBE / _GAIN_PROBE_EPS 逐像素 gain 分解（论文 Fig. 2）
MAGICIAN_PROBE_IMG_DIR / _PROBE_IMG_STEPS   probe 图与 PLY 导出
```

---

## 3. Stage 1：审计链（exp1–exp7）

每一步只回答一个问题，答案是否定的就记录否定。

| 实验 | 问题 | 答案 |
|---|---|---|
| **exp1** 复现 + 普查 | 基线复现得了吗？失败在哪？ | 忠实复现（final 0.906 / AUC 0.711 vs 论文 0.919 / 0.721）。失败集中在少数起点：pantheon2 = 0.353、neusch4 = 0.597、neusch2 = 0.672 |
| **exp2** 失败签名 | 失败长什么样？ | **向心塌缩**（mean_dist2center r = +0.43，path_spread r = +0.32），而不是「卡在 bbox 里」（r = −0.11） |
| **exp3** 幻觉存在性 | 幻觉真的存在吗？ | 存在且分两型：塌缩型（pantheon2，幻觉随失败增长 9% → 38%）；早期幻觉型（neusch4，step 0 = 90%，之后被观测纠正到 12%） |
| **exp4** 因果归因 | 幻觉真的在带偏决策吗？ | 是。被选视角的早期 gain 有 33–36% 由幻觉贡献，且早期步的 gain 量级最大 |
| **exp5** Oracle 反事实 | 把幻觉打折能救覆盖吗？ | **不能（NO-GO）**。GT 硬删幻觉，8 个起点只帮到 1 个。FP/FN 分解显示瓶颈是**漏检**不是幻觉（pantheon2：Δfp +0.009 vs Δfn +0.248） |
| **exp6** 验证引导 | 派无人机去核实幻觉，效率赚吗？ | **只在困难场景赚**。oracle 池化 Δfin −0.005（困难 +0.054 / 简单 −0.036 受伤）；可部署的 support 信号失败（−0.044，分不清「幻觉」和「还没看过」） |
| **exp7** 自纠错覆盖 | 不指挥无人机，只修正它信的地图呢？ | **成功**，成为主方法 SCCov |

**exp5 的推论是整篇论文的枢纽**：幻觉 gain 同时是规划器**唯一的向外探索驱动力**。
在观测之前删掉它，探索就塌缩。所以任何可用的修法必须发生在**观测之后**。

---

## 4. Stage 2：信念阶梯（同码 184 runs，5 场景 × 5 起点，配对 Δ）

固定规划器、beam、碰撞规则、预算、代码，**只换规划器所相信的那张地图**，从零信任到全知。

| 档位 | 用 GT？ | 池化 Δfin | 池化 ΔAUC | 困难 Δfin | 困难 ΔAUC | 简单 Δfin | 简单 ΔAUC |
|---|---|---|---|---|---|---|---|
| (i) 无想象 | 否 | −0.007 | −0.038 | **+0.100** | +0.055 | −0.050 | −0.073 |
| (ii) occ − FP（删幻觉） | oracle | −0.004 | −0.006 | +0.020 | +0.019 | −0.029 | −0.019 |
| (iii) occ（base，绝对值） | 否 | 0.846 | 0.644 | 0.676 | 0.485 | 0.908 | 0.706 |
| **(iv) occ + SCCov（ours）** | **否** | **+0.013** | **+0.018** | +0.004 | +0.025 | **+0.004** | **+0.010** |
| (v) occ + FN（补漏检） | oracle | −0.026 | +0.031 | −0.044 | +0.035 | −0.023 | +0.030 |
| (vi) ground truth | oracle | +0.031 | +0.086 | +0.095 | +0.178 | +0.004 | +0.055 |

分层：困难 = base final < 0.75（n=5），简单 = > 0.85（n=16），池化 n=25。

**三条结论**

1. **塌缩是想象的锅。** 在塌缩起点上，最便宜的档位和最贵的档位一样好：直接关掉想象把最差起点从 0.31 拉到 0.56，与相信 GT 地图（0.57）统计上无法区分。反面同样成立：在简单起点关掉想象要付 −0.05。**想象的价值不是架构常数，它随运行状态变号** —— 这直接指向「自适应信任」。
2. **地图错误不可加。** 单修 precision（ii）到处没用；单修 recall（v）效率涨了却**丢**最终覆盖（−0.026，最差 −0.39）：补回来的结构也进了碰撞集，agent 被自己修正过的地图围死。只有全修（vi）是安全的。**修一半地图可能比不修更糟。**
3. **完美地图买的是速度，不是高度。** GT 档在困难起点用 11.2 步达到 70% 里程碑，base 需要 39.6 步（3.5×），但最终覆盖只 +0.095。剩下的差距不是地图问题，是贪心 10 步前瞻、离散位姿图和可达性 —— 阶梯把每个失败拆成**地图账**和**规划账**并分别定价。

### 主方法 SCCov

每步对每个想象点（`occ > 0.5`）做两关审判，全程免 GT：

1. **成熟度门** `n_t(x) ≥ K`（K = 2）：没看过就无罪推定，保留全部 gain。这是保探索的抗塌缩条款，按构造成立，不靠调参。
2. **观测支撑** `d(x, P_t) > ε`（ε = 场景对角线的 3%）：看过 ≥K 次，自己攒的观测点云里却始终没有表面出现 → 判为 suspect。

Suspect 在想象被消费的**两条通路上同时中和**：从 gain 渲染里**变透明**（不再引诱），从碰撞集里**释放**（不再挡路）。
每步重新审判，所以真结构被误判后一旦有支撑出现就立刻无罪释放，幻觉则持续累积「成熟却无支撑」的证据。
**验证行为是涌现的，不需要专程绕路**（这正是 exp6 失败的地方）。

消融（5 个困难起点，Δfin 均值）：gain-only **+0.124** 但有尾部风险（被假墙围死）；full 主方法 +0.051 且唯一稳定为正、简单场景无损；collide-only +0.043；nogate（K=0）+0.033 且不稳，**重现 exp5 的塌缩** —— 证明「观测后才定罪」是关键设计。

---

## 5. Workshop 论文（已完成）

**标题**：*Don't Trust the Map You Imagined: Auditing and Self-Correcting Imagination-Driven Active Mapping*
**源码**：`jiahui/paper/workshop/`（`main.tex` + `sec/0..7`，CVPR 格式；HPC 上没有 pdflatex，需本地或 Overleaf 编译，另需 `cvpr.sty` + `ieee_fullname.bst`）

结构：§1 引言 → §2 相关工作 → §3 双重陷阱（审计 + 两个否定结果）→ §4 信念阶梯 → §5 SCCov → §6 实验 → §7 结论。
全文引用完整，6 图 3 表均在正文被引用，无 undefined reference。

**Fig. 2（幻觉因果性地引导早期决策）** 是本文最强的一张图，做法值得记住：
不用代理指标，直接**复用规划器自己的渲染**，只改点携带的颜色 —— 每个想象点画上自身贡献 `1 − novelty`，
从**规划器实际选中的位姿**渲染，得到的就是它最大化的那个标量，逐像素分解。
再用 GT 幻觉标签作为第二通道劈成两半，于是 `(总 gain) = (真实结构) + (幻觉)` **逐像素成立**。

这张图导出了一个决定性的量化事实：两个选中帧上，**100% 与 96.4% 的幻觉 gain 点位于真实表面的背后**
（沿各自射线，中位数分别为场景对角线的 +5.3% 和 +5.1%）。所以幻觉在 RGB 里**按构造就是看不见的**，
这不是图的缺陷而是机制的签名：**空间雕刻能证明一个区域是空的，永远无法证明它是有的**；
没有任何光线到达过的体积保持初始的「占据」状态，监督标签恒为 1，网络于是学会去填它看不见的地方。
这些点从未被覆盖过，所以在 gain 里满额计分 —— **规划器被奖励飞向几何上永远不可观测的表面。**

图 2 的幻觉 gain 占比：53.5% 与 35.0%，且这两帧分别处在各自轨迹 gain 峰值的 100% 与 86%；
全轨迹上，前 10 步的幻觉份额（gain 加权）为 19.1% / 21.5%，step 30 之后降到 6.7% / 6.4%。
**幻觉扭曲得最厉害的，正是最要命的早期决策。**

---

## 6. Stage 3 Phase 0：证据式（Beta）占据场 —— 诊断完成，预注册闸门 FAIL

**动机**：SCCov 之所以有效，是因为它近似了一个后验（成熟度门 ≈ 集中度 α+β，支撑检验 ≈ 后验均值低）。
Phase 0 想把这个后验真正写出来：`Beta(c·p, c·(1−p)) + (n_surface, n_free)`，让信念带上可相加的证据量，
用一个先验强度 c 取代两个手调常数 K 和 ε。

**已完成并验证**：三路计数器（严格保守，遮挡帧谁也不加）、Beta 后验、两个连续杠杆、
`c → ∞` 精确退化成 base（最大逐点差 5e-9）、κ=0 ⇒ LCB ≡ μ、σ 随证据收缩 0.231 → 0.049、
`analyze_p0.py --selfcheck` 6/6 精确复现 stage_2 结果表。

**结论（6 起点全程 dump，各 101 步）**：两个预注册闸门都 FAIL，原计划的 81-job 大批次**没有提交**。

- **G1（逐点信号）FAIL**：即使在同一份数据上给每个起点网格搜最优 (c, w_occ)（乐观上界，过拟合了还算它赢），
  中位 AUROC 增益只有 +0.020，3/6 起点连 0.02 都不到。两个大赢家都选了 `c = 0.5`，等于几乎完全无视网络，
  那是另一个 regime 而不是改良。
- **G2（全局信号 ≤30 步可分）FAIL**：σ 衰减与最终覆盖率的 Spearman 要到 step 50 才有预测力（0.83），远超预注册的 30。
- **「饱和假设」被证伪**：饱和度与后验增益的 Spearman 只有 +0.14，早先 n=2 时看到的模式是巧合。

**必须记下的自我修正**：G1 测的是**幻觉检测能力**，但 stage_2 早就证明**检测不是机制**
（SCCov 的 tag 对几何幻觉 AUROC ≈ 随机，却在 planner 上有效；GT 删幻觉的 oracle 池化 −0.004 毫无收益）。
既然检测不是机制，用检测能力做闸门就是**错的闸门** —— 所以 G1 FAIL **推不出**「Beta 杠杆在 planner 上无效」，
那需要 planner 级实验（22-job B-mini）才能裁决。

**真正立住的四条**：

1. 网络自己的标量在 5/6 起点上是中等强度的幻觉检测器（AUROC 0.64–0.76）；
2. 在 1/6 起点上完全失效（0.507）；
3. **我们试过的所有免-GT 统计量都预测不了它什么时候失效**；
4. 观测计数在 5/6 起点加不进信息，即便用 oracle 选参也一样。

⇒ 瓶颈不是「证据不够」，是**「不知道网络什么时候是错的」**。这正是 Phase 1 要解决的问题。

副产物：`n_surface` 的符号在两个域都稳（0.62–0.77）；`n_occluded` 符号会翻转（需要状态门控）；
`n_free` 在想象集上恒为 0（`occ_mask` 先剔除了 —— 与 carve-ratio 消融被证明惰性是同一个原因）。

---

## 7. 下一步

1. **B-mini（22 jobs）**：planner 级的决定性实验，裁决 Beta 杠杆到底有没有用。从 borah-login 提交。
2. **Phase 1：Beta head**。`SconeOcc.linear3` 由 1 维改 2 维，输出 `α, β = softplus(·) + 1`，
   Beta-Binomial 边缘似然 + Brier 锚 + KL 正则，只微调 head。
   **标定闸门（接 planner 之前必须过）**：用预测方差识别幻觉点的 AUROC —— 网络知不知道自己在瞎编。
   若接近随机，就保留纯观测后验，**而这个否定结论本身就是贡献**（「占据网络会自信地犯错，不确定性必须来自观测而非网络」）。
3. **Phase 2：自适应信任 controller**。阶梯的第 1 条结论（想象的价值随状态变号）说明任何固定选择都必然在一半场景上错，
   只有在线切换能两头都赢。有了逐点 (μ, σ) 之后，它就是在 base 档与 noimag 档之间连续插值。

**已知的方法论约束（务必遵守）**

- GPU 不确定性：单起点 σ 可达 0.12，**只有重复实验与池化数字可以引用**。
- 所有数字口径以同码配对 Δ 为准，跨代码基线的绝对值不可比。
