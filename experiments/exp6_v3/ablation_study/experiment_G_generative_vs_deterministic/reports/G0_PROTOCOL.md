# Experiment G 协议 G-v1：生成式 vs 确定性（冻结）

冻结时间：2026-11-14 21:54（在任何 D0 训练步之前；此前只跑过“flag 关闭”的 V3-T0 复现检查，它不训练 D0）。冻结后本文件不再改动；任何偏离写进 `logs/EVENTS.md`，并在最终报告里列出。sha256 记录在 `reports/G0_PROTOCOL.sha256` 和 `README.md`。

## 0. 问题与原则
论文说 Seen2Map（V3-T0@40K + D1）好于 MAGICIAN 是因为它是生成式的，而 MAGICIAN 只用确定性的占据预测。本实验在“其余全部相同”的条件下直接检验这一点。用户要求：变量尽量少，**D0 与生成式模型之间唯一的差别是预测头 / 目标函数**。不做 BCE 变体、不做概率加权解码、不调 D0 的超参数、不改 D1、不删射线编码器，室内不在本轮。

## 1. 四行
| 行 | 方法 | 生成式 | K | 权重 | 规划器 | 来源 |
|---|---|---|---|---|---|---|
| **D0** | 确定性 U-Net（一次前向回归） | – | 1 | 新训练（§2） | D1，经 Table 3 的 K = 1 路径 | 本实验新跑 |
| G0（诊断） | V3-T0 权重，从零噪声确定性解码 | 采样确定化 | 1 | V3-T0（sha `7ed0ddd7…`） | 同上 | 本实验新跑，只作诊断 |
| **G1** | 生成式 flow，单样本（噪声图 0） | 是 | 1 | V3-T0 | 同上 | 存档 Table 3 `V3-T0-K1D2`（不重跑） |
| **G4** | Ours（Seen2Map） | 是 | 4 | V3-T0 | D1（K = 4） | 存档 Table 1 `V3-T0-D1`（不重跑） |

存档数（All-14，场景等权）：G4 AUC@20 0.420 / AUC@50 0.633 / AUC@100 0.763 / Final 0.911；G1 0.420 / 0.631 / 0.756 / 0.900；G1 的噪声重复 N1–N3：AUC@100 0.765 / 0.757 / 0.760。

## 2. D0 的定义（训练）
1. **网络**：同一个 `seen2scene.v3.model.V3Model`，同一份 `experiments/exp6_v3/configs/model_v3.json`（`xattn.enabled = false`，即 T0），参数量 23,539,622（与 T0 相同，没有新增或删除的参数）。同一 seed（`train_v3.json` seed 0，在构造前 `torch.manual_seed(0)`），所以初始权重与 V3-T0 逐位相同（`common_init_sha256` 必须等于 T0 的 `6a9d6071…`）。
2. **预测头**：U-Net 的输入仍是 `[x (2) ‖ cond (76)] = 78` 通道；D0 中 `x` 两个通道恒为 0，`τ` 恒为 0（时间嵌入是常数向量，FiLM 退化为可学习的常数仿射）。U-Net 的 2 通道输出直接读作 `x̂1 = (b̂, ŷ)`。
3. **目标与损失**：目标 `x1` 与 V3 完全相同（`b ∈ {−1, +1}`，未标注的 `b = 0` 置 −1；`y = 2s − 1`，未命中 `y = 1`）。
   `L_det = Σ_{lmask} ‖x̂1 − x1‖² / (2 · N_lmask,global)`：与 `fm_terms` 相同的掩码（目标乘 `m_unk`，误差乘 `lmask = UNKNOWN ∧ b ≠ 0`）和相同的全局 2N 归一化（训练器原样做 all-reduce）。
   总损失 `L = L_det + λ_rec · L_recon + β(step) · KL`，`λ_rec = 1`、β 预热 0 → 0.02 / 5,000 步，全部不变。
4. **其余全部不变**（训练器原样运行，只在模块层面替换 4 个函数，见 §6）：数据缓存（E2 D100 视角分片 + `shared_cache/tsdf4`）、划分（室内 train / valsel，室外 Train-6 去掉 S-c / S-c）、world 4、每 rank 8 + 8 视角来自 1 + 1 个状态、accum 1、AdamW 1e-4（β 0.9 / 0.99，wd 0.01）、warmup 1,000 + 余弦、40,000 步、grad clip 1.0、bf16、EMA 0.999、每 5,000 步评估（含第 0 步）、每 1,000 步 `ckpt_last.pt`、早停规则与选模规则。
   训练器每步仍然抽 `eps ~ N(0, 1)` 和 `τ ~ U(0, 1)`（D0 不用），这样 torch 随机数流（因而 VAE 重参数化噪声）与 V3-T0 相同；numpy 批次流（seed `[0, rank]`）也相同。所以第 1 步 D0 与 T0 看到同一批数据、同一份 VAE 噪声、同一组初始权重。
5. **评估**（每 5,000 步，与 T0 的 `eval.jsonl` 字段完全相同）：`fm_*` = D0 目标在同一固定子集上的值（与 T0 的速度误差不同量纲，不直接比较）；`samp_*` = 同一个 `eval_dev.eval_samples`，采样器换成一次前向（K = 4 份相同的副本）：Brier、false wall、pred / label hit rate、hit ratio、fair ES（对确定性预测，配对项为 0，即 RMS 误差）、spread（= 0）、ambiguous bit rate、s_std_hits；`recon_*` 不变。
6. **选模**：冻结的 V3 规则原样：在步数 ≥ 5,000 的评估中取 `ES_in_val / ES_in_val(5k) + ES_out_sc / ES_out_sc(5k)` 最小者，平局取更晚的步；早停规则（预算 2/3 之后连续 2 次评估无改进）照常生效。选中的 EMA 拷到 `work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt`。不看任何闭环结果挑模型。
7. **中断**：若训练崩溃，从 `ckpt_last.pt` 续跑（训练器的既有规则：生成器按 `(seed, rank, step)` 重新播种，记录在案，作为偏离写进 EVENTS）；不换 seed 重跑。

## 3. D0 的推理
每个候选视角一次前向：`x̂1 = U-Net(x = 0, τ = 0, cond) · m_unk`；解码规则不变（`geometry.decode`：`hit_u = (b̂ > 0) ∧ (ŷ < 1) ∧ (n_u > 0)`，深度由 ŷ 在 UNKNOWN 弧长上的位置给出）。规划器 = 冻结的 D1（D-v1 `d_config.yaml` 的 `d1`：两步、穷举、λ = 0、1 m 格、eps_cov 1.0），经 Table 3 的 K = 1 路径（`kd_policy.install("d1", 1)`，b2：D100 输入 + S10 新颖性，R10，模拟器合法候选，`m_score.choose`）。K = 1 路径照常生成噪声图 0，D0 的预测器忽略它。

## 4. G0 的定义（诊断，不训练）
V3-T0 权重，`x0 = 0`（所有噪声图换成全 0），Euler 10 步，K = 1，其余与 G1 完全相同（同一预测器代码，只把 eps 换成 0）。G0 的第一步 Euler 输入恰好是 D0 的输入（`x = 0, τ = 0`）。G0 只用来区分“生成式训练得到的权重”和“随机采样”两种效应，不进入主结论。

## 5. 公平性审计（已知差异，均不做补偿）
1. **均值回归是确定性头的本性**：在共享的目标编码下，MSE 回归学到 `b̂ ≈ 2p − 1`，解码阈值 `b̂ > 0` 等价于 `p̂ > 0.5`；未命中射线的 `y = 1` 会把 `ŷ` 拉向远端（`ŷ ≈ p·E[y | 命中] + (1 − p)`）。这正是“确定性占据预测”与“生成假设”的差别本身，不是实现缺陷；固定状态的首次命中深度误差会量化它。
2. **损失量级**：flow 的速度误差包含噪声方差，量级大于 `L_det`；在 λ_rec、β 不变时，VAE 项相对预测项的权重在两组之间不同，梯度裁剪（1.0）触发的频率也可能不同。这是“只换目标”的直接后果，按任务书不调整。
3. **τ 与 x 通道**：`τ = 0` 让时间嵌入成常数；输入卷积中对应 `x` 的权重收不到梯度（只有 weight decay），参数量不变。选 `x = 0, τ = 0` 而不是 `τ = 1`：`τ = 1` 时 flow 的输入就是 `x1`，没有意义；`τ = 0` 时 flow 的输入是纯噪声，D0 = “把噪声换成 0、一步跳到 x1 并为此训练”的 U-Net，与 G0 的第一步输入相同。
4. **选模指标**：D0 的 fair ES 退化为 RMS 误差，选模公式相同但量的含义不同（没有多样性项）。
5. **单 seed**：两组都只有一个训练 seed；训练从第 2 步起不逐位可复现（grid_sample 反向的原子加），无法估计训练 seed 方差。G1 的 4 个噪声重复给出规划器噪声的量级（AUC@100 0.756–0.765），在 Table 3 脚注里列出。
6. **推理开销**：D0 每候选 1 次 U-Net 前向，G1 10 次，G4 40 次；覆盖指标按步数计，不按时间，开销单独报告。
7. **硬件**：D0 在 gpu114（4 × L40S）训练，V3-T0 在 gpu117（4 × L40S），软件环境相同（seen2scene env，torch 2.4.1）。

## 6. 实现与校验（全部通过才开始正式训练）
实现只在 `experiment_G_generative_vs_deterministic/scripts/`：`d0_head.py`（D0 / G0 的全部函数）、`train_g.py`（`--head det|flow`；`flow` 时**不做任何替换**）、规划器子进程与闭环入口、固定状态与汇总脚本。`seen2scene/` 下和其他实验的文件一律不改。
- (a) 生成式路径不变：① 共享代码 sha256 前后一致；② `train_g.py --head flow` 在 gpu114 上 4 卡跑 50 步：第 0 步评估的全部数值、第 1 步日志（loss / fm / rec / kl / grad_norm / 射线数）与存档 V3-T0 逐位相同，`common_init_sha256` 与 `data_manifest_sha256` 相同；③ 推理：固定状态工具的 G4 行在若干状态上与存档 V3-T0 的 D1 分数逐位相同；闭环入口在 flow 模式（K = 1）下前若干步的分数与选择与存档 `V3-T0-K1D2` 的规划器日志逐位相同。
- (b) D0 单元测试：形状；掩码（lmask 外的误差不进入损失、输出在 m_unk 外为 0）；损失等于手算值；梯度到达 VAE 编码器、射线编码器、U-Net（且 x 通道的输入卷积权重梯度为 0）；同一输入两次前向逐位相同、与 batch 中的位置无关；解码与 `geometry.decode` 一致；`det_sample` 与 `DetPredictor` 一致；G0 预测器等于 eps = 0 的 `V3Predictor`。
- (c) D0 的评估字段与 T0 的 `eval.jsonl` 一致，由同一个 `eval_dev` 函数计算。
- (d) D0 训练 smoke（4 卡，300 步）：损失有限且下降；第 1 步的 rec / kl 与 T0 第 1 步逐位相同（同初始化、同批次、同 VAE 噪声）；EMA 可被 K = 1 规划器加载；一个场景几步的闭环 smoke 通过（含官方覆盖回放一致）。

## 7. 固定状态评估（训练完成后）
状态池 = Experiment F 的 755 个室外状态（explore 395：E3 D1 轨迹 colosseum s0–s4 + pisa s1，G2 Dev-3 B0 / B2；confirm 360：D-v2 Table 1 的 6 个场景、t = 0, 5, …, 95；不含 Test-3、不含 dunnottar），回放链 = Experiment CT / U 的 `ct_fixed` → `f_run` → `mv_core`（D-v2 关闭 → D1）。同一进程里四行看到同一个回放状态、同一组候选、同一组噪声键（G4：噪声图 0–3；G1：噪声图 0；G0：0；D0：无）。
- 预测质量（第一层候选，带标签的 UNKNOWN 射线；标签 = 训练规则 `geometry.encode_labels` 作用于该状态自己的地图上的 GT 首次命中）：命中准确率 (TP + TN) / N、命中精确率 / 召回率、Brier（p = K 个样本中命中的比例，K = 1 时为 0 / 1）、false wall（带标签 NOHIT 射线上预测命中的比例）、首次命中深度误差（真阳性射线上 |ẑ − z*| 的均值，运行时 z 深度；以及误差 ≤ 1.0 的比例）。
- 排序（该行自己的 D1 两步分数，选择 = `m_score.choose`）：被选候选的官方一步增益（pp）、Top-1 率（与最大增益相差 ≤ 1e-4 pp）、状态内 Spearman（分数 vs 官方一步增益，合法候选）；regret 只放附录。
- 统计：状态均值；配对差 D0 − G1、D0 − G4、G0 − G1，按轨迹聚类的 bootstrap（10,000 次，seed 0，95 % 百分位区间）；explore / confirm 分开报告，再报合并。
- 回归：G4 的 D1 分数必须与存档 V3-T0（EXP9 Table 4 V2 的 T0 行）逐位相同，否则停。

## 8. 闭环评估与 Table 3
D0 与 G0：Macarons++ All-14 × 5 个官方起点 × 100 步，冻结模拟器（M 协议），每个 episode 跑完后用 `t25_out_cov.py` 原样重放官方覆盖，必须与存储的曲线逐位相同，否则该 episode 无效并重跑（≤ 3 次）。任务全部经共享独占 GPU 队列（`expG_*`）。G1 / G4 读存档（`cvpr_lib.load_manifest`），不重跑。指标定义直接调用 `cvpr_lib`：C[0..100] 官方覆盖；AUC@T = trapz(C[0..T]) / T；Final = C[100]；场景值 = 5 个起点均值；组值 = 场景等权均值。

**Table 3（预注册的版式）**
| Method | Generative | K | AUC@20 | AUC@50 | AUC@100 | Final |
|---|---|---|---|---|---|---|
| Deterministic U-Net (D0) | – | 1 | | | | |
| Generative flow (G1) | ✓ | 1 | | | | |
| Ours (G4) | ✓ | 4 | | | | |
| *诊断：flow 权重 + 零噪声解码 (G0)* | *(deterministic sampling)* | *1* | | | | |

每个数后附场景配对 bootstrap 差值与 95 % CI（10,000 次，seed 0，重采样单位 = 场景）：相对 G4，以及相对 G1；胜 / 负场景数。另附 Train-6 / Dev-5 / Test-3 分组数、G1 噪声重复（N1–N3）的范围、每步决策时间。输出 `results/table3/table3_G.{csv,md,tex}`。

## 9. 结论措辞（预注册）
主指标 AUC@100（All-14）；AUC@20 与 Final 为次要指标，单独在次要指标上显著不升级为主结论（写成“仅在 X 上”）。“显著”= 配对 95 % CI 不含 0。
- **C1：G1 − D0**（同 K = 1，只差预测头 / 目标）。
  - G1 > D0 显著 → “在相同网络、数据、规划器和 K = 1 下，生成式训练本身优于确定性回归（generative training helps）。”
- **C2：G4 − D0**。
  - C1 不显著而 G4 > D0 显著 → “单个生成样本与确定性预测相当，收益主要来自多假设预测 + 覆盖聚合（K = 4）。”
  - G4 ≈ D0（CI 含 0） → “不能把优于 MAGICIAN 的结果主要归因于生成建模；需要重新审视 CDGR 的贡献。”
- D0 显著优于 G1 或 G4 → 如实写“确定性头在 X 上更好”。
- G0 只作诊断：G0 vs G1 说明随机采样本身的作用，G0 vs D0 说明“同一 U-Net，生成式训练的权重 + 确定性解码”与“确定性训练”的差别；不用于主结论。
- 固定状态结果只解释机制，不替代闭环结论。

## 10. 冻结的输入
| 项 | sha256 |
|---|---|
| V3-T0 `final.pt` | `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6` |
| `configs/train_v3.json` | `a076e0880f20730c0b470a7fee4221badc20c2f4bd5406ef383c63fc63eff2e8`（= T0 运行清单记录值） |
| `configs/model_v3.json` | `d42e5d2671005520fe0566023110061528be554ef541dcbc6ba0487d3ba9f43c` |
| `seen2scene/v3/trainer.py` / `losses.py` / `data.py` / `model.py` / `eval_dev.py` / `predictor.py` | `43d498e8…` / `dd190a91…` / `0f5e1480…` / `6d8352cf…` / `cfe7282b…` / `4ac317ab…`（自 T0 的提交 `7de02cc` 起未改） |
| D-v1 `d_config.yaml` / `d_policy.py` | `e53c230d…` / `c8418279…` |
| Table 3 `kd_policy.py` / `kd_out_planner.py` / `sim_out_kd.py` | `3c61da69…` / `3bb1cc05…` / `146ed02a…` |
