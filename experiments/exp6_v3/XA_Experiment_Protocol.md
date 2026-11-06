# Experiment XA — Multi-Scale Geometry-Aware Ray–Scene Attention

**状态：设计协议，未在 BSU 执行；不含任何 XA 实测结果。**  
**建议位置：** `experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/`（实验与报告），`seen2scene/v3/xa/`（可复用模型组件）。  
**核心约束：** CT 安全存档后停止；XA 按实验 ID **逐个执行**，但提交后由调度器**不中途人工审批**地跑完整队列；**每一个可正常训练的变体都执行室外全部 70 条、100 步闭环**，不因早期预测指标差而取消；Indoor 只做事先冻结的小样本 sanity check。

## 0. 正式决策

1. 研究重点：**Outdoor Macarons++ All-14**，Train-6 / Dev-5 / Test-3，14 场景 × 每场景 5 起点 = **70 episodes/model**，每条 100 步。当前审计中 All-14 即现有已实现评估场景集合；不要把尚未接入的数据集（例如 UAVScenes / NavBenchGS）擅自加入。
2. 首要指标（全部是**官方 MAGICIAN coverage**）：`AUC@20`、`Coverage@20`、`AUC@100`。附加 `AUC@50`、`Coverage@50`、`Coverage@100 (=Final)`；按场景提供逐条曲线和摘要。
3. Indoor：在**已有 MP3D val/G3 列表**中事前固定 3 个楼层、每个 2 个相同起点，跑 150 步的 `C_obs` sanity check；**不能**把室内 C_obs 数值与室外官方 coverage 混合平均。Indoor D1 尚无官方实现入口；第一轮采用已有 H0 室内实现并对所有 XA 变体和 T1 一视同仁，若能正确适配 D1，再作为补充、不能阻塞 Outdoor 全量结果。
4. 主对照 Planner：**D1**（2-step、λ=0、K=4、Euler10、D100 输入/S10 新颖性、R10 刷新等完全固定），不要为每个模型调参。D-v2 作为日后独立补充，不能与 D1 结果混用。
5. 训练：**从零训练全部 6 个核心变体，各 40,000 步，尽量复用 T1 的相同 seed、数据划分、训练样本序列、优化器、EMA、loss、scheduler 和 batch 配置**。如与 CT 使用相同资源，先 CT 完整备份后停止。每 5K 或原有里程碑保存可恢复检查点；不得用早期 dev 指标裁掉模型。
6. 工程顺序 = XA-S2D → XA-SR → XA-SI → XA-M2D → XA-MR → XA-MI。**同一启动批次依次训练+闭环+聚合**，直到六组都产生结果；允许只因数据泄漏、损坏、OOM、NaN 等工程错误而修复/重跑，不允许按实验效果早停。
7. 历史 baseline V3-T1@40K 已有室外 D1 全量 70/70 的参考。为了代码与噪声策略回归，每项评估仍读取统一的 T1 官方结果；如 XA 改变了公共仿真接口，应重跑 T1 70/70 验证。

## 1. CT：安全归档和停止，**不要直接 scancel**

CT 的运行 ID、signal handler、训练输出位置和最新 step 未在当前会话中验证，必须在 BSU 现场发现：

```bash
squeue -u "$USER" -o "%.18i %.12j %.2t %.10M %.2D %R"
scontrol show job <CT_JOB_ID> | egrep 'JobId=|JobState=|WorkDir=|Command=|StdOut=|StdErr='
```

- 确认 CT 的 checkpoint 生成方法：如果代码支持 SIGUSR1/USR2 保存，可按真实 handler 发信号；否则等待已有周期 checkpoint，或先补充安全 checkpoint hook 并对当前进程可行性核实。**禁止向没有 signal-save 处理器的训练进程发送“保存信号”后假定已保存。**
- 至少备份：模型权重、EMA、优化器状态、global step、scheduler 状态/可复原参数、配置、代码 git SHA/dirty diff、日志和必要的 RNG 状态。已有审计指出 V3 的 resume 会重新播种 RNG；不能把恢复后的样本序列称为与不间断训练逐位相同。
- 归档到独立只读目录，避免 CT 与 XA 共写 checkpoint 和日志。计算 sha256；用可信来源的 checkpoint 检查关键 keys，实测用 EMA 推理可加载；写出 `CT_PAUSED_MANIFEST.md`（step、sha、Slurm job、代码版本、续训命令、恢复限制）。
- **只有上述保存与校验完成**，才取消 CT 的 Slurm job。`scontrol hold` 只适合 pending 作业，不等于暂停运行中的 GPU 训练；不依赖 login 节点的 `tmux+salloc` 来保证自动 XA 队列存活。
- 若无法保存当前 CT，保留 CT 直到可恢复检查点就绪；**不能以“尽快 XA”为理由丢失训练状态**。

## 2. 输入、模型以及严格保留的部分

现有 T1：Partial TSDF 的 4× 表征 `[sdf, band, free, valid]` → 3D Conv VAE → `μ`（16 channels @ 8×）→ 沿每 Ray 32 点采样 μ → 64 通道 Ray Encoder；另有 11 通道投影 + 1 列 mask；2D Flow U-Net 的瓶颈 `[256,8,16]`，128 Queries 对 512 Scene Tokens 进行 4-head Cross-Attention，输出经 10 Euler 步预测 `(b,y)` 并只在 UNKNOWN 弧长内生成 First-Hit。以上均来自已有审计；不要凭空添加完整场景补全或“跨视角同一世界”的宣称。

**固定内容：** P1 构造、D100/S10 密度、GT 标签生成方法、loss（FM + CE/L1 重建 + KL）、K=4、Euler=10、Hit/NOHIT 解码、未知射线 mask、Planner D1、simulator、官方 coverage 回放。

**只改网络：** VAE Feature Pyramid、Scene KV、Query 与 Cross-Attention。这让全量闭环中的模型差异尽可能可归因。

### 2.1 Multi-Scale VAE Feature Pyramid

```text
Partial x4 + valid4
  └── shared 3D VAE encoder
       ├── F4: stride-4 坐标上的中间特征（须清楚定义是哪一层）
       ├── F8: 原始 μ（16 ch，stride-8，受 KL 约束）
       └── F16: 对 μ 做 valid-aware 2× pooling + learnable projection

F4 → 只在 Query 需要的局部 3D 坐标按需采样，不展平完整 F4 网格。
F8 → 中等分辨率 Scene Tokens，保留 band/free 与 3D PE。
F16 → 低分辨率大范围 context Tokens，保留有效观测支持与 3D PE。
```

- 不能把 F4 说成“VAE μ 的高分辨率版本”：它是编码器中间特征，不直接由 KL 正则化。
- F16 的 pooling 必须携带 valid/support 权重，不可把无观测区域的零值误当已知空闲空间；避免跨域时使用未经统一的米制阈值（Macarons++ 模型单位不等于米）。
- F4/F8/F16 所有坐标与相机 Pose 必须共用一套坐标变换；场景级 global vs MP3D local P1 差异必须明确处理。
- 为避免性能伪归因，对每个变体记录训练参数量和 Attention token pairs、推理耗时/显存；大计算量提升应与 compute-matched control 分开解释。

### 2.2 三种 Query：基于同一输出接口

**Q-S2D：Spatial 2D.** 保留原有 8×16 Bottleneck Query 作为回归锚点。公平比较时另设 16×32 Query（512 个）或等量 token 稀疏版本。特征来源 U-Net 2D features + camera/ray PE。其输出上采样/残差注入原 Flow U-Net。

**Q-R：Ray.** 从固定二维锚点选 512 条代表性有效 Ray（不足则 pad + mask），以每 Ray 的 64D 特征、对应 U-Net 特征、相机原点/方向 PE 构成 Query。**选 Ray 的规则预先冻结，只使用当前观测的 UNKNOWN 长度、边界和可见性，不用未来深度/收益。** 可优先用确定性的 16×32 网格锚点，使空间覆盖可比；每条 Query 从 F4/F8 的局部 3D 邻域检索特征。散射/插值回 2D feature grid，不能直接改变 Flow 的 `(b,y)` 目标。

**Q-I：UNKNOWN-Interval.** 对相同候选 Ray 中 `t_known` 之前的 UNKNOWN 连通区间构造 `(r, t_start, t_end, index, L_unknown, ray_feature)` Query。最多 512 Query：超限采用事先冻结的、只依据观测的 deterministic selection；不足 mask。必须明确多区间 Ray 的聚合方式、顺序 PE、如何映射回 Ray feature。**只把 Interval 用作几何条件；第一轮不改变 First-Hit 标签/概率定义，不为 First-Hit 后面的区间构造伪阴性标签。**

### 2.3 Geometry-Aware Attention

可在核心六组之后再启用为第七个补充测试，不与 S/M × Query 主因子同时混合：

`logit(i,j) = q_i·k_j / sqrt(d) + λ * B_geo(ray/interval, scene_token)`。

`B_geo` 包含 ray-to-token 横向距离、沿射线距离、观测几何状态、已知遮挡与 UNKNOWN interval 关系；使用连续 soft bias、默认不做彻底硬遮挡。严禁使用 GT future depth 或隐含未来 coverage 计算 bias。

## 3. 六组核心对照 + 锚点

| ID | Scene K/V | Query | 训练 | Outdoor D1 闭环 | Indoor H0 检查 |
|---|---|---|---|---|---|
| T1 (历史锚点) | μ @8×，≤512 | 8×16 bottleneck（128） | 已有 40K | 已有 70/70，必要时回归复跑 | 固定 3 层×2 起点 |
| XA-S2D | μ @8× | 16×32 Spatial（含 compute-matched 约束） | scratch 40K | **必须 70/70×100** | 6×150 |
| XA-SR | μ @8× | Ray 512 | scratch 40K | **必须 70/70×100** | 6×150 |
| XA-SI | μ @8× | Interval ≤512 | scratch 40K | **必须 70/70×100** | 6×150 |
| XA-M2D | F4/F8/F16 | Spatial 16×32 | scratch 40K | **必须 70/70×100** | 6×150 |
| XA-MR | F4/F8/F16 | Ray 512 | scratch 40K | **必须 70/70×100** | 6×150 |
| XA-MI | F4/F8/F16 | Interval ≤512 | scratch 40K | **必须 70/70×100** | 6×150 |

公平性说明：Q-I 的 Query budget 是区间数，可能从 512 个 Query 覆盖少于 512 条独立 Ray；额外报告独立 Ray 覆盖率和 Query 选择覆盖比例。S2D 16×32 与 T1 8×16 **不是单因子比较**，必须优先对照 XA-S2D 与其他 XA 变体；T1 保留为生产历史基线。若要隔离 Query 机制与 Token 数量，可另加 `T1-Q512` compute-matched control，但不阻断六个固定闭环。

建议可选的第七组（六组结束后）：`XA-MR-HYBRID`：8×16 Global 2D Query（F16/F8）+ Local Ray Query（F4/F8）并用 gated fusion。这是候选架构扩展，不纳入核心六组主因子表，除非同意增加计算预算；如执行，也必须完整跑 Outdoor 70/70。

## 4. 顺序自动化执行流程（统一提交，一条条完整执行）

```text
00 CT snapshot/checksum → CT safely stop
01 freeze XA protocol/config/manifests/splits and baseline hashes
02 implement shared XA modules, validate shapes/leakage/decoder
03 XA-S2D: scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
04 XA-SR:  scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
05 XA-SI:  scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
06 XA-M2D: scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
07 XA-MR:  scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
08 XA-MI:  scratch 40K → official Outdoor 70/70×100 → indoor 6×150 → aggregate
09 freeze reports; compare scene-paired deltas, CIs, efficiency, failure cases
```

**没有“预测指标不好 → 取消闭环”逻辑**；模型训练完可被正确加载就进入闭环。Engineering smoke test 只检测 finite loss、shape、GT guard、数据合法性、执行器能否跑通 1 次短流程，不能用于筛选效果。每个模型训练/仿真/覆盖回放由 Slurm 持久化 `sbatch` 作业和 `afterok` 依赖串联；不依赖 login 的 tmux/salloc。`afterok` 若前项异常失败会阻断后项：必须有 supervisor/retry 或 recovery job 处理 failed episode 后继续该模型；不能悄悄跳过到下一模型或把缺失 episode 当作成功。

### 自动恢复规则

- **训练中断**：从最近的 full checkpoint（model/EMA/optimizer/global_step/config）恢复；保存恢复来源与 RNG 重播种限制；错误超过可恢复范围记录失败并进入人工异常通知，不造结果。
- **CUDA OOM**：先减少 view micro-batch / attention chunk，通过相同 global batch 的梯度累积恢复；记录配置变化，对照统一同等设置；不能直接丢掉大场景或降低 Outdoor 数量。
- **Episode 失败**：保留完整日志、固定 seed 重跑该条直到得到 101 个 coverage 样本；不得用提前结束的结果补齐。
- **只有合法性、安全、数值故障可阻断**；指标好坏不作为继续条件。完成 70/70、全部 checksum 与报告后再切下一个 ID。
- **不要自动删除原结果**，用唯一 run ID、原子保存和 SHA 归档每个版本。

## 5. 统一闭环与输出

### 5.1 Outdoor：强制官方 All-14

- **14 场景、70 episode、100 steps**；与现有 Table 1 同 5 起点、pose/seed、第一层官方合法候选、D100 输入、S10 novelty、R10 编码、K4/Euler10、D1 Planner。
- 同步保存官方 `coverage_official[0..100]` 与可选 `coverage_monotonic`。官方曲线可能下降；**不得做 `cummax()` 替代官方主表数据**。
- 每个 episode：`AUC20 = trapz(C[0:21])/20`; `Cov20=C[20]`; `AUC100=trapz(C[0:101])/100`; `Final=C[100]`。另外 AUC50、Cov50。
- 每个场景先平均 5 个 episode，再做 14 场景**等权平均**。必须分别报告 Train-6、Dev-5、Test-3、Unseen-8、All-14，不能称 Test-3 为 blind test；尤其主文应该显示 Unseen-8 和 Test-3，避免 All-14 中 6 个见过的训练场景导致泛化夸大。
- 与 T1+D1 及相同 Planner 的 B2（如未来补跑）做同起点配对差；以场景为单位 bootstrap 95% CI。不是把 70 个 episode 当作 70 个独立场景。
- 必须保留 70 条 coverage 曲线、per-scene 表、失败场景、early timestep 行为和规划时间，不仅有单行平均数。

### 5.2 Indoor：小样本固定检查

- 仅选 3 个 MP3D val 楼层、每个 2 个固定起点，150 步（从既有 G3 室内列表先冻结）。先给 T1 室内 H0 同步对照，再按相同 H0、n_win=5、C_obs 定义评估所有 XA 模型。
- 指标：`C_obs AUC@20`、`C_obs Coverage@20`、`C_obs AUC@100`、`C_obs Coverage@100`、`C_obs AUC@150/Final`。如果模型室内不适用，报告原因，不能省略模型或用室外覆盖代替。
- 不声称 Indoor D1 结果，因为审计显示 Indoor D1/H3 接口 `NOT_FOUND`；若后续确实新建并验证 Indoor D1，应另表清晰区分 protocol。

### 5.3 辅助指标（不做早停条件）

- Prediction：HIT/NOHIT、first-hit error（在对应有效标签上）、calibration、unknown-boundary/ray-depth 分组。
- Decision：候选视角 Top-1 vs oracle gain、oracle regret、rank correlation、早期 t≤20 的错误分布。
- 资源：总参数、peak GPU memory、每步真实 wall-clock planning time（尤其 D1 两步前瞻）、训练 GPU-hours、每 episode timeout/retry 数。

## 6. 日志、完成标准与论文追溯

建议：

```text
experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/
  README.md
  FROZEN_PROTOCOL.md
  configs/{base,xa_s2d,xa_sr,xa_si,xa_m2d,xa_mr,xa_mi}.yaml
  scripts/{train_xa,run_outdoor_d1,run_indoor_h0,aggregate,run_chain}.py
  manifests/{ct_paused,code_hashes,splits,checkpoints,runs}.json
  results/
    xa_s2d/{checkpoints,episodes,outdoor,indoor,logs,report.md}
    xa_sr/...
    xa_si/...
    xa_m2d/...
    xa_mr/...
    xa_mi/...
  reports/{MASTER_TABLE.csv,PER_SCENE.csv,EARLY_CURVES.png,FINAL_REPORT.md}

seen2scene/v3/xa/
  {vae_pyramid,memory,spatial_query,ray_query,interval_query,attention,xa_model}.py
```

**每个模型 COMPLETE 的定义：** (i) 训练正常结束，有可加载的最佳/最终 EMA Checkpoint 及 SHA、明确选模规则；(ii) Outdoor 70/70 各有 101 个官方 coverage 值；(iii) Indoor 预定义 6/6 各有完整 151 个 C_obs 值或明确客观 incompatibility；(iv) 原始结果、指标表、逐场景图、耗时和差异报告保存；(v) 代码与配置哈希记录。未完整跑完不得标记 COMPLETE 或进入最终统计。训练中如果 NaN 或形状不符应修复，而不是让坏模型被错误评价。

**最终交付：** `FINAL_REPORT.md` 同时包含模型架构、Q/K/V 维度、数据与训练协议、Outdoor 全量主表、Unseen-8、Test-3、Indoor sanity、Coverage Curves、有效预测证据、运行开销、失败模式和所有路径的可复现命令；以结果而非预设偏好决定是否最终选 XA-MR、XA-MI 或保留 T1。

## 7. 可直接交给仓库 Coding Agent 的执行指令

> 你现在在 seen2scene 项目中执行 Experiment XA，请先读取 `CVPR_MODEL_AUDIT.md`、现有 V3/T1 训练入口、D1 的官方 Outdoor All-14 规划入口、表格聚合与 MP3D G3 H0 室内入口，并以这份 `XA_Experiment_Protocol.md` 为冻结要求。第一件事是确认 CT 真实 Slurm 状态，保存 full checkpoint + EMA + optimizer + manifest + SHA 后再停止；保存未核验之前不能 scancel。不得修改 CT 历史结果或原始 V3 T1 代码路径。然后在 `experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/` 创建实验与日志，模型组件放 `seen2scene/v3/xa/`。按 XA-S2D、XA-SR、XA-SI、XA-M2D、XA-MR、XA-MI 顺序逐一实现、40K 从零训练、逐一运行 Outdoor D1 All-14 70/70 × 100 步官方覆盖闭环，再跑 MP3D 固定 3 楼层×2 起点×150 步 H0 检查，并保存单模型完整报告后继续下一模型。不能用预测 loss、固定状态分数或头 20 步效果差作为取消闭环的理由。每个模型记录 AUC@20、Coverage@20、AUC@100、Final、逐场景结果、CI、显存/耗时；户外是主结果，室内小规模 sanity。自动队列使用持久化 Slurm jobs + 安全重试，不要依赖 tmux；一旦工程检查通过不再等待人工批准，按冻结队列连续执行。对工程失败可自动诊断修复和安全恢复，但绝不能删减测试场景、虚构缺失 episode、泄露 GT 或把不完整结果标记 COMPLETE。最后统一生成 `FINAL_REPORT.md` 和机器可读聚合表。请真实读取代码核对命令，不要假定上述建议文件或脚本已存在。

---

**证据来源：** 用户提供的 `CVPR_MODEL_AUDIT.md`（2026-11-05，审计代码 commit `f7d8f06`）。本协议中的 Multi-Scale / Query / 自动队列设计为下一阶段建议，未被历史实验验证。
