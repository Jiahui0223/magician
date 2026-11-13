# U0 — Pipeline 审计 + 冻结设计（Experiment U，协议 U-v1）

2026-11-13 · 范围：Phase A（审计）、Phase B（指标定义）、Phase C（策略与适配层）。本文件与 `configs/u_config.yaml`（**sha256 `8a62ffec…`，2026-11-13 02:45 写入**）共同构成冻结协议 U-v1。冻结发生在任何 Experiment U 模型运行之前（第一次运行 = 05:23 的冒烟；固定状态 05:24 起）。所有冒烟、固定状态运行的规划器日志里都记录了同一个 config sha（`U_POLICY` 行）。

**一句话结论**
- V3-T0 对每个候选视角生成的是 K = 4 张**视角条件化的首次命中深度假设**（每条射线一个 `(b̂, ŷ)`），不是完整三维场景。分歧只能在这些深度及其反投影点上计算。
- 现有闭环在同一步内对所有候选只采样一次，结果可以直接复用来算 U1–U3，不需要额外跑模型（单步 + 0.01 s/步，见冒烟）。
- C1（D1 的一步覆盖分数）就是已归档的 Table 3 行 `V3-T0-K4D1`（70/70，回放逐位一致）；本实验的适配层在 C1 模式下逐位复现它（冒烟：执行位姿 16/16、覆盖差 0.0）。
- 同一噪声编号在不同视角下**不对应同一个三维世界**（多视角研究在 T1 上已测到，U 的 Phase D 在 T0 上用预注册规则复核），因此 U4 的“联合分歧”只有在 Phase D 通过预注册检验时才定义。

---

## A. Pipeline 审计（8 个问题）

### A1. V3-T0@40K 的 checkpoint 与 SHA256
| 项 | 值 |
|---|---|
| 路径 | `work_dirs/exp6_v3/task0_vae/checkpoints/final.pt`（= `work_dirs/checkpoints/exp6_v3/T0_vae/v3t0_ema40000.pt`，94,289,914 字节，2026-10-30 22:18） |
| SHA256 | `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6`（2026-11-13 `sha256sum` 实测，与 `cvpr_lib.T0_SHA`、架构文档一致） |
| 加载 | `seen2scene/v3/predictor.py: load_core(ckpt_path, device, steps=10, expect_sha)` → `V3Model(ck["model_cfg"])`，载入 `ck["ema"]`；闭环里由 `experiments/exp6_v3/shared/planner_v3.py: ckpt_override()` 用环境变量 `V3_CKPT` 替换命令行的 `--ckpt/--sha256`，`V3Core` 校验 sha |
| 本实验 | 只读；`u_fixed.py` 用 `expect_sha` 校验；每个闭环规划器日志 `pl_*.json["ckpt"]` 记录 path + sha（冒烟与 episode 测试中均为 `7ed0ddd7…`） |

### A2. K = 4 组噪声如何生成、缓存、在候选间复用
- 生成：`seen2scene/futureview/predictor.py: noise(device, K, seed_fn)`，每个 k 一个独立 `torch.Generator`，`manual_seed(seed_fn(k) % 2**63)`，形状 `[K, 2, 64, 128]`。
- 种子：`v2_planner.refresh()` 里 `seed_fn = lambda kk: RY.sha_seed(LC.seed_key(s, k, rep, t, kk))`；`seed_key` = `"exp5-V2|{scene}|s{start}|rep0|t{t_r}|k{k}"`（`task1/configs/closed_loop_v1.json`）；`sha_seed` = sha256 前 15 位十六进制。
- 缓存：R10 —— `LC.refresh_time(exp, t) = 10·⌊t/10⌋`；只在刷新步重新编码 `h = core.pred.encode(geo.coarse())` 和重新生成噪声，存在 `state["h"]`、`state["eps"]`，同一刷新块内每步复用。
- 候选间复用：`V3Predictor.sample(pack, m_unk, eps)` 中 `eps[ki]` 对所有候选 `vi` 共用（公共随机数）；D1 的第二层子视角也用同一 `eps`（`d_policy.lookahead → ensure → orig_score(..., eps, ...)`）。
- **潜在问题**：噪声是**图像空间**的（每个像素位置一个噪声值），不同视角的同一像素对应不同的三维射线。因此“样本 k”只在单个视角内有意义，跨视角不是同一个世界（多视角研究：不同相机之间同编号 / 不同编号的加权 Jaccard 0.0075 / 0.0071，“看穿”率 0.355 / 0.362，T1，395 状态）。这直接影响任何把不同视角的同编号样本当作联合样本的定义（D1 的并集配对、U4 的联合单元）。

### A3. 模型如何得到 (b̂, ŷ)
调用链（`experiments/exp5_future_view/task1/scripts/v2_plan_core.py: score_state`）：
1. `geo.view(R, T)` → `VF.view_rays`（7,296 条打分射线，64 × 114）→ `SceneCaster.cast` 得已知表面 `t_known` → `geometry.project(...)` 得 `pr`。
2. `core.pred.cond(h, views)`（`seen2scene/v3/predictor.py: V3Predictor.cond`）：11 通道投影 `G.channels(pr, tcap, d, o)` + `model.view_cond(h, o, d, tcap, ch)`（μ 沿射线 32 点采样 → LatentRayEncoder → 条件图 `[V, 76, 64, 128]`）；`m_unk = M.pad_rays(cls == CLS_UNKNOWN)`。
3. `core.pred.sample(pack, m_unk, eps)` → `seen2scene/v3/model.py: sample(model, cond, m_unk, eps, mem=None, steps=10)`：Euler 10 步，`x ← (x + v/steps)·m_unk`，初值 `eps·m_unk`；bf16 autocast；每批 `max_views = 64` 个（视角 × 样本）。
4. `M.unpad_rays(...)` 去掉补齐列 → `b, y = x[:, :, 0], x[:, :, 1]`，形状 `[V, K, 7296]`。
- **注意**：非 UNKNOWN 射线在每个 Euler 步都被 `m_unk` 置 0，因此 `b̂ = 0` → 解码时不算命中。

### A4. 生成结果如何解码为 First-Hit Depth
`seen2scene/futureview/geometry.py: decode(b_hat, y_hat, o, d, pr, struct, domain)`：
- `hit_u = (b̂ > 0) & (ŷ < 1) & (n_u > 0)`；
- `s = clamp((ŷ+1)/2, 0, 1−1e−6)`，取该射线第 `⌊s·n_u⌋` 个 UNKNOWN 样本（步长 `H_STEP = 0.5·VS = 0.0055` 模型单位）的位置作为命中距离 `t`；
- 其余射线（非 UNKNOWN 或该样本无 UNKNOWN 命中）取投影结果：`t_known`（已知表面）或 `inf`（无命中）。
- 输出 `t [K, R]`（模型单位；运行时射线长度 = `t·k_M`），`hit_u [K, R]`。
- **潜在问题**：命中只能落在 UNKNOWN 样本上，所以生成的表面永远不会落在已观测 EMPTY 里；“没有命中”的样本取 `t_known` 或 `inf`，**不能**拿 `t` 直接算深度方差（会把已知表面 / 无穷远混进去）——U2 只用 `hit_u` 样本的 `t`。

### A5. UNKNOWN Mask、Known Depth、Validity Mask
`geometry.project(o, d, tcap, ext, t_known, struct, domain)` 返回 `pr`：
| 字段 | 含义 | 代码 |
|---|---|---|
| `tmin, tmax, valid, M` | 射线进入 / 离开场景网格（`tmax = min(exit, tcap)`），`valid` = 与网格相交 | `march_range` |
| `t_known` | 当前 P1 已知表面（BAND）的首次命中，无则 `inf` | `Geometry.view` 中 `SceneCaster.cast(vol_obs, …)` |
| `n_u` | `t_known` 之前、`tcap` 之内、网格与 domain 内的 UNKNOWN 样本数 | `unknown_samples` + `ray_stats` |
| `d_lo, d_hi, n_seg` | 首 / 末 UNKNOWN 样本位置、段数 | `ray_stats` |
| `cls` | `CLS_UNKNOWN` 若 `n_u > 0`，否则 KNOWN（有 `t_known`）或 EMPTY | `ray_class` |
- `unknown_samples` 要求 `valid`，所以 `n_u > 0 ⇒ valid`；补齐射线（64 × 14 列）在 `unpad_rays` 后已不存在。
- **R_c 的实现**：`u_metrics.ray_set(v) = (pr["n_u"] > 0) & pr["valid"]`。

### A6. D1 如何计算新增覆盖单元、如何选第二步
- 单样本价值：`experiments/exp6_v3/H/scripts/h_policy.py: value_cov(t, v, geo, z0, cfg)`：所有有限命中（含 `t_known`）→ domain 检查 → 运行时坐标 → 与已观测 S10 采样点的 KD-tree 距离 > `eps_cov = 1.0` 才算新 → 1.0 运行时单位格子去重（key = `(c0<<40)|(c1<<20)|c2`，偏移 2¹⁹）→ 权重 `w = min((z/z0)², 1)`（z = z 深度，`z0` = 场景尺度/2），每格取最大权重。
- 一步：`d_policy.score_state` 里 `s_one = mean_k Σ w`（λ = 0）。
- 第二步：`d_policy.lookahead`：对每个 c1，`HP.neighbours`（设置网格 6 移动 + 4 旋转，钳位 / 方位取模）→ 去掉路径上的位姿 → 未访问优先（历史 + 路径算已访问）→ `HP.segment_blocked`（穿过已知 D100 表面带的线段丢弃；不读 GT）；c2 用同一 `h`、同一 `eps` 预测（16 个一批，每步每位姿只算一次）；每个样本 k 取 c1 与 c2 新单元的并集（每格最大权重），`Q = mean_k`，c1 分数 = 子节点 Q 的最大值；无子节点时用 `s_one`。
- 选择：`m_score.choose`：argmax、并列取最小下标、全 0 → `MC.fallback_choice(scene, start, t, n)`（种子化随机）。
- 配置：`experiment_D_h3_lookahead/configs/d_config.yaml`（D-v1，sha `e53c230d…`），代码 `d_policy.py`（sha `c8418279…`）、`h_policy.py`（sha `ccfdf4af…`）。
- **潜在问题**：(1) 并集按同编号配对（见 A2）；(2) `value_cov` 先把体素下标钳到网格内再查 domain，GT 深度代入时会把网格外命中记到边界体素（F0 已查明，只影响 GT 对照，不影响模型预测）。

### A7. Coverage Evaluator 接口、统计方法、All-14 名单
- 闭环：`exp5_future_view/task2/scripts/t24_out_sim.py`（冻结 M 协议 `m_t6_sim.EpisodeSim`：官方设置网格与 5 个官方起点，MAGICIAN `get_neighboring_poses → get_valid_neighbors →` GT 线段碰撞（只在模拟器侧），种子化观测，`gt_scene.scene_coverage`）。每步写 `coverage`（101 个值），覆盖有上限、点可替换，曲线可能下降，**不取 cummax**。
- 回放：`t25_out_cov.py`（本实验经 `scripts/u_out_cov.py` 原样调用，只把临时帧目录换到 `work_dirs/exp6_v3/experiment_U/cov_tmp`）：按执行位姿重放，`cov_official` 必须与存档曲线逐位相等，否则 STOP。
- 统计（`experiments/cvpr_results/scripts/cvpr_lib.py`）：`AUC@T = trapz(C[0..T])/T`，`Cov@T = C[T]`，`Final = C[100]`；场景内 5 起点平均，再场景等权；配对场景 bootstrap 10,000 次、seed 0。
- 名单（`table1/SCENE_REGISTRY.yaml` → `cvpr_lib.GROUPS`）：Train-6 = dunnottar, bannerman, redeemer, liberty, pisa, fushimi；Dev-5 = colosseum, pantheon, eiffel, barts, sestino_museum；Test-3 = alhambra, neuschwanstein, bridge（St. Sofia 不在 All-14）。All-14 **不能整体称为 unseen**：V3-T0 的训练数据来自 Train 场景。
- 已核对的基线（`load_manifest`）：`V3-T0-D1` All-14 = **0.420 / 0.646 / 0.633 / 0.763 / 0.911**（AUC@20 / Cov@20 / AUC@50 / AUC@100 / Final），70/70，ckpt `7ed0ddd7`，与任务书一致；`V3-T0-K4D1`（= C1）= 0.399 / 0.621 / 0.609 / 0.740 / 0.889。

### A8. 能否不重新运行生成模型而复用候选的 K 个预测
- **同一步内：可以，而且就是这样实现的**。`score_state(keep_hits=True)` 返回每个候选的 `t, hit_u, b, y`；U1–U3 直接从这些张量算，与 C1/D1 用的是同一批预测。冒烟实测额外开销 0.009–0.012 s/步。
- **跨步：不可以**。P1 每步都变（投影通道变），R10 只缓存编码特征 `h` 和噪声，不缓存预测。
- **离线（Phase D）**：闭环日志只存分数，不存原始预测；必须在回放的状态上重跑模型。Experiment F / CT / EXP9 Table 4 V2 的回放链（`ct_fixed.run_outdoor → f_run → mv_core.run_planner`）已对 V3-T0 + D1 逐位验证；本实验沿用，D1 分数必须与 Table 4 V2 的 T0 行逐位一致（回归）。

### A9. 审计中发现的其它问题 / 风险（供 U 的设计使用）
| # | 问题 | 对 U 的影响与处理 |
|---|---|---|
| 1 | 噪声编号跨视角不配对（A2） | U4 联合单元默认不成立，需 Phase D 预注册检验 |
| 2 | 1.0 运行时单位的格子相对表面曲率很粗；不同样本表面落在格子边界两侧就会被记为“分歧” | U3 有离散化下限；Phase D 报告 Jaccard、Chamfer 作对照 |
| 3 | 命中只在 UNKNOWN 样本上，且 `y` 在 UNKNOWN 弧长上是相对坐标 | U2 用运行时射线距离（真实三维），弧长相对版本只作离线诊断 |
| 4 | 远处表面：同样的相对误差在远处对应更大的绝对方差 | Phase D 报告 U2 与平均命中距离的相关 |
| 5 | 早期（t ≈ 0）几乎所有射线都是 UNKNOWN（`n_rc = 7296`），U 的分母不变，比较只看分子 | 正常；Phase D 统计 U 与未知规模的相关 |
| 6 | 全 0 分数 → 种子化随机回退 | 与 D1 规则相同；记录回退次数 |
| 7 | 已有负面证据：Experiment B 的 DepthU（熵 + 深度标准差/UNKNOWN 跨度，按有效射线归一化，α = 0.5）在固定状态上 regret 变差 65–200%，门槛 8/8 FAIL，未进闭环（V3-T1 / B2） | U 与 B 的差别：(i) U1/U3 分别按 R_c 和预测单元归一化，(ii) U3 是三维单元层面的分歧（B 没有），(iii) V3-T0，(iv) 有受控的 C1 一步对照和完整闭环；但先验上 U1/U2 与 B 的 P1 高度相似，预期不乐观 |

---

## B. 指标定义（冻结，`u_config.yaml: metrics`，代码 `scripts/u_metrics.py: view_metrics`）

记候选 c 的 7,296 条打分射线（已去补齐），K = 4。

| 符号 | 定义 |
|---|---|
| **R_c** | `{r : n_u[r] > 0 且 valid[r]}`：射线在 `t_known` 之前、量程内、网格与 domain 内至少穿过 1 个 UNKNOWN 样本。补齐射线、无效射线、纯已知 / 纯空射线全部排除 |
| **h_r^(k)** | `hit_u[k, r]` = 解码规则 `(b̂>0) ∧ (ŷ<1) ∧ (n_u>0)`：样本 k 在 UNKNOWN 段上生成了表面。命中已知表面（`t_known`）不算 |
| **U1** | `U1(c) = (1/|R_c|) Σ_{r∈R_c} 4 p_r (1−p_r)`，`p_r = mean_k h_r^(k)`；`|R_c| = 0 → 0`（与任务书一致） |
| **U2** | `U2(c) = (1/|R2_c|) Σ_{r∈R2_c} Var_{ddof=1}{ t_r^(k)·k_M : h_r^(k)=1 }`，`R2_c = {r∈R_c : Σ_k h_r^(k) ≥ 2}`；单位 = 运行时单位²（官方覆盖的单位，eps = 1.0），等于同一射线上反投影三维点的方差；无命中样本不参与，不用最大深度填充；`|R2_c| = 0 → 0`。用 ddof = 1 是因为参与的样本数在 2–4 之间变化，无偏估计让不同命中数的射线可比 |
| U2-rel（**仅离线诊断**） | 同上，但用 UNKNOWN 弧长坐标 `s = clamp((ŷ+1)/2, 0, 1−1e−6)`；另记平均命中距离、变异系数、射线内方差与距离的 Spearman |
| **U3** | `S_c^(k)` = 样本 k 在 R_c 上的生成命中点 `o + t^(k) d` 反投影到运行时坐标后，落入的 1.0 单位格子集合（与 `h_policy.value_cov` 同一网格和 key；只含生成的 UNKNOWN 表面，不含 `t_known`；不做新颖性过滤、不加 w(z) 权重）；`V_c = ∪_k S_c^(k)`；`p_v = (1/K) Σ_k 1[v ∈ S_c^(k)]`；**`U3(c) = (1/|V_c|) Σ_{v∈V_c} 4 p_v (1−p_v)`**；`|V_c| = 0 → 0` |

**U3 的归一化规则（冻结）：对预测单元并集取平均（强度量，取值 [0, 1]）。**
- 理由：任务书要求防止“视野更大、预测点更多”的候选仅因单元多而得高分，并优先采用跨视野、跨 UNKNOWN 规模可公平比较的定义。所有候选的相机与射线数相同，差别只在看到多少未知空间；求和版本（`U3_sum`）会直接偏向未知区域大的视角，平均版本对规模不变。
- 已知代价（不调参、只报告）：平均版本在单元很少时方差大，可能偏向“未知区域很小但样本互相矛盾”的视角；K = 4 时单个单元的取值只有 {0, 0.75, 1}。Phase D 用 U3 与 `|V_c|`、`n_rc` 的相关和低收益案例检查这一点。
- 同时记录（只作诊断）：`U3_sum`、`|V_c|`、`p_v = 1` 的单元数、样本两两 Jaccard 均值、样本两两对称 Chamfer 距离（运行时单位，每样本按固定步长下采样到 ≤ 3,000 点）、`U1_sum`。
- 任何一个都**不是**校准过的信息增益；报告里只称“分歧（disagreement）”。

---

## C. 规划策略与受控对照（冻结，`u_config.yaml: policies`）

| 策略 | 打分 | 前瞻 | Phase E 来源 |
|---|---|---|---|
| D1 Coverage | `value_cov` 两步并集（λ = 0） | 两步 | **复用** Table 1 `V3-T0-D1`（不重跑、不覆盖） |
| **C1** Coverage | `value_cov` 一步：`mean_k Σ w`（= d_policy `d_one_step`，λ = 0） | 一步 | **复用** Table 3 `V3-T0-K4D1`（KD-v1 depth1，同 ckpt、同 K = 4 噪声、同模拟器，70/70 回放一致）；本适配层 c1 模式与它逐位一致（冒烟 liberty s0，15 步：执行位姿 16/16、覆盖差 0.0）。如需同一代码路径的再跑，`make_queue_tasks.py --policies c1` 可直接生成 |
| U1 / U2 / U3 | 上节定义 | 一步 | 本实验新跑 |
| U-best | U1–U3 中按预注册规则选出 | 一步 | 不单独跑（就是 U1–U3 之一）；它与 C1 的配对比较是主比较 |
| U4 | 见下 | 两步 | 条件实验 |

**适配层（`scripts/u_policy.py`，只改分数）**
1. 原样导入冻结的 `d_policy.py`，把模块级 `lookahead` 换成一步版本（与 `kd_policy._lookahead_depth1` 相同：返回 `s_one`，不预测任何第二层视角），再调用冻结的 `d_policy.install(policy)`（观测捕获、`value_cov`、`s_one` 全部不变）。
2. 外层包一层 `score_state`：调用上一步（`keep_hits=True`，同一批 V3 预测、同一 R10 特征、同一 K = 4 共享噪声），对每个合法候选算 U1/U2/U3；`u1/u2/u3` 策略把对应指标写进 `out["distp1"]`，`c1` 策略不改。所有策略都把 `u_c1 / u_u1 / u_u2 / u_u3 / u_metrics` 写进规划器步日志（`all_scores`），供 Figure U3 和事后分析。
3. 不改：V3-T0 权重、VAE / 射线编码器 / U-Net、K = 4、Euler 10、P1 / TSDF 更新（b2：D100 输入 + S10 band）、模拟器合法候选与 GT 碰撞检测（只在模拟器侧）、噪声键、`m_score.choose`（含全 0 回退）、官方覆盖与回放、每场景同样 5 个起点、100 步。
4. GT 隔离：规划器子进程仍在 `V2Guard`（LeakGuard）下运行，冒烟日志 `guard.denied = []`；`u_metrics` 只读 `t, hit_u, (y), o, d, pr, fr`。

**U-best 预注册规则（冻结）**
- 选择集 = Phase D 固定状态池（Experiment F 的 755 个户外状态，Dev-5 + Train 场景，不含 Test-3），**不使用任何闭环数字**。
- 统计量 = 按 `m_score.choose` 从该指标选出的候选的**官方一步覆盖增益（pp）**，状态 → 场景平均 → 场景等权平均；取最大者；完全相等时按 U1 < U2 < U3 的顺序取前者。
- 理由：Phase E 三个 U 都跑 All-14，若用 All-14 闭环结果挑 U-best，会把选择偏差带进主比较；固定状态的官方增益与最终评价指标同一口径（覆盖），又与闭环数据独立。Test-3 在选择中完全未出现，Table U2（Dev / Test）里 Test-3 是干净的留出集；Dev-5 的固定状态参与了选择（只用离线 oracle，不用闭环），报告中注明。

**U4 预注册规则（冻结）**
- 联合分歧需要“同一噪声编号在两个视角里代表同一个世界”。检验（Phase D，`u_metrics.overlap_agreement`）：对每条 D1 路径 (c1, c2)，在两个视角都至少一次预测到的单元 O 上，计算样本 k(c1) 与样本 j(c2) 的单元一致度 `|A_k ∩ B_j ∩ O| / |(A_k ∪ B_j) ∩ O|`，分别汇总同编号（k = j）与不同编号（k ≠ j）。
- **只有当场景配对 bootstrap 的（同编号 − 不同编号）95% CI 下界 > 0，且同编号 / 不同编号 ≥ 1.25 时，U4 才定义为联合单元分歧**（每个样本取 c1、c2 单元并集后按 U3 的平均式计算，c1 分数 = 子节点上的最大值）。
- 否则 U4 = NOT DEFINED：不构造联合单元，不进入 Phase E；“逐视角分歧两步相加”只能作为单独标注的启发式，并需用户批准后再跑。
- 先验证据（T1 + D-v2，多视角研究）：同 / 不同编号加权 Jaccard 0.0075 / 0.0071，不支持配对。

---

## D. Phase D 状态池的选择

| 池 | 状态 | oracle | 是否使用 | 原因 |
|---|---|---|---|---|
| **Experiment F 池**（explore 395 + confirm 360 = 755） | 户外 Macarons++；Dev-5：colosseum 192、barts 91、pantheon 92、eiffel 60、sestino 60；Train：bannerman / fushimi / liberty / redeemer 各 60、pisa 20；无 Test-3、无 dunnottar | 每个合法候选的**官方一步覆盖增益** `dcov_pp`（F 的 `oracle`，由实验 E 的 GT 反事实工具生成，F0 中 755/755 覆盖全部候选）；另有 F 的几何一步真值 `geo1` | **使用** | V3-T0 + D1 的回放链已在 EXP9 Table 4 V2 上逐位验证；记录全部 D1 第二层子视角（U4 检验需要）；与 Phase E 同协议（户外、同一模拟器规则） |
| Experiment A Pool CL（1,461） / Pool FX（365） | EXP5 闭环与 EXP6 G2 状态，含大量 MP3D 室内；室外部分是 V2 轨迹，稀疏 P1 输入口径 | 室外有 `gain_official` | 不使用 | D1 没有室内入口；室外 Dev-3 B0/B2 部分已包含在 F 的 explore 层；其余需要新的回放路径，本轮预算不值 |
| 多视角研究 395 | = F explore 层 | 同上 | 作为 F 的一部分使用 | — |

oracle 标签就是官方一步覆盖增益（MAGICIAN 官方覆盖，eps 1.0，覆盖上限规则）；它只在离线分析中读取。另外 `u_fixed.py` 对第一层候选做了**离线 GT 归因**（GT 首次命中 → `encode_labels` 的类别），只用于回答“分歧来自哪里”，不进入任何规划分数。

---

## E. 冒烟与回归（2026-11-13，共享 GPU 队列 `expU_a_*`、`expU_b_*`、`expU_d_*`）

| 项 | 结果 |
|---|---|
| C1 适配层 vs 归档 `V3-T0-K4D1`，liberty s0，15 步 | 执行位姿 16/16 相同，覆盖差 0.0；官方回放逐位一致 |
| U1 / U2 / U3，liberty s0，15 步 | 均跑通，无报错，回放逐位一致，分数有限，`u_*` 字段全部记录，分数 = 对应指标，回退 0 次；决策 0.78–0.90 s/步，U 计算 0.009–0.012 s/步，峰值显存 1.8 GB |
| Phase E 任务脚本 `u_episode.sh`（测试覆盖 10 步、测试目录） | 闭环 + 回放通过；第二次运行正确跳过（`SKIP done`） |
| Phase D 回放 `u_fixed.py`（conf_liberty_s0，t = 0、5） | keys / paths 与 F 一致，D1 分数与 Table 4 V2 的 T0 行逐位一致 |
