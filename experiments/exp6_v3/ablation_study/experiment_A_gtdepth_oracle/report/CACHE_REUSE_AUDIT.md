# Experiment A — 缓存复用审计（2026-10-31 09:45，运行任何新计算之前）

> 结论先行：**EXP5 A1 / A2 的 1,461 个闭环固定状态可全部复用**（oracle 增益、GT 深度 t\*、V2O / V2M 的 K=4 原始样本、GT 深度 DistP1 均逐位复用，标注 `historical EXP5`）；**必须补算的只有 B2 / E2-D100 / V3-T0 / V3-T1 在这些状态上的 K=4 预测与 DistP1**（它们需要 D100 输入地图，Task 2 的 svc 记录保存了完整深度帧，可逐帧重放）。EXP6 G2 的 365 个固定状态已有四个模型的 K=4 分数，**只缺 GT 深度 DistP1**（t\* 已在 oracle.npz 中）；其中 Dev-3 B0 层的 150 个状态与 EXP5 池重合，用作新重放的逐位回归测试。

## 1. 已读文件
`plan.md`、`experiments/README.md`、`exp6_v3/protocol.md` v1.0、`exp6_v3/reports/V3_COMBINED_REPORT.md`、`task*/report/T*_{G12,G3}.md`、`exp6_v3/reports/V3_DEVIATIONS.md`（#1–#10）、`exp5_future_view/ablation_study/{protocol.md, DEVIATIONS.md, reports/ABLATION_SUMMARY.md}`、`A1_prediction_scoring/A1_REPORT.md`（摘要）、`common/{ab_common, ab_load, s0_states, s1_oracle_in, s1_oracle_out, s2_infer, s5_verify}.py`、`A1_prediction_scoring/{a1_analyze, a1_region_diag, a1_relevance_diag}.py`、`task3_full_depth_p1/density_ablation/scripts/e12_fixed.py`、`e2_pilot/scripts/e2_fixed.py`、`exp6_v3/shared/eval_fixed.py`、`seen2scene/futureview/{geometry, scoring, predictor}.py`、`task1/scripts/v2_plan_core.py`、`seen2scene/v3/predictor.py`。

## 2. 两个固定状态池

### 2.1 Pool CL — EXP5 A1 / A2 的闭环固定状态（`work_dirs/exp5_future_view/ablation_study/`）
| 项 | 内容 | 复用方式 |
|---|---|---|
| 状态清单 | `manifests/states_{indoor,outdoor}.json`：MP3D 368（5 val 楼层 × 2 起点 × 4 来源 V2O / V2M / MAG-G / MAG-B × t ∈ {0,5,20,25,50,55,100,105,140,145}，去重）；Macarons++ 1,093（Test-3 + Dev-3 × 5 起点 × 4 来源 × t ∈ {0,5,10,15,20,25,50,55,80,85}）；每条含 `sid, scene, start, t, primary_t, stale, cand, legal_idx, sources(含 V2 来源的线上分数与执行动作)` | **逐位复用**（状态 id = 执行历史的 sha256） |
| 候选集 | MP3D = Task 2 V2 选择规则的候选（合法动作中未执行过的位姿，`ab_common.candidate_set`）；Macarons++ = 环境合法列表 | **逐位复用**（新模型打分遍历同一 `legal_idx`，排序只在 `cand` 上算） |
| Oracle 增益 A3 | `states/<d>/<sid>/oracle.npz`：`gain`（MP3D C_obs 规则；Macarons++ 单调覆盖 `gain` + 官方 `gain_official`）、`cov_t`、`remaining`、`cand_pts`、`legal_idx`；EXP5 S5 校验 V1 / V2 通过（执行动作的增益 = Task 2 曲线增量 400/400、1,200/1,200；全部候选独立重算 368/368） | **逐位复用**，模型无关 |
| GT 深度 t\* | 同文件 `tstar [n_legal, 7296]`（64 × 114 打分射线，jitter 0，冻结渲染器） | **逐位复用** |
| GT 深度 DistP1（A2） | `v2.json: gtd_distp1`（`SCO.new_set_from_depth(t*, …)` 在**稀疏打分图**上，距离权重 min(1,(z/z0)²)，与线上 DistP1 同一函数）；S5 V3 重打分 36/36 逐位 | **逐位复用**：B2 / V3 的打分图同样是稀疏 band_dil（"稠密输入 / 稀疏打分"），几何 = 同一 P1 记录 0..t，因此 A2 分数对所有待比模型相同 |
| V2O / V2M 预测（A1，稀疏输入 = B0 口径） | `v2.npz`：每候选 K=4 的 `t, hit_u, b, y`、标签 `lab_cat, lab_b, m_unk, lm, gt_new, valid`、射线 `o, d, t_known, cls`；`v2.json: distp1_k, distp1, repro` | **逐位复用**，标注 `historical EXP5`（V2M-sparse = B0 配置；**不是** B2） |
| 噪声 | 室内 `exp5-V2\|task0cl\|<tag>\|t<r>\|k<k>`，室外 `closed_loop_v1.seed_key(scene,start,0,r,k)`，r = 刷新步 | **复用同一种子** → 新模型与 V2O / V2M 严格配对 |
| K 消融 | `kabl.npz`（V2O / V2M，K ∈ {1,4,8,16} × 4 组种子） | 复用为历史参照；本轮不扩大 K |
| svc 记录（重放输入） | `work_dirs/exp5_future_view/task2/eval/{indoor,outdoor}/closed_loop/<src>/…/svc/svc_*.npz`：`zbuf [256,456] float32, mask, R, T, cam_center, pose_idx, sample_pixel_flat, sample_points, executed_pose_idx_history, legal_idx/R/T/center`；四个来源组室内 10 / 室外 30 个 episode **全部在**（10-30 清理只删了 Task 3 / Task 1 的 svc） | **D100 重放可行**：`P1FusionFull.integrate_record` 用 `zbuf` 全部有效像素；稀疏图用 `sample_points` |

**必须补算**（Pool CL）：B2（V2-Mix sha `343f41ab…`，D100 输入）、E2-D100（`e2/training/d100/ema_005000.pt`）、V3-T0（`7ed0ddd7…`）、V3-T1（`639c9d3a…`）的 K=4 预测（原始 `t, hit_u`）与 DistP1（稀疏打分）。理由：这四个模型在 EXP5 A1 时不存在 / 未纳入；其中 V2M 的 D100 输入版本（= B2）与 A1 的 V2M 稀疏输入版本是不同的规划器配置。成本估计见协议 §4。

### 2.2 Pool FX — EXP6 G2 的固定状态（`work_dirs/exp6_v3/eval_fixed/`、`e2/fixed/`）
| 子集 | 状态 | 已有 | 缺 | 处理 |
|---|---|---|---|---|
| VALSEL（MP3D 6 val 房屋 9 楼层，E2 轨迹 ogr / rnv） | 9 ep × 10 = 90；候选 = 合法列表（≤ 10） | B2 / T0 / T1（EXP6 行，逐位核对 E2）、E2-D100 / B0 / S10（E2 行）的 K=4 分数与聚合预测误差；oracle `gain`、`tstar`（`e2/valsel/states/indoor/<sid>/oracle.npz`） | GT 深度 DistP1；逐射线原始样本（Experiment B 需要） | 补算：稀疏几何重放 → GTD；重放 D100 → 原始样本（分数须与 EXP6 行逐位相同 = 回归） |
| Dev-3 B0 层（Task 2 V2M 轨迹） | 15 ep，150 状态 | 同上（EXP6 / E2 行）；**这些 sid ⊂ Pool CL**（`e12_fixed.states_of("outdoor","B0")` 取自 EXP5 清单的 V2M 来源） | 无（GTD、V2O/V2M 样本在 Pool CL） | **回归测试**：Pool CL 新重放的 B2 / T0 / T1 / E2D 分数必须与 EXP6 / E2 行逐位相同 |
| Dev-3 B2 层（Task 3 B2 闭环轨迹） | 15 ep，125 状态 | 同上；oracle + `tstar`（`e12/abroot_B2/states/outdoor/<sid>/oracle.npz`） | GTD；原始样本 | 同 VALSEL |

### 2.3 不复用 / 不重跑的历史结论（直接引用，标注 `historical EXP5, not new V3 result`）
A1 的 V2O / V2M 排序表（Spearman 0.43 / 0.55，GTD 0.66；Top-1 0.41 / 0.56 / 0.69；regret 0.210 / 0.119 / 0.064 pp 室内，0.607 / 0.482 / 0.160 室外）、区域裁剪诊断（GTD 选错状态中 oracle 最优候选 53% 增益在区域外；只计区域内 Top-1 0.69 → 0.81）、室外相关性诊断（选中候选新表面 74% / 88% 与评估相关）、K 消融（K > 4 无收益）、A2 多样性与幻象遮挡率。本轮只在 **新模型（B2 / T1 / T0 / E2D）的选点** 上重算区域与相关性诊断，算法复用 `a1_region_diag.py` / `a1_relevance_diag.py`。

## 3. 协议差异与风险
| 项 | 说明 |
|---|---|
| 输入地图 | Pool CL 的历史 V2O / V2M 分数来自稀疏 P1 输入（B0 口径）；新模型用 D100 输入。两者在同一状态、同一候选、同一噪声、同一打分图上比较，差异 = 模型 + 输入口径（与 Task 3 的 B0 / B2 对照一致），报告中明确标注 |
| 打分图 | 所有模型（含 GTD）都用稀疏 band_dil 打分（B2 / V3 / Task 3 冻结规则）；GTD 在新口径下无需重算 |
| 相位分箱 | Pool CL 沿用 EXP5 `t_bin`（early ≤ 25 / 15，mid ≤ 55，late）；Pool FX 沿用 EXP6 G2 的 `phase_of`（< 0.2 T early，≥ 0.8 T late）；两套都在表头注明，不混用 |
| Test-3 | Pool CL 含 Test-3，Task 2 / EXP5 A1 已使用，**非盲**，全部诊断标 exploratory |
| 显存 | D100 融合 + 稀疏融合 + 4 个模型（V2M 共享一个 core 给 B2）+ K=4 采样：eval_fixed 在 46 GB 卡上同时跑 B2 + T0 + T1 无问题；E1.2 记录 D100 室外大场景（colosseum）峰值较高 → 每卡 1 个进程，`expandable_segments` |
| GT 泄漏 | 规划侧（融合、编码、采样、DistP1）只读 svc 记录；`oracle.npz` 在该状态全部模型分数写出之后才读取（沿用 s2_infer / eval_fixed 的顺序） |
| 非确定性 | EXP6 G1 / G2 出现过一次性 B2 回放不逐位（偏差 #10）；本轮回归测试若失配，先原样重跑该 episode 一次再判定 |

## 4. 复用率
- Pool CL：oracle / t\* / GTD / V2O / V2M / 候选 / 噪声 **100% 复用**；新增 4 个模型 × 1,461 状态的预测（约 2–3 GPU-h，见协议）。
- Pool FX：四个模型分数 100% 复用（365 状态）；新增 GTD（215 状态需稀疏重放；150 状态直接取 Pool CL）与原始样本（215 状态）。
- 不重跑：GT 渲染、oracle 评估、EXP5 的任何表格与图。
