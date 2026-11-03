# Camera Pose Phase 1 —— 最终报告（2026-11-02）

## 结论：**NO-GO**（对 C1 Frontier-Aware Pose Selection）
在官方单步动作集合不变的前提下，加入 Frontier 几何先验来重新排序候选：
- 固定状态上只改变 3–5% 的选择，一步收益变化 ≤ 0.03 pp；
- 短闭环 pilot 中 4 个组合（B2 / E2-D100 × MP3D / Macarons++）的 mean ΔAUC 为 +0.002、−0.019、−0.020、−0.029，**全部没有通过预注册门槛**，因此没有运行完整 G3。

C2（多尺度 / 由粗到细）：室外在当前模拟器中 NOT_SUPPORTED；室内只能作为新动作协议，本阶段未运行。

所有数字来自 `RESULTS.md`、`fixed_state.json`、`pilot.json`。评估场景是此前用过的非盲场景。**pilot 结果不是完整 G3**。

## 1. 算法说明
**C0**（原始基线，不改）：
- 官方邻居生成，规则为 ±1 格只动一个位置轴，或 ±1 档俯仰 / 方位，最多 10 个；
- 经过合法过滤（GT 线段碰撞；室外由环境优先给出未访问位姿）；
- 模型 → K = 4 个未来深度样本（10 步 Euler）→ S10 DistP1（B2 / E2：D100 输入）；
- 原选择规则：室外 `m_score.choose`，室内 `nv_choice`（未执行过的候选优先），全为 0 时用带种子的 fallback。

**C1**（同一动作集合内重新排序）：
- FrontierPrior F_j：候选 j 的规划器射线中，先穿过已知 FREE、再在量程内进入 UNKNOWN 的射线，其进入点所在不同粗格（8 VS）的个数。只用规划器自己的 D100 P1 统计，**不用 GT**；被已知表面遮挡的射线不计入。
- 分数：`Score_C1 = D/maxD + λ·F/maxF`，选择规则不变。
- λ 从 {0, 0.1, 0.3} 中按预注册规则在开发固定状态上选出：B2 0.1、E2-D100 0.1、V4 0、V5 0。
- 实现方式：只在规划器子进程中包装 `v2_plan_core.score_state`（`scripts/c1_patch.py`）；模型、Flow、K、DistP1 都不变。λ = 0 时与 C0 逐位一致，已用 100 步回归验证。
- **它属于 Geometry-Aware Selection / Score Prior，不是候选扩展**：Oracle Best Gain 与 C0 完全相同，已断言。

**C2**：见 §6。

## 2. 当前 Camera Motion 的真实实现与限制（详见 `AUDIT.md`）
**两个域的动作都是“单步 = 一个轴 ±1”**：
- 室外：MAGICIAN fork 的 `get_neighboring_poses`（只移动或只旋转）+ `get_valid_neighbors` + GT 线段碰撞；位姿只能是 settings.json 网格（每场景 5D 索引）。
- 室内：1 m 网格、5 档俯仰、8 档偏航，`grid_env.neighbours` + GT 线段碰撞。

**能力边界**：
- 原地旋转和垂直移动**已经包含**在候选里；
- 一步跨多格、任意连续位姿在当前协议中不允许；
- Planning Region（室内 n_win = 5，每 10 步刷新）只影响模型的输入和评分范围，**不增加候选**。

**候选数**（实测）：Dev-3 平均 7.05（2–10），VALSEL 平均 7.80（5–10）。

## 3. 固定状态结果（Stage B，365 个开发态）
- Oracle best gain：Dev-3 1.740 pp，VALSEL 1.099 pp。所有模型、所有 λ 都相同。
- C0 的 selected gain：B2 1.275 / 0.882，E2 1.214 / 0.921，V4 1.110 / 0.570，V5 1.145 / 0.486（Dev-3 / VALSEL）。
- C1（λ 0.1）的 Δ selected gain：

  | 模型 | Dev-3 | VALSEL |
  |---|---|---|
  | B2 | −0.012（CI 不含 0） | +0.029（CI 下界 0） |
  | E2 | +0.011 | 0.000 |
  | V4 | +0.008 | −0.151（CI 不含 0） |
  | V5 | −0.006 | −0.025 |

  λ 0.3 更差或持平。完整表格见 `RESULTS.md`，图见 `figures/summary_fixed_pilot.png`（左）。

## 4. 闭环结果
**Stage A 回归**：新入口 λ 0 的 B2 Outdoor barts s0（100 步）与历史 C0 的选择和曲线完全相同。

**Stage C pilot**（配对、开发场景、λ 0.1；AUC / Final / Cov@20）：

| 模型 · 域 | C0 | C1 | ΔAUC | ΔFinal |
|---|---|---|---|---|
| B2 · MP3D VALSEL（4 ep × 150） | 0.720 / 0.901 / 0.518 | 0.723 / 0.894 / 0.563 | +0.002 | −0.007 |
| B2 · Macarons++ Dev-3（6 ep × 100） | 0.745 / 0.912 / 0.570 | 0.726 / 0.877 / 0.580 | −0.019 | −0.035 |
| E2-D100 · MP3D VALSEL | 0.789 / 0.938 / 0.551 | 0.770 / 0.957 / 0.551 | −0.020 | +0.019 |
| E2-D100 · Macarons++ Dev-3 | 0.727 / 0.908 / 0.580 | 0.698 / 0.872 / 0.543 | −0.029 | −0.037 |

曲线见 `figures/pilot_coverage.png`。

**Stage D（完整 G3）未运行**：门槛不通过。因此没有完整 G3 的 AUC / Final / Cov@20。历史参考值（B2 MP3D 0.334 / 0.530，Macarons++ Dev-3 + Test-3 30 ep 0.730 / 0.906 等）未与 C1 比较。

## 5. C1 是否改善 Camera Selection
**没有。它改变了排名，但没有改善覆盖率。**
- 固定状态：λ 0.1 只改变 3–5% 状态的选择。Top-1 的变化在 ±0.02 以内，与单步收益一致：没有系统提升，部分组合显著下降（B2 Dev-3、V4 VALSEL）。
- 闭环：只有 4–5% 的步偏离 DistP1 的 argmax，但偏离一旦发生，轨迹就整体分叉（首个分歧步多在第 2–30 步），单 episode ΔAUC 在 −0.14 到 +0.06 之间。
- 4 个组合中 3 个的均值为负；唯一不为负的 B2 室内是 +0.002（2 胜 2 负）。
- **一致的副作用**：C1 在 4 个组合中访问的不同位置**都更少**（C0 → C1：61.0 → 59.3、56.0 → 55.0、63.3 → 60.0、56.5 → 53.8）。

## 6. C2 与动作预算
本阶段**没有改变动作预算**：C1 与 C0 的动作集合、步数、单步代价都相同，路径统计可以直接比较（§5）。

C2 没有运行：
- **室外 NOT_SUPPORTED**：MAGICIAN 环境只执行 `pose_space` 网格索引，碰撞和覆盖率都在该网格上定义。改网格就是改官方环境。
- **室内**：网格模拟器是我们自己的代码，渲染和碰撞都支持任意位姿。但改步长或角度档就是新动作协议，需要：
  - 在新协议下重跑全部 C0；
  - 为网格外候选新建 oracle；
  - 路径成本记账（不能把多步当成一步）。

  Phase 1 按预注册不做，因此**没有任何可比或不可比的 C2 结论**。

## 7. 提升或退化的原因（基于证据的分析）
1. **候选空间没有余量可挖**。单步动作已经穷举（≤ 10 个），C1 的 oracle 与 C0 完全相同。真正的差距在排序质量：Dev-3 上 B2 选中 1.275 pp，oracle 是 1.740 pp。这个差距来自未来深度和 DistP1 的预测质量，而不是候选生成。
2. **先验与 DistP1 信息重叠**。F 衡量“视野里有多少 FREE→UNKNOWN 边界”，DistP1 衡量“预测的新表面”，两者在大多数状态排序一致。所以 λ 0.1 只翻转少数选择，翻转的平均收益约为 0。
3. **近场偏置（定义本身的性质，预注册时未发现）**：
   - 候选位于未观测区域内时，射线一出发就进入 UNKNOWN，所有进入点落在同一粗格，F ≈ 1（开发态 t = 0 时普遍如此，见 `AUDIT.md` / Stage A）。
   - 因此 F 偏好“站在已知空间里向外看”的视角，抑制“走进未知区域”。
   - 这与 C1 访问位置更少、个别 episode 被困（例如 colosseum s1：B2 Final 0.879 → 0.764，E2 0.818 → 0.608，分别从第 4 / 第 2 步开始分叉）一致。
   - 按协议，pilot 后不再修改定义。
4. **局部场景的正面信号**：pLe4wQe7qrG（B2 +0.060）、barts s0（B2 +0.012）、E2 barts s1（+0.021），以及 B2 的早期覆盖（室内 Cov@20 +0.045，室外 +0.010）。但在 4–6 个 episode 的规模下，这些信号不足以抵消退化。

## 8. 运行时间、GPU 与失败记录
- **资源**：gpu115 交互分配的 4 × L40S。没有申请新作业（不需要），也没有影响他人作业。
- **Stage B**：先验 1,120 GPU-秒（4 卡并行约 5 分钟）。
- **Stage A + C**：40 分钟墙钟（19:56 → 20:36），8 个 worker，约 2.7 GPU-h；42 个作业全部 exit 0。
- **C1 额外开销**：0.05–0.07 s / 步。
- **失败与修复**：
  - 一次 zsh 下命令变量没有展开，导致首次先验任务没有启动，改用脚本文件后正常；
  - 分析脚本对室外 episode 字段名写错（`pick` → `action_key`），已修正，没有重跑任何实验；
  - 覆盖率 done 文件路径少了一层 `V2M/`，在覆盖率阶段开始前用原子替换修正。

  三者都不影响结果。

## 9. 文件位置
- **代码 / 配置**（git）：`experiments/exp_camera_pose/`
  - `protocol.md`、`configs/c1_lambda.json`、`configs/pilot_*.txt`
  - `scripts/{frontier, c1_patch, c1_in_planner, c1_out_planner, c1_in_sim, c1_out_sim, fixed_prior, analyze_fixed, analyze_closed}.py`、`run_queue.sh`
- **报告**：`reports/{AUDIT, RESULTS, FINAL_REPORT, fixed_state, pilot}.md` 以及对应 JSON、`figures/`。
- **运行产物**（不进 git）：`work_dirs/exp_camera_pose/`
  - `fixed/prior/` 先验；
  - `closed_loop/<arm>_<cond>/` 的 episode、svc、覆盖率和 `c1_steps*.jsonl`；
  - `logs/`、`claims/`。
- **权重**（只读，未改动）：`work_dirs/checkpoints/exp6_v2/{B2/v2mix_ema17500.pt, E2/e2d100_ema5000.pt}`、`exp6_v4/V4/v4_ema40000.pt`、`exp6_v5/V5/v5_ema12000.pt`。
- **Git**：740065d（代码与预注册），以及本报告的提交。
- **种子**：沿用历史种子键（室内 `nv_choice` / 室外 `fallback_choice` 和 Flow 噪声）。

## 10. 下一阶段建议
在当前动作协议下，Camera Pose 的选择侧**没有可挖的空间**：候选已经穷举，oracle 不变；C1 在固定状态和闭环上都没有收益。
- **建议转向 Depth / DAPS 方向**：提升未来深度和打分的排序质量。这正是 selected gain（1.275）与 oracle（1.740）之间差距的来源。
- **若坚持 Camera Pose**：唯一有理论空间的是改变动作协议（C2，室内）。这需要重新定义公平基线（同一新协议下的 C0、路径 / 动作成本记账、新 oracle），属于新的实验，不应在 Phase 1 中顺带完成。
- 多步前瞻（Beam Search）不在本阶段范围内。

## 附录（2026-11-02 22:40）：后续 C2 与附加比较
- **C2**（MP3D 细格 / 连续位姿，任务书 C2）：Stage A 的一步 oracle 上限只提高 1–8%（C2c +0.084 pp），未通过预注册门槛 → **NO-GO**。见 `../C2/FINAL_REPORT.md`。
- **附加比较**（用户要求，`addon.md`）：在同一 pilot episode 上比较 B2 / E2-D100 / V4 / V5 的 C0 与 C1，以及 MAGICIAN beam。
  - V4 / V5 的 C1 为探索性（预注册 λ* = 0）：VALSEL ΔAUC −0.046 / −0.015，Dev-3 +0.024 / −0.012。
  - 4 个模型的 C0 在 MP3D VALSEL 上都高于 MAGICIAN beam（ΔAUC +0.015 至 +0.084）；在 Macarons++ Dev-3 上与 MAGICIAN beam 互有胜负，Final 都略低。
  - 本报告的 NO-GO 结论不变。
