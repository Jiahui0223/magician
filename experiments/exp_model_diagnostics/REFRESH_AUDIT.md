# REFRESH_AUDIT —— 时间刷新（Temporal Refresh）审计（2026-11-02，只读）

**标注**：
- **【代码】** 由真实执行路径的代码确认（路径:行号）。
- **【日志】** 由历史闭环日志实测确认（`refresh_log_evidence.json`）。
- **【报告】** 来自历史报告。
- **【假设】** 尚未验证。

路径缩写：
- `T0` = `experiments/exp5_future_view/task0/scripts`
- `T1` = `…/task1/scripts`
- `T2` = `…/task2/scripts`
- `T3` = `…/task3_full_depth_p1/scripts`

## 0. 结论
1. **所有模型（B2、E2-D100、V3-T0、V3-T1、V3.1、V4、V4C、V5）在闭环中都走同一套冻结的规划器循环**：
   - 室内是 `T0/t06_planner.py`（由 `T2/t23_v2_in_planner.py` 驱动）；
   - 室外是 `T1/v2_planner.py`（由 `T2/t24_v2_out_planner.py` 驱动）。
   - 各模型的包装只改了三处：P1 类（`p1_full.patch`）、模型 Core（`planner_v3.install`、`v5_core.install` 或换 checkpoint）、V3 的 `Geometry.coarse`。**没有任何包装改动刷新逻辑。**【代码】
2. **R10 在两处分别硬编码，没有环境变量或参数可以修改**：
   - 室内：`T0/t06_planner.py:112` 写死 `if t % 10 == 0:`。
   - 室外：`T1/v2_planner.py:194` 判断 `LC.refresh_time(a.exp, t) == t`，而 `T1/v2_common.py:49-51` 中 `refresh_time` 返回 `10 * (t // 10)`（arm 的 refresh 为 "native" 时每步刷新）。V2O / V2M 的 arm 在 `T2/t24_v2_out_planner.py:25` 写死为 `"refresh": "R10"`。
   - `task1/configs/closed_loop_v1.json` 中的 `refresh_R10` 列表**不被任何规划器读取**，只用于 `v2_run.py:189` 的 smoke 检查。【代码】
3. **日志实测**（`refresh_log_evidence.json`，全部 8 个模型）：
   - 室内每个 150 步 episode 刷新 15 次，全部在 t % 10 == 0（各 10 个 episode）。
   - 室外每个 100 步 episode 刷新 10 次，全部在 t % 10 == 0（各 30 个 episode）。
   - 室外 3000 / 3000 步都有 P1 更新和几何构建。【日志】

## 1. 各阶段的刷新频率（B2 / E2-D100 / V3 / V3.1 / V4 / V4C / V5 完全相同）
| 阶段 | 室内 MP3D（150 步） | 室外 Macarons++（100 步） | 证据 |
|---|---|---|---|
| **P1 融合**（b2 模式：D100 输入图 + S10 评分图双胞胎） | **每步**：非刷新步 `self.fu.integrate_record(rec)`（t06:115）。刷新步在新区域内**重建**融合并重放第 0..t 条记录（t06:85-87）。`P1FusionB2.integrate_record` 同时更新两张图（T3/p1_full.py:56-58） | **每步**：`fu.integrate_record(rec)`（v2_planner:186）。融合对象每个 episode 只建一次（:112-113），整场景网格 | 【代码】【日志】室外 p1_s > 0 共 3000 / 3000 步 |
| **几何 / SceneCaster / band_dil（S10）** | **每步**：`Geometry.from_fusion`（t06:127）。刷新步在 `refresh()` 里为编码器**再建一次**（t06:100）。b2 模式下 band_dil 换成 S10（p1_full:71-81） | **每步**（v2_planner:190） | 【代码】【日志】 |
| **候选视图 / 射线投影统计（pr）** | 每步：`score_state` 内的 `geo.view`（T1/v2_plan_core.py:74） | 每步 | 【代码】 |
| **3D 编码器 → h** | **每 10 步**（t06:112-113），输入为刷新时（第 0..t 条记录）的 D100 coarse 网格（t06:101） | **每 10 步**（v2_planner:194-195、149-150），输入为第 0..t 条记录的 D100 | 【代码】【日志】 |
| **cond**（射线特征；V3-T1 的 SceneMemory、V4 的 RayXAttn、V5 的 Memory + RayDecoder） | 每步，由缓存的 h 和当前候选射线计算（v2_plan_core:91） | 每步 | 【代码】 |
| **室内规划区域**（n_win = 5，以当前相机为中心） | **与编码器刷新耦合**，在同一个 `refresh()` 里：`F.region_geometry(card, recs[-1]["cam_center"])`（t06:83-84）；n_win 补丁见 T2/t23_v2_in_planner.py:25。两次刷新之间，新记录写入上次刷新的网格，网格外的点丢弃（m_p1.py:45-51） | 无（整个场景一个网格） | 【代码】【日志】室内非首次刷新中区域变化 137–139 / 140 |
| **Flow 噪声 eps**（K = 4） | **每次刷新重新生成**：t06:102，种子键 `exp5-V2|task0cl|{level}_S{start}|t{t_r}|k{k}`。不含模型名，所以同一 episode、同一 t_r 下所有模型噪声相同 | 每次刷新重新生成：v2_planner:151，键 `exp5-V2|{scene}|s{start}|rep{rep}|t{t_r}|k{k}`（closed_loop_v1.json；v2_common:54-55） | 【代码】 |
| **显式清缓存** | 只有 `p1_full.py:79-80` 的 `torch.cuda.empty_cache()`：每次 `from_fusion` 都执行，刷新步执行两次。h / eps / fu 只是被重新赋值，没有其他清理 | 同上，另有每步 `del geo`（v2_planner:230） | 【代码】 |

## 2. 各模型在刷新时缓存的内容（只在刷新时构建）
| 模型 | 缓存的 h | 每步由 h 重新计算的部分 | 证据 |
|---|---|---|---|
| B2 / E2-D100（V2） | (F0, F1, dom) | 射线特征（32 点采样 + Conv1D）、cond | seen2scene/futureview/model.py:292-295 |
| V3-T0 / V3.1 | {mu, logvar, valid8, band8, free8, ext}（VAE 潜变量 + 刷新时的 band / free） | cond（无 memory） | seen2scene/v3/model.py:60-66 |
| V3-T1 | 同上 | SceneMemory tokens + cond（v3/model.py:88-94） | 【代码】 |
| V4 | (F0, F1, dom) | V2 射线 + RayXAttn 残差（model.py:297-319） | 【代码】 |
| V5 | (F0, F1, dom, coarse_in) | Memory(F0, F1, cin) + V5RayDecoder（v5_model.py:204-213、233-244） | 【代码】 |

V3 的 h 里还包含刷新时的 `band8` / `free8`，V5 的 Memory 里有刷新时的 `obs0` / `obs1`。**它们与 3D 特征一样，最多可能落后 9 步**；而每步的射线投影统计（11 个 proj 通道）始终是当前 P1。【代码】

## 3. 各模型的执行路径
- **B2 / E2-D100**：
  - 室内：`t33_in_sim.py` → `t32_in_planner.py`（`PF.patch()` + T3_CKPT）→ `t23_v2_in_planner.py` → `t06_planner.Planner`；选择用 `v2nv_planner.nv_choice`。
  - 室外：`t33_out_sim.py` → `t32_out_planner.py` → `t24_v2_out_planner.py` → `v2_planner.main()`。
  - E2-S10 对照与此相同，只是 p1 = sparse。
- **V3-T0 / T1**：`exp6_v3/shared/sim_{in,out}_v3.py`（`T3_P1_MODE` 写死为 b2）→ `v3_{in,out}_planner.py`（`PF.patch()` → `planner_v3.install()`，其中 Core 为 V3Core，`Geometry.coarse` 为 `pack_grid(aggregate_fusion(fusion))`，读取 D100 图）→ P23 / P24。
- **V3.1**：没有自己的规划器，由 `exp6_v4/scripts/run_queue_v4.sh:15` 通过 `sim_in_v3.py --v3-ckpt work_dirs/exp6_v3_1/checkpoints/final.pt` 运行（exp6_v3.1/README.md:14）。
- **V4**：`exp6_v4/scripts/sim_{in,out}_v4.py`（`V4_P1_MODE` 默认 b2；`full` 只用于 Experiment A）→ `v4_{in,out}_planner.py`（`ckpt_override` + `PF.patch()`）→ P23 / P24。
- **V5**：`exp6_v5/scripts/sim_{in,out}_v5.py` → `v5_{in,out}_planner.py`（先 `v5_core.install()`，只换 Core 为 V5Model，并给 encode / cond / sample 加 grid_sample 守卫）→ 同上。

【代码】以上均由子审计逐行追踪，关键行已人工抽查（t06:80-116、v2_common:46-56、v2_planner:190-196、t24:25-29、p1_full:52-58）。

## 4. 固定状态评估器（G1 / G2）是否复现同一刷新规则
- **规则相同**：`exp6_v3/shared/eval_fixed.py`（V4、V3.1、V5 共用）、`e2_pilot/scripts/e2_fixed.py`、`density_ablation/scripts/e12_fixed.py` 都按 `blocks = {t − t % 10}` 回放：
  - 室内：在每个块 r 用 `recs[r]["cam_center"]` 重建区域（n_win = 5），新建融合并写入第 0..r 条记录，然后刷新，再增量写入 r+1..t。
  - 室外：一个场景网格，每步写入，t ∈ blocks 时刷新。【代码】
- **实现上的差异**（不是刷新规则的差异）：
  - 固定状态评估分两遍跑：先用 D100 图做预测，再用 D10 图，以 S10 的 band_dil 给 DistP1 打分；闭环则是双胞胎融合。
  - VALSEL 的噪声键是 `exp5-T3E2|valsel|…`，与闭环键不同（VALSEL 是独立轨迹，没有对应的闭环）；Dev-3 和 e12 用闭环键。【代码】
- **逐位复现记录**：
  - e12 B2 层 `I100_eq_online_B2`：室内 84 行、室外 247 行全为 True；B0 层：室内 100、室外 300 全为 True。
  - eval_fixed `repro_B2_eq_E2`：VALSEL 90、Dev-3 150 + 125 全为 True。【报告 / 行数据】

## 5. 训练端的“陈旧编码”分布（对 R5 / R2 / R1 很关键）
- **【代码】** V2-Mix 的数据构造按 R10 模拟陈旧编码（T2/t21_build.py:33；t22_data.py:29、104-128）。配置项 `stale_p 0.5`、`stale_max 9`（train_mix_v1.json、e2_d100_v1.json），含义是：训练时一半样本使用最多落后 9 步的编码。
  - 由此，**R1–R10 的陈旧程度（0–9 步）都在训练分布之内**：R1 对应 0 步，R10 对应 0–9 步。
  - V3 / V3.1 / V4 / V5 的训练器是否同样模拟了陈旧编码，见 TRAINING_COMPARISON.md。
- **【假设】** 更频繁的刷新（R5 / R2 / R1）会让 3D 特征与当前 P1 保持一致，可能改善 UNKNOWN 区域的预测和排序。但它也会改变噪声刷新频率；室内还会改变规划区域的重新居中频率。

## 6. 对 A/B 实验的含义（设计要点，详见 ABLATION_PLAN.md）
1. **改刷新周期需要改代码**：两处常量（t06:112、v2_common:51 / t24:25）需要加一个可配置参数（例如环境变量 `CP_REFRESH`），默认值 10，以保证 R10 与历史逐位一致。**不需要重新训练**。
2. **当前刷新同时触发三件事**：重算 3D 编码、重新生成噪声、室内重建区域（及融合）。要分清是谁起作用，必须解耦：
   - **编码刷新周期**：R10 / R5 / R2 / R1。
   - **噪声策略**：按刷新重新生成（现状），或固定按 R10 的 t_r 生成（使 R1 与 R10 用同样的噪声）。
   - **室内区域**：随刷新重建（现状），或固定按 R10 节奏重建、只加快编码器刷新。
3. **成本**：R1 每步多一次编码（日志中室外编码约 0.18 s / 次，室内刷新约 0.5 s / 次），刷新步还要多做一次几何构建。B2 室外决策时间约 1.5–3 s / 步，预计 R1 增加不超过 20%。【假设：未实测】
