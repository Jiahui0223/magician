# SCORING_AUDIT — V3-T1 冻结规划路径的逐行审计（H0 的真实定义）

## 1. 调用链（Table 1 V3 行）
- **启动**：`exp6_v3/shared/sim_out_v3.py` 启动规划器子进程 `v3_out_planner.py`。
  - `p1_full.patch()` 设为 `T3_P1_MODE=b2`：`m_p1.P1Fusion` 换成 P1FusionB2，模型输入用 D100，`band_dil` 换成稀疏 S10 孪生地图的；
  - `planner_v3.install()`：`PC.Core` 换成 V3Core，`Geometry.coarse` 返回 D100 融合的部分 TSDF 4× 网格；
  - 然后进入 `task2/t24_v2_out_planner.main` → `task1/v2_planner.main`。
- **每步流程**（`v2_planner.py`）：
  1. `fu.integrate_record(rec)`（:186）；
  2. `Geometry.from_fusion`（:190）；
  3. 在 R10 时刻 `refresh`（:194），编码当前 P1 的特征 h，噪声由 `PR.noise(dev, K, sha_seed(seed_key(s, k, rep, t, kk)))` 生成（:151）；
  4. `PC.score_state(core, geo, h, legal_R, legal_T, eps, z0)`（:202）；
  5. `scores = out["distp1"]`；`MS.choose(scores, s, k, t, True)`（:205–207）。
- **选择规则**：`m_score.choose`（`exp3b/m_score.py:129-135`）：全零时走带种子的回退 `MC.fallback_choice`，否则取 `MC.argmax_lowest`（并列取最小索引）。
- **候选**：模拟器的合法列表（`m_magician_replay.legal_neighbours:84-102`）：
  - `get_neighboring_poses`：6 个平移 + 4 个旋转，边界截断，方位角取模，去重（`MAGICIAN macarons_utils.py:3922-3970`）；
  - `get_valid_neighbors`：未访问优先；mesh 有效性检查在源码中被注释掉了（:3972-4000）；
  - GT 线段碰撞测试。
  - 规划器只拿到通过碰撞测试的结果：`legal_idx`、`legal_R`、`legal_T`、`legal_center`。

## 2. 打分与 K 聚合（`task1/scripts/v2_plan_core.py:68-126`，`seen2scene/futureview/scoring.py:14-41`）
1. **视角投影**：`views = [geo.view(R, T)]`（:74）——对当前 P1 投射 64 × 114 条射线，并计算投影通道。
2. **条件**：`cond, m_unk = core.pred.cond(h, views)`（:91）。
3. **采样**：`b, y = core.pred.sample(cond, m_unk, eps)`（:94）。K = 4 个样本共用同一组噪声 eps（所有候选也共用），Euler 10 步。
4. **逐视角解码**：`t_dec, hit_u = G.decode(b[m], y[m], ...)`（:100）。已知射线取投射结果，UNKNOWN 射线取预测的首个交点，无命中为 inf，不会伪造深度。
5. **逐样本打分**：`SCO.new_set_from_depth(t_dec[k], o, d, dz, k_M, dims, domain, band_dil, z0)`（:103-105；scoring.py:28-41）：
   - 命中点 p = o + d·t，落到体素 floor(p / 0.011)，并截断到网格内；
   - 只保留落在 `domain` 内、且不在 `band_dil` 中的体素；`band_dil` 是 S10 观测带做 3×3×3 最大池化，也就是 ±1 个 P1 体素（`m_score.py:39`）；
   - 体素去重；
   - 每个体素的权重取 max min(1, (z / z0)²)，其中 z = t·k_M·dz。
6. **K 聚合**：`ws.append(wv.sum())`，`out["distp1"].append(np.mean(ws))`（:105-108）。
   - 也就是说，H0 已经是“K 个样本分数的均值”。
   - 所以 H2 不能再叫“取均值”，它的新意只在 −λ·std 这一项。

## 3. 尺度：H0 的“已见”半径与官方覆盖容差（H1 的依据）
- **官方覆盖容差**：eps = 2 × TEST_RESOLUTION（0.05）× scene_scale_factor（10）= **1.0 运行时单位**（`m_t6_sim.py:72`）。
- **P1 体素**：1 个 P1 体素 = 0.011 模型单位 = 0.011 · k_M 运行时单位（`m_common.SceneFrame`：模型单位 = 运行时单位 / k_M）。

| 场景 | k_M | eps 折合的 P1 体素数 |
|---|---|---|
| eiffel | 33.66 | 2.70 |
| neuschwanstein | 29.64 | 3.07 |
| liberty | 28.71 | 3.17 |
| bannerman | 21.68 | 4.19 |
| pantheon | 17.01 | 5.34 |
| redeemer | 16.82 | 5.40 |
| alhambra | 16.46 | 5.52 |
| fushimi | 15.90 | 5.72 |
| pisa | 13.99 | 6.50 |
| bridge | 12.47 | 7.29 |
| colosseum | 11.92 | 7.63 |
| barts | 11.56 | 7.86 |
| dunnottar | 9.67 | 9.40 |
| sestino_museum | 4.43 | 20.54 |

- **H0 的偏差**：H0 只排除 ±1 个体素以内的点，所以与已观测表面相距 1 到 eps 之间的预测命中也会被计为新的，尽管它们对官方覆盖没有贡献。
- **H0 的重复计数**：H0 按 0.011 的体素去重，同一个覆盖单元（边长 eps）可能被计数 (eps / 0.011)² 量级的次数。
- **H1 的修正**：
  - 用智能体自己的 S10 观测点（服务记录 `sample_points`，已执行帧 0..t，与官方覆盖评价使用的是同一批观测点）判断“与已观测相距 > eps”；
  - 在边长 eps 的运行时格子上去重；
  - w(z) 与 z0 不变，也不另乘面积系数。

## 4. 随机性
- **噪声**：H0–H3 都使用冻结的噪声键 `seed_key(scene, start, rep, t_r, k)`，只在 R10 刷新时生成，一步内所有候选共用（因此 H3 的第二层视角也用同一组噪声）。H1–H3 没有新增任何随机源。
- **KD 树**：H1–H3 的 KD 树查询（scipy cKDTree）是确定性的。
- **回退**：走带种子的回退时与 H0 规则相同。

## 5. 泄漏审计
- **规划器读到的信息**：H1–H3 只读取规划器本来就能看到的信息，包括服务记录白名单中的 `sample_points`、`executed_pose_idx_history`、`legal_idx` 和相机，当前 P1，以及冻结配置与位姿表。
- **防护**：配置与位姿表在 LeakGuard 安装之前读取。位姿表只来自 settings.json，不含 mesh 或 occupied_pose。
- **H3 的第二层**：不做 GT 碰撞测试，用已知 D100 表面带做阻挡判定；不把虚拟观测写回 P1。
- **oracle**：oracle 增益只在 G2 分析时、所有打分完成之后读取。
