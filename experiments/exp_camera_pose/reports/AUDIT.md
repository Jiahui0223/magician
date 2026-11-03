# Camera Pose Phase 1 —— Task 0 审计：现有相机候选机制（2026-11-02，只读代码审计）

所有结论来自真实代码（路径与行号如下），没有假设模拟器支持连续运动。下文缩写：
- `MU` = `MAGICIAN/macarons/utility/macarons_utils.py`（运行时导入的 fork）
- `EXP5` = `experiments/exp5_future_view`
- `MCS` = `EXP5/task0/magician_comparison`

## 1. 三个概念的区分（本审计的核心）
| 概念 | 在代码中的实体 | 范围 |
|---|---|---|
| **Candidate Pose**（下一步可执行位姿） | 每步的 `legal_R / legal_T / legal_idx`（服务记录 svc） | 当前位姿的单步邻居：**±1 格只动一个位置轴，或者 ±1 档只动俯仰 / 方位**，二者不同时发生；最多 10 个 |
| **Planning Region**（模型预测 / 评分的空间） | 室内：`t2_common.region_box_m`，n_win = 5，每 10 步以相机为中心刷新（`EXP5/task2/protocol.md:44,72`；`t21_build.region`）；室外：整个场景框 | 只决定 3D 编码器的输入网格和 DistP1 计数的空间，**不增加任何候选** |
| **Motion Constraint**（能否真的走到该位姿） | 室内：`MCS/scripts/t12_sim.py:legal_actions` + `Blocker`（GT 网格线段求交）；室外：`EXP3b/m_magician_replay.legal_neighbours`（GT 线段求交） | 位置移动必须线段不穿过 GT 网格；原地旋转总是合法；越界的位移被截断（`clamp`）并去重 |

扩大 n_win 或 Planning Region 只会改变“看多远、评多大”，不会改变“下一步能去哪”。

## 2. 室外 Macarons++（M 协议，`EXP5/task1/scripts/v2_sim.py` → `EXP3b/m_t6_sim.py`）
**邻居生成**：`MU:4178 get_neighboring_poses`（fork 版本，注释写明 “Each step can EITHER move OR rotate, not both”）。
- 位姿索引为 5 维 `[l, w, h, θ, az]`。
- 6 个纯位移（x/y/z ±1 格）和 4 个纯旋转（俯仰 ±1、方位 ±1）。
- 越界时截断到边界，方位按 `% n_azim` 取模，然后 `torch.unique`。
- 结果最多 10 个；在边界上可能更少，也可能包含当前位姿（截断造成）。

**有效性**：`MU:4228 get_valid_neighbors`。
- 只保留**未访问**的邻居；如果全部访问过，则返回已访问的邻居。
- 碰撞检测在这一步被注释掉（`# is_valid = ...`）。

**碰撞**：`EXP3b/m_magician_replay.py:84 legal_neighbours`。从当前相机中心到候选中心做 GT 网格线段求交，相交的候选被剔除。

**死胡同**：没有合法候选时原地悬停（`m_t6_sim.move(key=None)`）。

**网格**：来自各场景 `settings.json` 的 `camera` 块，单位为 OBJ 单位 × 10 = 运行时单位，非公制。

| 场景 | pose_l × w × h | 位置框（OBJ 单位） | n_azim |
|---|---|---|---|
| alhambra | 10 × 6 × 11 | x [−6, 3]、y [1, 4]、z [−8, 2] | 10 |
| barts | 7 × 9 × 5 | x [−1.02, 1.79]、y [−0.21, 2.88]、z [−1.30, 1.12] | 6 |
| pantheon | 10 × 6 × 13 | x [−4.5, 4.5]、y [0, 6]、z [−7.5, 5] | 10 |
| bridge | 15 × 5 × 8 | x [−10, 10]、y [0, 4]、z [−3, 3] | 10 |

- 格距大致为 1 个 OBJ 单位（= 10 个运行时单位）。
- 俯仰按 `pose_n_theta` 分档，`elev = −90 + 180(1+i)/(nθ+1)`。
- 位姿只能是 `camera.pose_space` 字典中的网格点；执行接口为 `update_camera(idx)`。

**选择规则**：`EXP5/task1/scripts/v2_planner.py:202-213`。
- `PC.score_state` 得到 `out["distp1"]`（K = 4 个样本的 DistP1 均值），交给 `m_score.choose`：取 argmax，并列时取最小下标。
- 若全部为 0，用带种子的 fallback（`EXP3b/m_score.py:129`）。
- “优先未访问”已在环境的合法列表中完成。

## 3. 室内 MP3D（`EXP5/task2/scripts/t23_in_sim.py` → `MCS/scripts/t12_sim.py`，`MCS/adapters/grid_env.py`）
**网格**：`grid_env.py:17-31`。
- `STEP_M = 1.0`，即公制 1 m 格（运行时单位 = 7 × 米）。
- 楼层框内缩 0.25 m，地面 / 天花板各留 0.5 m 余量。
- `N_THETA = 5` 档俯仰，`N_AZIM = 8` 档偏航（45°）。

**邻居**：`grid_env.neighbours`（与官方规则相同）。
- 5 个轴各 ±1，截断并去重，环境侧去掉自身。
- 合计最多 10 个：6 个位移 + 2 个俯仰 + 2 个偏航。

**合法性**：`t12_sim.legal_actions`。
- 位移候选做 GT 线段求交（`Blocker`，trimesh）；与当前位置相同的候选（纯旋转）直接合法。
- 起点吸附到网格。每局 150 步，151 帧。

**选择规则**：`MCS/adapters/v2nv_planner.py:20 nv_choice`。
- 只在**本局未执行过**的合法候选中取 DistP1 最大者；若全部执行过，则在全部候选中取。
- 若最大值 ≤ 0，用带种子的随机 fallback。
- DistP1 来自 `MCS`/`EXP5/task0/scripts/t06_planner.py:128` 中的 `PC.score_state`。

## 4. 每状态的实际候选数（固定状态集，取自 manifest 的 `n_legal` / `cand`）
- 历史固定状态：MP3D VALSEL 90 态、Macarons++ Dev-3 275 态。
- 每个状态的合法候选数都 ≤ 10。
- 实测分布（候选数均值 / 范围）由 Stage B 的固定状态脚本统计，写入 `reports/RESULTS.md`。
- 第一轮 smoke 中：Dev-3 barts_s0 共 10 态 / 73 个候选（平均 7.3）；VALSEL 8194nk5LbLH 共 10 态 / 79 个候选（平均 7.9）。

## 5. 支持哪些扩展
| 问题 | 结论 |
|---|---|
| 原地旋转 | **已经包含**在候选中（俯仰 / 方位 ±1 档），两个域都是 |
| 垂直移动 | **已经包含**：室外 y ±1 格；室内楼层内 z ±1 m |
| 不同步长 / 一步跨多格 | **当前协议不支持**。单步就是 ±1 格；跨多格等于把多个标准动作当成一步，不允许 |
| 任意连续位姿 | **室外不支持**：位姿只能是 `pose_space` 网格点，`update_camera(idx)` 只接受网格索引；覆盖率协议也按网格执行。**室内只能作为新的动作协议**：t12 网格模拟器是我们自己的代码，`Sensor` 可以渲染任意位姿，`Blocker` 可以检测任意线段，但改格距或角度档就是新协议，需要在同一新协议下重跑 C0 |
| 用 P1 的 FREE / BAND / UNKNOWN 提取 frontier | **支持，且不需要 GT**。规划几何 `v2_plan_core.Geometry` 带细网格 struct（EMPTY / UNKNOWN / BAND，来自 D100 融合）；每个候选视图的射线统计 `view["pr"]` 给出 `cls`（UNKNOWN / KNOWN / EMPTY）、`d_lo`（第一个域内 UNKNOWN 采样点距离）、`t_known`（已知 TSDF 交叉）。B2 模式下这些统计用 D100（`p1_full._patch_geometry_b2` 只把新颖性的 band_dil 换回 S10） |
| 碰撞检测 | 室内 / 室外都有，GT 线段求交，**属于模拟器 / 运动约束侧**。规划器不读取 GT |
| 任意位姿的深度渲染 | 室内可以（`t12 Sensor`，模拟器侧）；室外只能通过 MAGICIAN 相机渲染网格位姿 |
| n_win = 5 与动作范围 | n_win 是室内 Planning Region（规划区域，5 m 窗口，每 10 步刷新）；动作范围永远是 ±1 格 / ±1 档。两者无关 |

## 6. 对本阶段的影响
1. **两个域都属于任务书的“情况 1”**：官方邻居已经枚举了全部合法单步动作。所以 **C1 只能在同一动作集合内重新排序**：`Score_C1 = norm(DistP1) + λ·norm(FrontierPrior)`。它是 Geometry-Aware Pose Selection / Score Prior，**不是候选集扩展**。
   - C1 与 C0 的 Oracle Best Gain 必须完全相同（同一集合），Stage B 会检查。
   - 不合法动作率恒为 0：候选集合不变，执行仍走模拟器原来的合法列表。
2. **C2（多尺度 / 由粗到细）**：
   - **室外 NOT_SUPPORTED**：MAGICIAN 环境的位姿空间、碰撞和覆盖率协议都绑定在固定网格上，改变它就是修改官方环境。
   - **室内只能作为新动作协议**。这需要：
     - 新网格（例如 0.5 m、16 档偏航）下的 C0 重跑；
     - 为网格外候选新建 oracle（现有 oracle 只覆盖合法网格候选）；
     - 路径长度 / 动作成本记账。
   - 按任务书优先级（C1 优先），本阶段**先完成 C1**。C2 的处理和原因写进 protocol.md 与最终报告。
3. Frontier 先验只读取规划器自己的 D100 P1 地图（与 DistP1 / 模型输入同源），不读取 GT 网格或未来真实深度。GT 只用于评估端的 oracle 收益和覆盖率。
