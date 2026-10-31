# E2 — MP3D 规划审计（Phase 0；2026-10-31；代码 `task0 mp3d_sim.py / mp3d_frame.py`、`task0/magician_comparison/{scripts/t12_sim.py, adapters/grid_env.py, adapters/v2nv_planner.py}`、`task0/scripts/t06_planner.py`、`task2/scripts/t2_common.py`）

## 1. 环境（冻结，T0.12 网格协议 = Task 2 / Task 3 / EXP6 / Experiment C 共用）
| 项 | 事实 |
|---|---|
| 位姿空间 | 相机网格：楼层框横向收缩 0.25 m、高度 floor + 0.5 m 至 ceiling − 0.5 m，每轴格数 = round(extent / 1.0 m)（`grid_env.grid_def`）；俯仰 5 档（−60…60°）、方位 8 档（45°）；位姿 = (l, w, h, θ, φ) |
| 邻居 / 合法性 | 官方 `get_neighboring_poses`：沿 l / w / h 之一 ±1 格或 ±1 俯仰 / 方位档（去重、裁剪、方位回绕）；移动合法 = 两相机中心线段不与 GT 网格相交（`t12_sim.legal_actions`，官方 layer-0 规则）；旋转总是合法；自身不是动作 |
| 选择 | `v2nv_planner.nv_choice`：候选集 = 合法集中未执行过的位姿（MAGICIAN prefer-unvisited；若全执行过则全部合法），argmax，并列最小索引，全 ≤ 0 → 种子随机 |
| 传感器 / 覆盖 | 冻结 MAGICIAN 传感器 456 × 256、60°、射程 70 运行单位 = 10 m；C_obs = 可见 GT 点落入历史 10% 采样点 eps（7.8–12.8 cm）邻域的比例；可见 = 从任一网格位姿可见 |
| 起点 / 预算 | 起点 = 起始全景中心吸附到最近无阻挡网格位置，俯仰 0、方位 0；C0 + 150 步 |

## 2. 规划器（B2 / T1 / C1 共用，仅评分不同）
| 项 | 事实 |
|---|---|
| 规划区域 | 刷新步 t_r = 10⌊t/10⌋ 以当时相机为中心的 n_win = 5 窗口区域（`t2_common.region_box_m`：半边 = 0.5(1 + 4 × 0.8) × 楼层高 × 0.99 = 2.1 × 高；楼层高 2.75–4.5 m → 区域边长 11.6–18.9 m，裁剪到楼层框），z = 楼层全高；P1 融合记录 0..t 重建于该区域；编码器输入 = 刷新时的粗网格；候选射线在区域出口截断 |
| 预测范围 | 模型只在区域网格内解码（UNKNOWN 采样段），区域外无预测；GT 深度可越出（Experiment C / E2 R1 诊断用）|
| 评分 | DistP1（E1 审计）；K = 4 均值 |
| 不存在 | 无 frontier、无路径规划、无多步前瞻、无重访惩罚（除 prefer-unvisited 一步规则）、无 beam |

## 3. Experiment C / A 的停滞证据（MP3D）
| 量 | 数值 |
|---|---|
| 停滞 episode（t ∈ [30,120] 内 ≥ 30 步覆盖增长 < 0.01）/ 10 | MAG-G 7、MAG-B 8、B0 3、B2 4、E2D 2、T0 7、T1 5、**C1 5、C2 1** |
| 任一时刻剩余可见表面在当前 n_win = 5 区域内的比例（C1 / C2 闭环均值）| **0.25 / 0.29** |
| 停滞窗口内：合法候选最优真实增益中位（pp）/ 剩余 / 区域内比例 / 被阻挡邻居优于合法集的步数比例 | C1：0.031 / 57.5% / 0.23 / 0.28；C2：0.038 / 50.4% / 0.35 / 0.42 |
| 单步 oracle greedy 的 Final | 0.615（B2 0.530，E2D 0.553）|
| Experiment A（固定状态，53 个停滞态）| 66% 有正收益合法候选但 G\* 中位 0.019 pp；oracle 最优候选 31% 的增益在区域外；未覆盖表面仅 27% 在区域内 |

## 4. 五类问题与本实验的对应诊断
| 类 | 判据（离线，GT 仅评估侧）| 实验 |
|---|---|---|
| A 有好候选但评分排错 | 合法集 G\* 大（> 0.1 pp）且 S0 / 模型选错 | E1（评分器）；E2 失败案例第 6 / 7 项 |
| B 区域外有收益、评分看不到 | 候选 oracle 增益在 n_win = 5 区域外的份额高；扩大评分区域后 GT 排序改善 | R1（`e2_r1_region.py`：n_win 5 / 7 / 9 / 全层的 GT 评分 regret 与区域内增益份额）|
| C 所有候选真实增益都低，但存在有价值前进方向 | G\* < 0.05 pp 且剩余表面大、可达 frontier 存在 | R2（frontier 进展；无 GT）|
| D 合法动作集无法进入未探索区 | 两步可达扩展候选的 G\* 显著高于原合法集；frontier 不可达 | R3（候选空间 counterfactual）|
| E 候选足够但 greedy 长期不理想 | C2（单步 oracle）仍停滞 / Final 低 | R2 触发模式 + R4 |

## 5. R2 的 2D 可通行 / frontier 图（设计；无 GT）
- 栅格 = 相机网格的 (l, h) 平面（1 m），高度层 w 合并；来源只有真实记录：相机所在格 = FREE；每条记录中世界方向接近水平（|dir_y| < sin 10°）的像素射线从相机格沿 2D 投影向前雕刻：深度 < 70 的射线把到命中点前的格标 FREE、命中格标 OCCUPIED（累计计数，占多数者胜；FREE 证据 ≥ 2 条射线才生效）；无命中射线雕刻到射程；其余格 UNKNOWN。
- frontier = UNKNOWN 且 4 邻接 FREE 的格；可达性 = 从当前格经 FREE 格的 4 连通 BFS；候选相机的格若合法则视为 FREE（合法性由模拟器的 GT 线段检查给出，是环境规则，不是 GT 泄漏）。
- FrontierProgress(C) = dist(当前格 → 最近可达 frontier) − dist(候选格 → 最近可达 frontier)；旋转 = 0；无可达 frontier → 0；已访问格 FP × 0.5。
- 模式：bonus S = Gain + λ·FP；trigger：max Gain < θ 时按 FP 排序（并列按 Gain）。λ、θ 由 dev 楼层（VALSEL）上的模型分数分布预先定（协议 §4 R2）。
