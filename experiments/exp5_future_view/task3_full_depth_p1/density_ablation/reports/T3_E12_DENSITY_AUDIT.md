# Task 3 / E1.2 — 审计与运行计划（2026-10-28，运行前）

## 1. 现有代码与数据（已核实）
| 项目 | 事实 | 出处 |
|---|---|---|
| 原始 10% 采样 | 每帧记录保存 `sample_pixel_flat`（像素索引，全部是有效像素、无重复、乱序）和 `sample_points`。数量 = floor(0.1 × 有效像素数)。有效像素 = mask ∧ clip(z, 0.5, 750) < 70，与 `p1_full.full_points` 的定义一致 | svc 记录（6 帧抽查） |
| B0 / B1 / B2 的 P1 | B0 = 冻结 `m_p1.P1Fusion`（只用 10% 点写 BAND）；B1 = `P1FusionFull`（10% 点原样 + 其余有效像素按升序反投影）；B2 = B1 作输入 + B0 的 `band_dil` 评分 | `scripts/p1_full.py` |
| 融合 | BAND 累加使用定点整数（int64 sum + int32 cnt），结果与点的顺序无关；EMPTY 挖空始终使用完整深度，与密度无关 | `m_p1` |
| `t_known` | **不是首个 BAND 命中**，而是"仅观测"TSDF（BAND 体素保留 SDF，其余体素 = +TRUNC）沿射线三线性插值后的第一个 + → − 零交叉，步长 0.5 体素 | `m_score.SceneCaster.cast`、`v2_plan_core.Geometry.view` |
| d_lo / d_hi / n_seg | 在 `t_known` 之前的 UNKNOWN 采样（最近体素），由 structure 判定：BAND 优先于 EMPTY | `seen2scene/futureview/geometry.py` |
| DistP1 新颖性 | 预测命中点所在体素不在 `band_dil` 中（`band_dil` = BAND（\|v\| < 0.999 TRUNC）∩ domain，再做 3×3×3 膨胀）；分数 = 对唯一新体素按 min(1, (z/z0)²) 加权求和 | `seen2scene/futureview/scoring.py` |
| 现有伪表面定义 | err = z_known − z_gt；容差 tol = TRUNC·k_M + 0.01·z_gt；伪表面 = err < −tol；**分母 = 有已知表面且有 GT 的射线**（不是全部 GT 射线） | `scripts/t31_geometry.py: ray_block` |
| 覆盖容差 | MP3D：eps = 0.08 模型单位（约 7.3 个体素）；Macarons++：sim.eps（存在 oracle.npz 中）。都用 10% 采样点判定 GT 点"已覆盖" | `mp3d_sim.Coverage`、`s1_oracle_out` |
| 固定状态 | Task 3 的 400 个状态（MP3D 100、Macarons++ 300），**全部来自 Task 2 V2M 轨迹 = B0 轨迹**（t = 0 与其他组合并）。每个状态有：候选集 C（MP3D 平均 7.0，Macarons++ 7.1）、oracle 增益、t*（候选 GT 深度） | ablation manifest、`oracle.npz` |
| B2 轨迹状态 | B2 闭环保存了完整的 svc 记录（MP3D 10 条、Macarons++ 30 条）和 planner 日志，可以按同一规则重建状态。oracle 便宜：现有 V2M 组 MP3D 共 3 min，Macarons++ 共 55 min | `W3/closed_loop/*/B2` |
| 噪声 | 室内用 `exp5-V2\|task0cl\|<tag>\|t<r>\|k<k>`，室外用 `LC.seed_key(scene, start, 0, r, k)`，r 为最近一次刷新步。与轨迹无关，B2 闭环使用同一规则 | `s2_infer.StateReplay.seed_key` |

## 2. 实际修改范围（全部为新增文件，不改冻结代码）
| 文件 | 作用 |
|---|---|
| `density_ablation/scripts/e12_density.py` | 嵌套密度采样 `density_points(rec, x)` 与 `P1FusionDensity(x)` |
| `density_ablation/scripts/e12_states_b2.py` | 用 ablation 的 `s0_states` 生成 B2 轨迹状态清单（输出到 W3，不写 ablation 目录） |
| `density_ablation/scripts/e12_oracle_b2.py` | 用 ablation 的 `s1_oracle_in/out` 计算 B2 轨迹状态的 oracle（输出到 W3） |
| `density_ablation/scripts/e12_fixed.py` | 固定状态主程序（四档密度依次运行，同一时间 GPU 上只有一张地图） |
| `density_ablation/scripts/e12_analyze.py`、`e12_viz.py` | 分析、Figure 1–6 |
| `p1_full.py` 的新模式（只在需要闭环时） | `d50` / `d75`：Dx 输入 + 稀疏评分，与 B2 同一实现方式 |

**嵌套采样（10% ⊂ 50% ⊂ 75% ⊂ 100%）**：每帧
- rest = 有效像素 − 原始 10% 像素；
- 种子 = sha256(R, T, `sample_pixel_flat` 的字节)，只由观测内容决定，因此固定状态和闭环中同一帧的采样完全相同；
- 用 `default_rng(seed)` 生成 rest 的一个排列，Dx 取前 floor(x · n_valid) − n10 个，再按像素索引升序排列；
- 点集 = 原始 `sample_points`（逐位不变）+ 这些像素的反投影（与 `full_points` 同一公式）。

由此：
- D10 就是冻结的 `P1Fusion`，B0 不变；
- D100 的点集与 `full_points` 完全相同（顺序也相同）；
- 每帧的有效像素数和各档实际点数都记录下来。

**一致性检查（任何一项不通过就 STOP）**：
1. D10 自身分数 = 已存的 V2M 分数（`v2.json`），逐位相同；
2. D100 输入 + D10 评分 = Task 3 几何诊断中的 B2 分数，D100 / D100 = B1 分数，都逐位相同；
3. B2 轨迹状态：D100 输入 + D10 评分 = B2 闭环在线分数，逐位相同；按 Task 2 选择规则的选点 = 闭环实际执行的动作；
4. 嵌套关系与每帧点数核对。

## 3. 运行计划
1. **B2 轨迹状态**：生成清单并计算 oracle（magician 环境）。t = 0 状态与 B0 层重合，oracle 应逐位相同，作为确定性检查。
2. **固定状态主实验**：每条 episode 一个进程，密度按 100 → 75 → 50 → 10 顺序各重放一次，每次只维护一张地图。
   - 室内只做目标状态需要的区域重建（r ∈ {0, 20, 50, 100, 140}）。地图只取决于最近一次重建和之后的增量，结果与逐步重建相同，并由一致性检查验证。
   - 每个状态计算：
     - 几何指标（当前相机 + 候选相机）；
     - 本密度的预测（K = 4，同一噪声）；
     - 输入消融分数：在 D10 那一遍用稀疏 `band_dil` 对 D50 / D75 / D100 的预测评分；
     - 评分消融分数：每一遍用本密度的 `band_dil` 对 D100 的预测评分；
     - 新颖性掩码统计；
     - 伪表面审计。
3. **分析与图**：Figure 1–6。
4. **闭环（有条件）**：按 protocol 中冻结的筛选规则，最多选一档进闭环。
5. **E2 Pilot 方案**：审计训练代码后提交，不训练。

## 4. 资源估计（待冒烟实测后修正）
| 部分 | 估计 |
|---|---|
| B2 状态 oracle | 约 1 GPU-h |
| 固定状态 | MP3D 20 条 episode × 4 遍，Macarons++ 60 条 × 4 遍，约 8–12 GPU-h；4 张 GPU 上约 3 h |
| 闭环（若需要） | 一档 40 条，约 3 GPU-h |
| 显存 | 一次只有一张地图：MP3D 约 10–14 GB，Macarons++ 约 6–8 GB |

## 5. 不在本轮范围内
- 25% 等其他密度；
- 修改模型、DistP1 公式或评分参数；
- 训练（E2 只提交方案）；
- 覆盖 B0 / B1 / B2 的任何结果。
