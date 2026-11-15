# EXP9-SCONE：原版 SCONE 与协议适配逐项对照（2026-11-14，Phase B 冻结前版本）

标签建议：表中写 “SCONE†”（† adapted：官方 ShapeNet 预训练 SCONE 权重 + MACARONS 仓库场景推理流程，在我们冻结的模拟器 / 起点 / 合法动作 / 评价器下由我们的运行器运行；不是原作者测试脚本）。与 MACARONS† 行同一运行器，只有权重不同。

## A. 原样来自官方（不改）

| # | 组件 | 出处 | 我们的调用 |
|---|---|---|---|
| O1 | 占据概率网络 SconeOcc + 权重 `pretrained_scone_occ.pth` | SCONE 论文 §2.3；MACARONS `b23c180` `networks/SconeOcc.py`；Google Drive | `macarons.scone.occupancy`，strict 加载，sha256 运行时核对 |
| O2 | 覆盖增益网络 SconeVis（球谐可见性增益，sigmoid）+ 权重 `pretrained_scone_vis.pth` | SCONE §2.4；`networks/SconeVis.py`；Google Drive | `macarons.scone.visibility` |
| O3 | 体积表示：场景框内 proxy points（80 万），按深度做 carving（`update_proxy_supervision_occ`，tol 10）、view-state 更新、out-of-field 标记 | SCONE §2.4 / 附录 C.2；MACARONS `Scene` | 室外：`m_t7_macarons.py` 同款调用；室内：`MagicianCore.observe`（同一组官方调用） |
| O4 | 占据场：逐 cell，用相邻 3 个 cell 的部分点云 + view harmonics 预测，在首帧相机坐标系中按预测框归一化（尺度不变） | `compute_scene_occupancy_probability_field` | 原函数，未改 |
| O5 | 覆盖增益：候选 frustum 内 occ > 0.1 的 proxy points 按占据概率采样 2048 个 → SconeVis → 可见性增益 × 距离因子 → 均值 × frustum 内占据体积 | `predict_coverage_gain_for_single_camera`（含 MACARONS 版距离因子 `distance_factor_th = 17`，对应 SCONE 附录 C.2 的距离惩罚） | 原函数，未改 |
| O6 | NBV 规则：对当前位姿的直接邻居逐个预测，取最大（严格 `>`，从 −1 起，并列取先者；默认第一个候选） | SCONE §3.2；MACARONS `testers/scene.py` `compute_trajectory` | `scone_common.scone_greedy`：同一组调用、同一顺序 |
| O7 | surface 点云每 20 步用本轨迹已观测点重建（`fill_surface_scene`） | `testers/scene.py` | 同 MACARONS 行 |
| O8 | 管线参数 | `macarons_default_training_config.json`（MAGICIAN_clean / 官方 MAGICIAN 两份内容相同的配置） | 不改 |

## B. 适配（与 Table 1 其他方法共享，或与 MACARONS 行相同）

| # | 原版 | 适配 | 理由 / 影响 |
|---|---|---|---|
| A1 | MACARONS 原动作：单轴平移 1 格 + 可选 1 格旋转（≤30 邻居）；`occupied_pose.pt` 判有效 | 统一协议合法动作：MAGICIAN `get_neighboring_poses`（单轴平移或单角旋转，≤10）→ 优先未访问 → 仿真侧 GT 线段碰撞；室内为 T0.12 网格同规则 | 所有方法同一动作约束；不改变其他方法 |
| A2 | SCONE：深度传感器（ray casting）；MACARONS tester `use_perfect_depth_map` | 冻结传感器 GT zbuf（clamp 0.5–750），量程 70 runtime 单位；只给深度，不给 RGB | SCONE 本来就是深度方法 |
| A3 | SCONE 附录：按 ε_cloud 去重累积点云（代码未公开）；MACARONS 默认 5% 采样 | 共享 10% 随机像素采样（同一种子同时进 surface 点云与覆盖评估） | 与 MACARONS / MAGICIAN / Random 行相同（EXP3b R5） |
| A4 | 无合法动作时原版行为未定义 | 原地不动一步（MAGICIAN dead-end 规则，EXP3b R3） | 与 MACARONS 行相同 |
| A5 | 原版随机性未固定 | 种子包装：观测采样 / cell cap 用共享种子，planner 用 `sha256(exp3b-M|planner|scone|…)`（室外）/ `sha256(cvpr-SCONE|<tag>|t)`（室内） | 可复现；不改变算法 |
| A6 | SCONE 论文距离惩罚 1/(η+d²)，η 未给 | MACARONS 代码默认阈值版：d ≤ 17 时 1，d > 17 时 17²/d² | 官方代码可得的唯一实现；`smooth` 版（=1/(1+(d/th)²)，与论文形式同）不启用 |
| A7 | 室内无原版设置 | settings 与 EXP7 MAG-B 室内完全相同（楼层框、1.40 m 水平 cell、T0.12 相机网格）；参数不按室内尺度重调（runtime = 7 × 米：距离因子阈值 17 rt ≈ 2.4 m，量程 10 m） | 与 MAG-B 室内一致；参数冻结 |
| A8 | — | 诊断（只记录，不参与决策）：每步占据场统计、可选占据快照、结束时信息流检查（surface 点云的每个点都必须是本轨迹观测到的 10% 深度采样点） | 用于 smoke 与审计 |

## C. 明确不采用

- MAGICIAN `e4e08fe` `scone_planning.py` 的 30% 随机选择、写反的碰撞判定、`trained_macarons.pth` 权重（见 `AUDIT_AND_PLAN.md` §3）。如需与 MAGICIAN 论文 0.534 对照，可另跑 “SCONE-ε0.3” 敏感性分析，必须先经用户批准，且不进 Table 1 主行。
- 重训 / 微调：不需要；SCONE 权重只在 ShapeNet 上训练过，Macarons++ 与 MP3D 的任何场景（含 val/test）都未参与训练，对两个 benchmark 都是零样本。
- 任何手写覆盖启发式替代 SconeOcc / SconeVis。

## D. 公平性说明

- 训练数据：SCONE 仅 ShapeNet（3D 监督）；MACARONS / MAGICIAN 权重在 14 个室外评测场景中的 8 个上自监督训练过；Ours 用 61 个 MP3D train 房屋训练。SCONE 是唯一对两个 benchmark 都零样本的学习方法，这一点应在表注中写明。
- 输入：SCONE 与 MACARONS / MAGICIAN 行一样使用 GT 深度（完美深度）和同一 10% 采样；不使用 RGB、GT 网格、未来深度、GT 覆盖增益。
- GT 只用于：仿真渲染观测、合法动作的线段碰撞（仿真侧，所有方法相同）、独立的覆盖评估。
