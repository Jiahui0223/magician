# EXP9-SCONE Phase A：代码审计与实施计划（2026-11-14）

任务书：`TASK_BOOK.md`（逐字保存）。本轮只做 Phase A–C，不启动 Phase D。所有事实均经命令核对；未核对的内容标为【未核实】。

## 1. 结论（先说结果）

- **可以用官方预训练权重直接推理，不需要重训或微调。** SCONE 官方代码与权重的正式出处是 MACARONS 仓库（SCONE 仓库 README 明确指向它）。官方 Google Drive 上的 `pretrained_scone_occ.pth` / `pretrained_scone_vis.pth`（ShapeNet 上 3D 监督训练）已下载，并逐张量核对与本地 `pretrained_macarons.pth` 的 scone 部分**逐位相同**。
- **场景级 SCONE 推理流程有官方实现**：MACARONS 仓库 `testers/scene.py` 的 `compute_trajectory`（占据场 → 每个邻居位姿预测覆盖增益 → 贪心 argmax），配合 `use_perfect_depth_map`，正是 MACARONS 论文 Table 1 里 “SCONE（完美深度、ShapeNet 3D 监督）” 的评测方式；SCONE 论文 §3.2 的场景实验也是同一规则（5D 位姿网格、直接邻居、取预测覆盖增益最大者）。
- 我们 Table 1 的 MACARONS 行（0.660，论文 0.647）用的正是这条流程（EXP3b `m_t7_macarons.py`，修复 R1–R8，GT 深度）。**SCONE = 同一冻结运行器 + 官方 SCONE 权重**，除权重外，动作空间、起点、深度、采样、覆盖计算与 MACARONS 行完全相同。
- 旧记录里 Table 1 的 SCONE 状态为 BLOCKED（理由：“仓库里没有场景级 SCONE planner”）。这个判断不成立：场景级 planner 就是 MACARONS 仓库的 scene tester；MAGICIAN 作者的 SCONE 测试脚本也在其 git 历史 `e4e08fe` 中（见 §3）。

## 2. 代码与权重来源

| 项目 | 位置 | 版本 | 用途 |
|---|---|---|---|
| SCONE 官方仓库 | `/bsuscratch/jiahuizhang/projects/occupancy/SCONE_official/SCONE`（本次新 clone，只读） | `24f1765`（2023-10-20） | 只有 ShapeNet 训练/测试脚本，**没有**场景级推理代码，**没有**权重；README：“MACARONS 仓库包含完整的、更新后的 SCONE 代码”。只作对照 |
| SCONE 论文 | `SCONE_official/paper/scone_2208.10449v3.pdf`（arXiv 下载） | v3 | §3.2 + 附录 C.2：场景实验 = 5D 网格、直接邻居、贪心；附录 C.2 的距离惩罚 1/(η+‖x−c‖²)（η 未给出数值） |
| MACARONS 官方代码 | 本地 clone `/bsuscratch/jiahuizhang/projects/occupancy/MACARONS`（用户 fork，上游 `upstream/main` = `b23c180`，2023-12-23） | `b23c180` | SCONE 网络（`SconeOcc.py` / `SconeVis.py`）、`scone_utils.py`、场景推理函数 `compute_scene_occupancy_probability_field`、`predict_coverage_gain_for_single_camera`、tester `compute_trajectory` |
| 实际执行的代码 | 室外：`MAGICIAN_clean_9521d69`（EXP3b / Table 1 MACARONS 行所用）；室内：官方 MAGICIAN `e697009`（EXP7 MAG-B 室内所用） | — | 与 `b23c180` 逐函数对比：`SconeOcc.py`、`SconeVis.py`、`Attention.py`、`scone_utils.py`、`spherical_harmonics.py` 完全相同；两个场景推理函数除一行空行外相同。差异只在 MAGICIAN 改过的 `get_neighboring_poses`（≤10 邻居，协议统一动作空间）、`get_valid_neighbors`（优先未访问）、`compute_partial_point_cloud`（多一个可选返回值）以及用户 fork 的 Beta 证据计数器（不被 SCONE 读取） |
| 官方权重 | `SCONE_official/weights_gdrive/{occupancy,coverage_gain}/`（MACARONS Google Drive，gdown 下载，已设只读） | 见下 | SCONE 两个模块 |

权重 sha256（运行时由 `scripts/scone_common.py` 再次核对，不符即停止）：

| 文件 | sha256 | 说明 |
|---|---|---|
| `pretrained_scone_occ.pth` | `257674e17f7d693bfdd555eed50e4a2ce9e77d5f3c31e45183d68d0faf404bd9` | 占据概率模块，epoch 301，2,257,769 参数，MSE 损失，ShapeNet |
| `pretrained_scone_vis.pth` | `72c9bfdccde7e50578a7c830f3c09e7653074e9df3f23b8241546210b1a315f0` | 覆盖增益模块，epoch 209，1,392,888 参数，`uncentered_l1` 损失 + sigmoid（MACARONS 仓库的 SCONE 训练配置） |
| `pretrained_macarons.pth` | `c7d66c9f46ac07e676a0a956390460c067aa5ed9bc5ebbaa298300a85d9e54ad` | `generate_macarons_model.py -p` 用上面两个文件生成的容器；scone 部分与两文件 232/232 张量逐位相同（`scripts/inspect_weights.py`） |
| `trained_macarons.pth` | `34c5247f…dbb59c` | MACARONS 自监督微调后的权重（MACARONS / MAGICIAN 行用）；与 SCONE 预训练权重最大差 0.069。**SCONE 行不加载** |

注意：MACARONS 仓库发布的 `pretrained_scone_vis.pth` 用的是 MACARONS 论文提出的 `uncentered_l1` 覆盖增益损失；SCONE 原文用 KL 散度损失训练的权重（`generate_macarons_model.py` 注释里的 `..._kl_divergence_...pth`）**未公开**。这是“官方可得的 SCONE 权重”，即 SCONE 仓库所指的“更新版 SCONE”。

## 3. MAGICIAN 论文里的 SCONE 是怎么跑的

- 论文 Table 2：SCONE AUC 0.534 / Final 0.670（15 场景）；按 Table 6/7 去掉 St. Sofia 的 14 场景均值：**AUC 0.535 / Final 0.673**。
- MAGICIAN 官方仓库首个提交 `e4e08fe`（2025-10-26）含 `macarons/testers/scone_planning.py` + `test_scone.py`，在 `4df5063` 中被删除。拷贝已存：`work_dirs/cvpr_results/scone_baseline/reference/magician_e4e08fe_*`。与 MACARONS 官方 tester 相比，它：
  1. 每步以 **30% 概率随机**选一个合法邻居（`random.random() < 0.3`），否则贪心——这不是 SCONE 的 NBV 规则；
  2. 贪心分支的碰撞判定写反（`if not line_segment_mesh_intersection(...): continue`，只给会撞网格的邻居打分），与其 MACARONS tester 的 R2 错误相同；
  3. 默认测试配置 `model_name = trained_macarons.pth`，即加载的是 **MACARONS 微调后的权重**，不是 SCONE 预训练权重；脚本访问 `trained_weights["depth_losses"]`，`pretrained_macarons.pth` 没有这个键，按原样无法加载 SCONE 权重；
  4. 覆盖未除以 visibility_ratio（R4）。
- 论文数字用的是哪个配置、哪份权重、是否启用 30% 随机：**【未核实】**，论文和仓库都没有说明。因此论文 0.534 只能作参考，不是“官方代码路径”的可复现目标。
- 对照：MACARONS 论文 Table 1（另一个协议）SCONE 平均 0.643，SCONE 论文附录 Table 2 平均 0.625；三者场景、起点、步数、覆盖定义都不同，不可直接比较。

## 4. 我们的统一协议（复用，不改动）

| 部分 | 室外 Macarons++ | 室内 MP3D-150 |
|---|---|---|
| 场景 / 起点 | All-14 × 5 官方起点（`settings.json` start_positions） | EXP7 `mp3d150_manifest.csv`：22 个 val 楼层 × 2 起点 |
| 步数 | C0 + 100 步 = 101 个覆盖值 | C0 + 150 步 = 151 个覆盖值 |
| 动作 | MAGICIAN `get_neighboring_poses`（单轴 ±1 或单角度 ±1，≤10）→ `get_valid_neighbors`（优先未访问）→ 仿真侧 GT 线段碰撞 | T0.12 网格官方邻居 + 仿真侧 GT 线段碰撞；planner 内“优先未访问”（同 MAG-B layer 0 / RAND-NV） |
| 传感器 | 冻结 MAGICIAN 传感器 456×256、60°、GT zbuf、量程 70 runtime 单位；共享 10% 随机采样（种子固定） | 同一冻结传感器，量程 70 rt = 10 m；同一 10% 采样种子 |
| 覆盖评估 | MAGICIAN 官方覆盖（eps 1.0，cell cap 1000，÷ visibility_ratio），`t25_out_cov.py` 重放须逐位一致 | C_obs（30 万 GT 点、固定可见集、eps 0.08 模型单位），EXP7 validator 重放 |
| 汇总 | AUC@T = trapz(C[0..T])/T，Cov@T = C[T]；起点平均 → 场景等权；场景 bootstrap 10,000 次 seed 0 | 同口径，楼层等权，楼层 bootstrap |
| 运行器 | `scripts/scone_sim_out.py` = `m_t7_macarons.py` 逐行拷贝 + 权重/方法名/诊断 4 处差异 | `scripts/scone_sim_in.py`（`t23_in_sim.main()` 原样运行，只替换 child）+ `scripts/scone_planner_grid.py`（MAG-B child 的设置、相机记账、观测更新、LeakGuard 原样复用，只换决策） |

## 5. 原版 SCONE 与我们的适配（详见 `PROTOCOL_ADAPTATION.md`）

原版保留：SconeOcc + SconeVis 网络与官方权重；proxy points 表示体积；每个 cell 用邻近 cell 的部分点云 + view-state 球谐预测占据概率；候选视角 frustum 内按占据概率蒙特卡洛采样 proxy points，用 SconeVis 预测可见性增益并体积积分；NBV = 当前位姿直接邻居中预测覆盖增益最大者（严格 `>`，并列取先出现者）；MACARONS 仓库的场景管线参数（`macarons_default_training_config.json`：80 万 proxy points、cell 划分、`prediction_neighborhood_size` 3、`min_occ_for_proxy_points` 0.1、`seq_len` 2048、距离因子阈值 17、每 20 步重建 surface）。

适配（与 MACARONS 行完全一致）：候选集合 = 统一协议的合法动作列表（MACARONS `b23c180` 原动作：沿一个轴平移恰好 1 格，同时可再转 ≤1 个角度格，最多 30 个邻居，`macarons_utils.py:1953-1963`；统一协议 = MAGICIAN 版“要么平移一格、要么转一格”，最多 10 个；SCONE 论文只写“5D 网格上的直接邻居”。“优先未访问”在 MACARONS 原版 `get_valid_neighbors` 中就有，原版另用 `occupied_pose.pt` 判有效，统一协议改为仿真侧 GT 线段碰撞）；GT 深度来自冻结传感器，10% 点采样（SCONE 附录用 ε_cloud 去重，代码未公开）；无合法动作时原地不动（MAGICIAN dead-end 规则）；种子包装（R8）。室内额外：室内场景 settings（楼层框、1.40 m 水平 cell）与 MAG-B 室内相同；不按室内尺度改任何参数。

不做的事：不加入 RayForge / CDGR / beam search；不加 30% 随机（MAGICIAN e4e08fe 的做法，不是 SCONE 规则，可作为可选敏感性分析，需要用户批准）；不读 GT mesh / 未来深度 / GT 覆盖增益做决策（室外 planner 只接收观测构建的 surface/proxy scene、相机位姿和合法动作键；室内 child 进程装有 LeakGuard，GT/评估文件一律拒绝打开）。

## 6. 计划与成本

- Phase B（本轮完成）：`scone_common.py`、`scone_sim_out.py`、`scone_planner_grid.py`、`scone_sim_in.py`、`scone_out_cov.py`、`scone_validate_in.py`、`scone_queue_tasks.py`、`scone_smoke_check.py`。
- Phase C（本轮）：colosseum s0 室外 20 步 + oLBMNvg9in8_L1 S36 室内 20 步，详见 `SMOKE_TEST.md`。
- Phase D（待批准）：`scone_queue_tasks.py full --write` 一次写入 114 个队列任务（70 室外 + 44 室内），可中断恢复（已完成的任务直接退出 0；残留的半成品目录先移到 `_failed/`）。成本估计见 `SMOKE_TEST.md` §成本。
- Phase D 之后（另行批准）：把 SCONE 写入 Table 1 canonical records（`work_dirs/cvpr_results/table1/episodes/SCONE/`，与 MACARONS 同格式）、EXP7 汇总、`fig/compare_full/` 新版本输出（不覆盖已有文件）。
