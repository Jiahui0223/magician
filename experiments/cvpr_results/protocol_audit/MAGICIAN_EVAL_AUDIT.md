# MAGICIAN_EVAL_AUDIT — 评估协议逐项审计（A1）

- **日期**：2026-11-04。只读审计：没有改动任何历史结果、没有重跑闭环、没有训练。
- **状态**：PASS = 证据充分且一致；PARTIAL = 部分一致或有已披露的差异；FAIL = 不一致；UNKNOWN / UNRESOLVED = 证据不足。

**缩写**：
- **OFF**：官方仓库 github shiyao-li/MAGICIAN，本地克隆 `work_dirs/exp5_future_view/task0/magician_comparison/external/MAGICIAN`，HEAD e697009（2026-08-29）；
- **FORK**：`MAGICIAN_clean_9521d69`，用户 fork，冻结模拟器实际导入它；
- **SIM**：冻结 M 协议模拟器 `experiments/exp3b_macaronspp_transfer/m_t6_sim.py`（经 `exp5_future_view/task1/scripts/v2_sim.py` 运行）；
- **PAPER**：`MAGICIAN/Magician.pdf`（arXiv 2603.22650），页码按 PDF 页；
- **T1**：`experiments/cvpr_results/table1/`。

**自动生成的依据**（`scripts/`，结果在 `results/`）：
- `diff_eval_functions.py` → `results/eval_function_diff.md`：OFF 与 FORK 的评估函数逐个比较；
- `audit_starts.py` → `INITIAL_POSES_COMPARISON.csv`：起点；
- `audit_metrics.py` → `results/metric_audit.json`、`SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv`：公式测试与论文对照。

## 1. 场景与数据集
| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 1.1 | 论文表 6 / 7 的 15 个场景与本地 15 个目录一一对应 | PASS | PAPER p.12 表 6 / 7；本地 `MAGICIAN/data/Macarons++/` 15 个目录；名称对照见 `audit_metrics.py` 的 `PAPER_NAME` |
| 1.2 | 我们的 Train-6 / Dev-5 / Test-3 / Other 划分 | PASS（已披露） | 来自冻结的 `exp4_multiworld_belief/task0/configs/dataset_manifest_v1.json`（T1 `SCENE_REGISTRY.yaml`）。这是我们自己的划分，与 MACARONS 的 train 8 / val 2 / test 2（`configs/macarons/macarons_default_training_config.json:18-35`）不同，两者都在 T1 `DATASET_AUDIT.md` §2 披露 |
| 1.3 | sofia_church 缺失的原因 | PASS（原因确认）/ 结果缺失 | FORK `macarons_utils.py` 的 `load_scene` 多出一行 `torch.tensor()`（多纹理分支）。sofia_church 是唯一一个有 4 张 `map_Kd` 纹理的场景，因此报 `TypeError`（T1 `load_test_results.json`）。OFF 在 eec9e1d（2026-08-29）才把这行注释掉，之前的官方提交 e4e08fe / 4df5063 / deb7264 都有这一行。去掉这一行后，在诊断补丁下可以构造，起点也一致（T1 `sofia_diag_result.json`）。OFF 的测试配置 `configs/test/test_in_default_scenes_config.json:4-19` 只列 14 个场景，但 PAPER 表 6 / 7 有 15 个 |
| 1.4 | All-14 / All-15 严格区分 | PASS | T1 只输出 All-14；All-15 在所有生成文件中都标为 `all15-partial14of15`（T1 `PROTOCOL.md` §1） |
| 1.5 | 几何、坐标系、尺度与资产版本 | PARTIAL | 资产 sha256 与冻结清单一致（T1 `sha256_check.txt`）；×10 尺度、y 轴向上，OFF / FORK 的加载与设置函数相同（§4.1）。**PAPER 使用的数据版本没有说明**，无法确认与公开的 HF 数据集 sli016/Macarons-plus-plus 是同一版本（UNKNOWN） |

## 2. 初始相机位姿（最高优先级）
**明细**：`INITIAL_POSES_COMPARISON.csv`。每个场景 × 起点一行：settings 起点下标、位姿表得到的位置 / 俯仰 / 方位、LMDB 中 MAGICIAN 上游循环的第一个位姿与差值、Table 1 各模型第 0 步位姿的核对数、是否被用作我们训练轨迹的起点。

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 2.1 | 起点的定义：`settings.json camera.start_positions[k]`（5 维网格下标 [l, w, h, 俯仰档, 方位档]） | PASS | OFF `macarons/testers/magician_planning.py:834-835` 依次取 `settings.camera.start_positions[start_cam_idx_i]` 传给 `setup_test_camera`；SIM `m_t6_sim.py:69-71` 把 `start_positions[k]` 传给同一个函数。OFF README 称其为“5 个随机采样的起始位姿，用来初始化独立轨迹” |
| 2.2 | 与 MAGICIAN 上游循环实际运行的起点一致 | PASS | 我们用上游 `test_magician_planning.py` 复现的 LMDB（14 场景 × 5）中，`X_cam_history[0]` 与 `V_cam_history[0]` 和 settings 起点经 MAGICIAN 相机代码换算的结果 **70 / 70 一致**（最大 \|ΔX\| = 4.5e-5，俯仰 / 方位差 0） |
| 2.3 | 不同模型严格使用相同的 5 个起点 | PASS | Table 1 共 26 个模型、1655 条 episode，第 0 步位姿**全部**等于对应的 `start_positions[k]`，0 处不一致 |
| 2.4 | 重复、缺失、顺序错位、坐标转换错误 | PASS | 每个场景的 5 个起点互不相同；k = 0..4 的顺序与 settings 一致；所有完整模型每个（场景, 起点）都有 episode；位置由 MAGICIAN 自己的相机代码换算（位姿表 `work_dirs/exp6_v3/H/pose_tables/*.npz`，已与服务记录核对） |
| 2.5 | 是否存在按 AUC / Final 挑选起点 | PASS | T1 `scripts/aggregate.py` 只接受 5 个起点都完整的场景，不完整则标 partial，没有任何选择或替换起点的逻辑（T1 `PROTOCOL.md` §5）；LMDB 复现同样是全部 5 个 |
| 2.6 | 是否与**论文实际使用**的 5 个位姿相同 | UNKNOWN（推断为相同） | PAPER p.6 只说“every method uses identical random initial camera poses”，p.13 说“five starting poses in each scene are randomly sampled”，没有给出位姿，也没有发布论文的轨迹。我们与官方发布的 settings 和官方代码完全一致；“论文用的就是这 5 个”只是推断 |
| 2.7 | 随机种子 | PASS（差异已披露） | OFF 测试器不设种子：配置里有 `random_seed` / `torch_seed` 字段，但 `test_magician_planning.py` 与 `magician_planning.py` 都没有 `manual_seed` 调用。SIM 对 GT 采样、10% 采样、覆盖填充都按（场景, 起点, 步）设种子（`m_t6_sim.py:61,87,97`），所有模型在同一（场景, 起点）上的评估采样相同。种子只影响评估噪声，不改变起点 |
| 2.8 | 训练数据与评估起点的重叠 | 已披露 | 在我们实际的训练集 train6_minus_Sc 中，每个 Train-6 场景有 3 个官方起点是训练轨迹的第一帧：bannerman 1/2/3、dunnottar 0/1/2、fushimi 1/2/3、liberty 0/1/3、pisa 0/1/2、redeemer 1/2/3。被 S-c 留出的那个起点不在训练集中；官方起点 4 从未使用（`INITIAL_POSES_COMPARISON.csv` 的 `first_pose_of_a_TRAINING_split_trajectory` 列；划分取自 `exp5_future_view/task1/frozen_values.json`）。若算上 S-c 在内的全部 180 条轨迹，起点 0–3 都出现过。Dev-5 / Test-3 没有任何训练轨迹。MAGICIAN 所用 MACARONS 权重的情况见 `MAGICIAN_TRAINING_AUDIT.md` |

## 3. 评估步数与动作空间
| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 3.1 | 100 步；起点观测记为第 0 步；101 个覆盖值 | PASS | OFF `magician_planning.py:464` 的 `while pose_i <= params.n_poses_in_trajectory`（配置 `macarons_default_training_config.json:55` 为 100），每次循环在 516 行追加一个覆盖值，共 101 个。SIM / `v2_sim.py:125-135`：t = 0..100 观测，第 100 步只观测不移动。Table 1 所有 1655 条 episode 都有 101 个覆盖值与 101 个位姿；PAPER p.6 表 1 写 Macarons++ 为 100 步 |
| 3.2 | 动作空间与合法移动 | PASS | 候选 = `get_neighboring_poses`（单轴 ±1 格或俯仰 / 方位 ±1 档，最多 10 个）→ `get_valid_neighbors`（优先未访问，没有未访问的才用已访问的；OFF `macarons_utils.py:3898-3930`）→ 对当前位置到目标位置的线段做 GT mesh 碰撞检测（OFF `magician_planning.py:611-627`，第 1 层）。这三个函数在 OFF 与 FORK 中**逐字相同**；SIM 的 `legal_neighbours`（`m_magician_replay.py:84-102`）按同样顺序调用它们 |
| 3.3 | 边界、碰撞、已访问位姿、无合法动作 | PARTIAL | 边界（网格截断）、碰撞、已访问的规则都相同。**无合法动作时**：OFF 直接 `break`，提前结束轨迹（`magician_planning.py:706-721`）；SIM 原地停留一步（hover，`m_t6_sim.py:123-128`）并照常记一帧。实际影响：Table 1 中 hover 只在 V4 出现（70 条中共 12 次），MAG-B 与所有主表模型都是 0 次；LMDB 的 70 条也都满 101 步。**对主表数值没有影响** |
| 3.4 | 真实观测预算一致 | PASS | 每一步只执行一个动作、拍一帧（`n_interpolation_steps` = 1，配置 :54；OFF :723-733；SIM `move()`）。beam search 的 Nd 步前瞻只用想象的视图，不消耗真实观测 |
| 3.5 | Beam 与 Greedy 设置 | PASS | MAG-B = beam 宽 10、深 10、每次执行 1 步，与 PAPER p.6（Nb = 10，Nd = 10，Nf = 1）和 OFF 测试配置（`beam_width` 10，`beam_steps` 10）相同；MAG-G = 1 × 1（`task0/magician_comparison/adapters/magician_planner.py:36`）。MAG-B 由 OFF 代码经适配器运行，与官方逐位一致（`t24_mag_out_planner.py` 头注释，测试 E4） |

## 4. 覆盖率计算
| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 4.1 | 官方代码的覆盖计算 | PASS（与官方发布代码一致） | 计算链：GT 点 = `get_scene_gt_surface` 在 settings 场景框内按面积采样 `n_gt_surface_points` = 100,000 点（OFF `magician_planning.py:306-312`，配置 :67），每格上限 `surface_cell_capacity` 1000（:63）→ 每帧反投影 z < 70 的像素并随机保留 10%（`gathering_factor` 0.05 × 2，OFF :437）→ `covered_scene.fill_cells` → `gt_scene.scene_coverage(eps = 2 × 0.05 × 10 = 1.0)` / `settings.scene.visibility_ratio`（OFF :510-516）。`get_scene_gt_surface`、`Scene.scene_coverage`、`Scene.fill_cells`、`compute_partial_point_cloud`、`setup_test_scene`、`load_current_frame_perfect_depth` 在 OFF 与 FORK 中**逐字相同**（`results/eval_function_diff.md`）；SIM 的 `observe()`（`m_t6_sim.py:79-102`）按同样顺序调用 |
| 4.2 | 论文所述的 GT 点云 | UNRESOLVED | PAPER p.13 B.1：“unlike prior work … that directly samples point clouds from the ground-truth mesh … we generated the ground-truth point cloud by rendering depth maps from all accessible viewpoints and projecting them into a 3D point cloud”。**官方发布的代码仍从 mesh 上采样并除以 `visibility_ratio`**，没有发布渲染 GT 的代码或数据。我们与发布的代码一致，但与论文文字描述的 GT 不同；两种 GT 下数值差多少无法离线重算（缺少论文的 GT 点云） |
| 4.3 | 累计与去重 | PASS（已披露） | 观测点逐帧放入 `covered_scene`，每格最多 1000 点，满了会被替换，因此官方曲线**可能下降**：主要模型 94–97% 的 episode 至少有一次下降，平均最大回落约 0.002–0.005。GT 点在格内用最近邻判定（< 1.0），不做额外去重 |
| 4.4 | 官方口径 vs 历史单调口径 | PASS（不混用） | 单调口径保留全部已观测点、没有容量上限（`exp5_future_view/task2/scripts/t25_out_cov.py`），All-14 上比官方口径高约 0.02 AUC、0.03 Final（例如 MAG-B 官方 0.711 / 0.906，单调 0.730 / 0.932）。主表一律用官方口径；单调口径只出现在 `results/*_mono.*` 中 |
| 4.5 | S10 / D100 是否影响评估 | PASS | S10 / D100 只决定规划器自身输入和评分用多少点；评估由模拟器按 10% 采样独立计算，规划器只能看到服务记录（`exp5_future_view/task2/protocol.md:63-75`）。同一（场景, 起点）上所有模型的 GT 点与评估采样都相同（种子见 2.7） |
| 4.6 | 所有方法使用同一评估器 | PASS | Table 1 的所有行（含 MAG-B / MAG-G）都在同一个冻结 SIM 中运行；每条 episode 都由 `t1_out_cov.py` 重放，官方曲线与存档逐位一致才会导入（T1 `EVALUATION_AUDIT.md` §2、T1 `validate.py`） |

## 5. AUC 与 Final（详见 `AUC_COVERAGE_FORMULA_AUDIT.md`）
| # | 项 | 状态 | 证据 |
|---|---|---|---|
| 5.1 | AUC 用梯形积分，包含第 0 步与第 100 步，除以 k | PASS | 对 T1 `aggregate.py:metrics` 原样调用的单元测试：常数 1 → 1.0；线性 t / 100 → 0.5；只在 t = 0 或只在 t = 100 为 1 → 0.005（端点权重 1/2）；第 50 步起为 1 → 0.505。全部与解析值一致 |
| 5.2 | AUC@20 / 50 / 100 与 Cov@20 / 50 / Final 的定义 | PASS | AUC@k = trapz(C[0..k]) / k；Cov@k = C[k]；Final = C[100]（`aggregate.py` 断言 Cov@100 == Final） |
| 5.3 | Final 取最后一步还是最大值 | PASS（已披露） | 我们取 C[100]。官方曲线不单调，max(C[0..100]) 比 C[100] 平均高 0.002–0.005（MAG-B 0.9085 vs 0.9063），51–59% 的 episode 有 C[100] < 最大值 |
| 5.4 | 场景内先对 5 个起点平均、再场景等权平均 | PASS | `aggregate.py` 先求场景均值再取场景等权平均。因为每个场景都正好 5 条，与直接对 70 条 episode 平均完全相同（本次重算 MAG-B / B2 / D1 / YBEST-H3 / V3 均一致） |
| 5.5 | 与论文的 AUC 写法是否一致 | UNKNOWN（差异有界） | PAPER 只说“area under the curve of coverage over time”，没有给出归一化方式。OFF 的 `evaluation_lmdb.py` 用不归一化的 `np.trapz(c[:100])`，不含 C[100]；它报告的“Max Cov”= max(c[:100])。在同一份 MAG-B 数据上，各种写法得到 0.702–0.711，与我们的写法最多相差 0.009 |

## 6. 论文公布值 vs 复现（明细：`SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv`）
**三类数值**：
- **MAGICIAN Paper Reported**：PAPER p.12 表 6 / 7，15 场景。本次从 PDF 文本独立转录，与 T1 `SCENE_REGISTRY.yaml` 逐值一致（0 处不同）。
- **MAGICIAN Reproduced**：分两份：
  - MAG-B：冻结 SIM、官方代码与权重、设种子，即 Table 1 的行；
  - LMDB：用上游 `test_magician_planning.py` 原样运行、不设种子，只作参照。
- **Our Methods**：Table 1 的其余行。

| 汇总 | AUC | Final |
|---|---|---|
| Paper，All-15（论文表 2 / 6 / 7 的平均） | 0.721 | 0.919 |
| Paper，相同的 All-14（去掉 sofia_church，由逐场景值重算） | 0.7214 | 0.9211 |
| Reproduced MAG-B，All-14 | 0.7108 | 0.9063 |
| Reproduced LMDB，All-14 | 0.7111 | 0.9060 |
| MAG-B − Paper（All-14，场景级 bootstrap 95% CI） | −0.0105 [−0.025, +0.002]，9/14 个场景低于论文 | **−0.0148 [−0.032, −0.001]**，10/14 个场景低于论文 |

- **两份复现彼此一致**：MAG-B − LMDB 为 AUC −0.0002，Final +0.0004。设种子的冻结模拟器与上游原样循环给出几乎相同的均值。
- **逐场景**：
  - AUC 14/14 个场景、Final 11/14 个场景的差值在“复现起点标准差 / √5 的 2 倍”以内；
  - 与论文的相关系数：AUC 0.96，Final 0.94。
- **差距的集中位置**：
  - Final：pantheon −0.102、neuschwanstein −0.051、sestino_museum −0.046、bannerman −0.021；
  - AUC：pantheon −0.083、neuschwanstein −0.043。
- **pantheon 起点 2**：MAG-B 0.332、LMDB 0.353，两次独立复现都很低；MAG-G、B2、D1 在这个起点上也只有 0.56–0.57。
- **neuschwanstein 起点 2 和 4**：对所有方法都偏低。
- **这些低值不是运行噪声**，两次独立复现都出现，所有方法也都受影响。而论文中这两个场景的 Final 达到 0.84–0.85，意味着论文的 5 次运行都没有落入同样的低值。

| 候选原因 | 结论 |
|---|---|
| AUC 写法 | **排除**：任何写法都只让复现值更低（0.702–0.711），与论文的差距没有缩小 |
| Final 取最大值 | **几乎排除**：只能缩小约 0.002 |
| 运行随机性 | **对 AUC 不能排除**（CI 含 0）；**对 Final 不足以解释**：差距集中在几个复现两次都出现的困难起点 |
| 起点不同 | **可能**：论文没有公布它的 5 个位姿（2.6） |
| GT 点云不同 | **可能**：论文用渲染得到的可见点，发布代码用 mesh 采样再除以 visibility_ratio（4.2） |
| 场景数据或代码版本 | **可能**：论文发表时的官方代码连 sofia_church 都无法加载，说明论文结果不是用发布的代码版本跑出来的（1.3） |
| 结论 | **UNRESOLVED**：没有足够证据确定原因，也不宣布已对齐 |

**另一个发现**：论文表 7 中 FisherRF 的平均 Final 写作 0.786（表 2 同），但它自己 15 个逐场景值的均值是 0.772；其他方法的平均都与逐场景值一致（差异在四舍五入范围内）。**论文内部不一致（FAIL，论文侧）**。我们的论文表格一律由逐场景值重新计算平均，并在脚注说明。
