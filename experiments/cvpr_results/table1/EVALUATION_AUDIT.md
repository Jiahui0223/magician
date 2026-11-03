# EVALUATION_AUDIT —— 历史室外闭环 episode、覆盖率口径、AUC 定义与运行时间

- **日期**：2026-11-03。只读：没有启动闭环，没有使用 GPU，没有修改历史文件。
- **逐 episode 结果**：`audit/historical_outdoor_episodes.json`（含汇总与重复 episode 列表）和 `.csv`（390 行），由 `scripts/audit_model_episodes.py` 生成，只用 CPU，只读取 episode / metrics / 规划器日志。
- **标注约定**：文中“推断”或“未验证”表示没有通过运行或逐位比较确认。

## 0. 结论
1. **每个模型恰好有 30 个可复用的室外 episode，13 个模型共 390 个。**
   - 场景：Dev-3（barts、pantheon、colosseum）和 Test-3（alhambra、bridge、neuschwanstein），每个场景 5 个官方起点；
   - 每个 episode 都完整：101 个执行位姿、101 个覆盖值、moves = 100、101 条步记录、规划器正常结束（bye），没有续跑，没有失败重跑；
   - 每个 episode 的 checkpoint sha256 和 P1 模式都与注册表一致；
   - 覆盖率文件都在，同时含官方口径和单调口径；官方口径重放与 episode 中存储的曲线逐位相同。
2. **这 13 个模型在另外 9 个 Macarons++ 场景上没有任何历史 episode。**
   - 这 9 个场景是 bannerman、dunnottar、eiffel、fushimi、liberty、pisa、redeemer、sestino_museum、sofia_church；
   - 所有辅助目录（回放、smoke、camera pose 等）也只覆盖上述 6 个场景（§4）。
   - 唯一覆盖 14 个场景的是 MAGICIAN 自己的 LMDB（§6）。它不是冻结协议产生的数据，只能作参考列。
3. **历史报告的 Macarons++ 数字都是单调口径**（`auc_mono` / `cov_mono_final`，先在场景内对起点平均，再对场景平均）。由文件重算后与报告逐个一致（§3.2）。
   - 本目录的 `scripts/aggregate.py`:6-7 以官方口径为主口径。按官方口径，各模型的 AUC 比历史报告低约 0.02，Final 低约 0.02–0.03。
4. **AUC 有 4 种写法**（§5）。
   - 我们历史上一直用 `trapz(c[0..100]) / 100`；
   - MAGICIAN 官方脚本 `evaluation_lmdb.py` 用不归一化的 `trapz(c[0..99])`，并且不包含最后一个值；
   - 101 个值的均值比我们的定义低约 0.002。
5. **MAG-B 每个 episode 约 28 分钟**（中位 1706 s，范围 1232–2450 s；规划器每步 15.5 s），是所有组中最慢的；我们的模型每个 episode 2–5 分钟（§7）。

## 1. 冻结闭环协议（全部历史 episode 的来源）
- **模拟器**（magician 环境）：`experiments/exp5_future_view/task2/scripts/t24_out_sim.py` → `task1/scripts/v2_sim.run` → `exp3b_macaronspp_transfer/m_t6_sim.EpisodeSim`。
  - 动作：官方 settings 网格与邻居；
  - 合法性：`get_valid_neighbors`（优先未访问）→ GT 线段检测；没有合法动作时原地不动（hover）；
  - 观测带种子：`MC.derived_seed(scene, start, t, "gt"/"sample"/"cap")`（m_t6_sim.py:61,87,97）；
  - 传感器：456 × 256、60°、量程 70；
  - 预算：C0 + 100 次移动 = 101 帧（task2/configs/eval_outdoor_v1.json:4-7）。
- **规划器子进程**：所有组拿到同一模拟器生成的服务记录（深度、掩码、10% 采样点、合法位姿列表），见 task2/protocol.md:63-75。
- **episode 文件**（`episodes/ep_<tag>.json`）的内容：
  - 每步记录：`t`、`cov`、`pose_key`、`legal_keys`、`pick`、`action_key`、`hover`、`decide_wall_s`，以及 `planner` 下的 `decision_s` / `scores` / `diag`；
  - 规划器握手与结束信息：`planner_hello`（我们的模型含 `ckpt_sha256`）、`planner_bye`；
  - 曲线与汇总：`coverage`（101 个值）、`auc`（= trapz/100，v2_sim.py:162）、`cov_final`、`executed_keys`（101 个）、`n_hover`、`seconds_this_process`。
- **文件保护**：
  - episode 文件写完后设为只读（v2_sim.py:169）；
  - 同一根目录下已有同名 episode 时拒绝运行（v2_sim.py:73）；
  - 中断后可从逐步 checkpoint 逐位续跑。
- **tag 规则**：MAGICIAN / V2O 组为 `<组>_<scene>_s<k>_r0`；其余组都以 `V2M` 组运行，为 `V2M_<scene>_s<k>_r0`，用目录区分模型。

## 2. 覆盖率口径（每个 episode 存了哪条曲线）
| 口径 | 定义 | 存放位置 | 性质 |
|---|---|---|---|
| **官方（official）** | 模拟器内 MAGICIAN 式覆盖率：GT 为场景框内按面积采样的 100k 点，每格最多 1000 个；观测点放入同样的网格（每格上限 1000）；ε = 1.0 运行单位；结果除以 visibility_ratio（magician_protocol_facts.md:74-83） | episode 的 `coverage`；metrics 的 `cov_official` | 带种子；因为每格有上限、点会被替换，曲线**可能下降** |
| **单调（mono）** | `t25_out_cov.py` 按执行位姿重放：GT 点、ε、visibility_ratio 都与官方相同，但保留全部已观测的 10% 采样点（t25_out_cov.py:6-9,57-65） | metrics 的 `cov_mono` | 单调；Task 2 起作为主口径（task2/protocol.md:80） |

- **重放一致性**：metrics 文件是用 `t25_out_cov.py` 重放得到的（wrapper 有 t35_out_cov.py、out_cov_v3.py、out_cov_v4.py、da_out_cov.py，都不改计算）。重放的官方曲线若与 episode 存储曲线不逐位相同，脚本立即停止（t25_out_cov.py:69-71）。
- **本次核对结果**：390 个 metrics 文件的 `official_equal_stored` 全部为 true，且 `cov_official` 与 episode `coverage` 逐位相同；`cov_mono` 全部单调。
- **B0 有两份 metrics**（Task 2 `metrics/V2M` 与 Task 3 `metrics/outdoor/B0/V2M`），30/30 逐位相同。

## 3. 逐模型清单
### 3.1 可复用 episode（每格 = 完整且可复用的起点数 / 5）
| 模型 | 历史组 / closed-loop 根目录（`work_dirs/` 下） | metrics 根目录（`work_dirs/` 下） | barts | pantheon | colosseum | alhambra | bridge | neuschw. | 合计 | sha | P1 模式 | 续跑 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAG-B | `exp5_future_view/task2/eval/outdoor/closed_loop/MAG-B` | `…/task2/eval/outdoor/metrics/MAG-B` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | — | — | 0 |
| MAG-G | `…/task2/eval/outdoor/closed_loop/MAG-G` | `…/metrics/MAG-G` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | — | — | 0 |
| V2O | `…/task2/eval/outdoor/closed_loop/V2O` | `…/metrics/V2O` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | sparse | 0 |
| B0 | `…/task2/eval/outdoor/closed_loop/V2M`（Task 3 `closed_loop/outdoor/B0/V2M` 是指向它的软链接） | `…/task2/eval/outdoor/metrics/V2M`（另有 Task 3 `metrics/outdoor/B0/V2M`，相同） | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | sparse | 0 |
| B2 | `exp5_future_view/task3_full_depth_p1/closed_loop/outdoor/B2/V2M` | `…/task3_full_depth_p1/metrics/outdoor/B2/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| B1 | `…/task3_full_depth_p1/closed_loop/outdoor/B1/V2M` | `…/metrics/outdoor/B1/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | full | 0 |
| E2D | `…/task3_full_depth_p1/closed_loop/outdoor/E2D/V2M` | `…/metrics/outdoor/E2D/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| E2S | `…/task3_full_depth_p1/closed_loop/outdoor/E2S/V2M` | `…/metrics/outdoor/E2S/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | sparse | 0 |
| V3 | `exp6_v3/eval/closed_loop/outdoor/T1/V2M` | `exp6_v3/eval/metrics/outdoor/T1/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| V3.1 | `exp6_v4/eval/closed_loop/outdoor/V31/V2M` | `exp6_v4/eval/metrics/outdoor/V31/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| V4 | `exp6_v4/eval/closed_loop/outdoor/V4/V2M` | `exp6_v4/eval/metrics/outdoor/V4/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| V4C | `exp6_v5/eval/closed_loop/outdoor/V4C/V2M` | `exp6_v5/eval/metrics/outdoor/V4C/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |
| V5 | `exp6_v5/eval/closed_loop/outdoor/V5/V2M` | `exp6_v5/eval/metrics/outdoor/V5/V2M` | 5 | 5 | 5 | 5 | 5 | 5 | **30** | ✓ | b2 | 0 |

- **说明**：
  - `work_dirs/exp6_v3/eval/closed_loop/outdoor/{B0,B2,E2D}` 和对应的 metrics 都是指向 Task 2 / Task 3 的软链接，不是额外数据；
  - hover 只在 V4 中出现，30 个 episode 共 12 次，其他组都是 0；
  - 所有组都在 L40S 节点运行：MAG-* 在 gpu114，其余组在 gpu114–117。
- **可复用的判定**：同时满足 `ep_complete`、`mt_fields_ok`、`sha_ok`、`p1_ok`、`official_equal_stored`，且 metrics 中的官方曲线等于 episode 存储曲线（见 CSV 的 `reusable` 列）。

### 3.2 由文件重算的场景均值（Dev-3 + Test-3，先起点后场景）
| 模型 | 单调 AUC / Final（= 历史报告值） | 官方 AUC / Final | 历史报告出处 |
|---|---|---|---|
| MAG-B | 0.6853 / 0.8907 | 0.6687 / 0.8679 | MODEL_VERSIONS.md:31 “0.685 / 0.891” |
| MAG-G | 0.7036 / 0.9054 | 0.6861 / 0.8821 | exp6_v3/reports/g3.md “0.704 / 0.905” |
| V2O | 0.7243 / 0.8983 | 0.7035 / 0.8683 | MODEL_VERSIONS.md:32 |
| B0 | 0.7290 / 0.8841 | 0.7087 / 0.8541 | exp_density_ab/G0_REPRODUCTION.md:44 |
| B2 | 0.7302 / 0.9062 | 0.7100 / 0.8792 | MODEL_VERSIONS.md:33 |
| B1 | 0.6913 / 0.8692 | 0.6707 / 0.8380 | G0_REPRODUCTION.md:46 |
| E2D | 0.7112 / 0.8802 | 0.6906 / 0.8499 | MODEL_VERSIONS.md:34 |
| E2S | 0.7189 / 0.8805 | 0.6989 / 0.8511 | G0_REPRODUCTION.md:47 |
| V3 | 0.7121 / 0.8825 | 0.6919 / 0.8539 | MODEL_VERSIONS.md:35 |
| V3.1 | 0.6966 / 0.8652 | 0.6760 / 0.8381 | MODEL_VERSIONS.md:36 |
| V4 | 0.7099 / 0.8783 | 0.6889 / 0.8490 | MODEL_VERSIONS.md:37 |
| V4C | 0.7135 / 0.8816 | 0.6925 / 0.8528 | MODEL_VERSIONS.md:38 |
| V5 | 0.7111 / 0.8909 | 0.6895 / 0.8606 | MODEL_VERSIONS.md:39 |

- 按 Dev-3 / Test-3 分开的均值见 JSON `summary[*].split_mean`。
- 例：MAG-B 官方口径 Dev-3 0.700 / 0.868，Test-3 0.638 / 0.867。

## 4. 重复与回放 episode（已核对，不计入可复用数）
| 模型 | 目录（`work_dirs/` 下） | episode | 与主 episode 的关系 |
|---|---|---|---|
| B0 | `exp_density_ab/g0_replay/closed_loop/outdoor/B2_S10_S10/V2M` | colosseum_s0（100 步） | 执行位姿与曲线逐位相同 |
| B0 | `exp5_future_view/task3_full_depth_p1/timing/outdoor_B0/V2M` | colosseum_s0（30 步） | 计时运行，前缀相同 |
| B2 | `exp_density_ab/g0_replay/…/B2_D100_S10/V2M`、`exp_camera_pose/closed_loop/B2_C0/…` | colosseum_s0、barts_s0（100 步） | 逐位相同 |
| B2 | `…/task3_full_depth_p1/timing/outdoor_B2/V2M`、`exp6_v2/smoke_g3/…/B2/V2M` | 30 步 / 2 步 | 前缀相同 |
| B1 | `exp_density_ab/g0_replay/…/B2_D100_D100/V2M` | colosseum_s0 | 逐位相同 |
| E2S | `exp_density_ab/g0_replay/…/E2S_S10_S10/V2M` | colosseum_s0 | 逐位相同 |
| E2D | `exp_density_ab/g0_replay/…/E2D_D100_S10/V2M`、`exp_temporal_refresh/g3/…/E2D_R10/V2M` | colosseum_s0 | 逐位相同 |
| E2D | `exp6_v2/smoke_g3/…/E2D/V2M` | 2 步 | 前缀相同 |
| V4 / V5 | `exp_camera_pose/closed_loop/{V4_C0,V5_C0}/…` | barts、colosseum、pantheon 的 s0 / s1，共 6 个（100 步） | 逐位相同 |
| V3 / V3.1 / V4 | `exp6_v{3,4}/eval/smoke/outdoor/*` | barts_s0（2 步） | 前缀相同 |
| V5 | `exp6_v5/eval/smoke/outdoor/V2M` | barts_s0（2 步） | **不同**（smoke 用的是初始化 checkpoint，不是 v5_final） |
| V2O | `exp5_future_view/task1/closed_loop/E1_V2G_R10_K4/episodes` | Dev-3 × 5 × rep 0/1/2，共 45 个 | rep 0 的 15 个与 Task 2 V2O 逐位相同；rep 1/2 的 30 个是不同 flow 噪声种子的重复（种子键含 rep，closed_loop_v1.json:6），只在 Dev-3 上有，mono 口径未审计 |

## 5. AUC 与 Final 的定义
### 5.1 定义与出处
| 记号 | 公式（c = 101 值曲线，c[0] = C0） | 出处 | 用在哪里 |
|---|---|---|---|
| **A（历史主定义）** | `trapz(c[0..100], dx=1) / 100`（梯形，端点权重 1/2） | task2/protocol.md:81；t25_out_cov.py:76-77；exp3b m_t7_common.py:32-35；task1/scripts/v2_sim.py:162（episode 的 `auc`） | Task 1–3、EXP6 全部报告；本目录 aggregate.py:6-7 |
| **B（101 值均值）** | `mean(c[0..100])` | 没有找到任何脚本把它作为 AUC 使用 | 仅作对照；A − B = mean(c)/100 − (c[0] + c[100])/200，约 +0.002 |
| **C（MAGICIAN 官方 `evaluation_lmdb.py`）** | `np.trapz(c[:100], x=arange(100))`，**不归一化**，只用 t = 0..99（丢掉 c[100]）；同时报告 “Max Cov” = `max(c[:100])` | MAGICIAN/evaluation_lmdb.py:21-25,73-76；e697009 官方克隆中的同名文件与根目录逐字相同 | MAGICIAN 仓库的指标脚本，输出形如 “AUC=70.20” |
| **D（MAGICIAN 仓库 jiahui/exp1）** | `trapz(cov, dx=1) / (len(cov) − 1)`，对 101 值即 A | MAGICIAN/jiahui/stage_1/exp1/run_exp1.py:38-42 | LMDB 复现的 “0.711” |
| 论文 | 只说“最终覆盖”和“覆盖–时间曲线下面积”，没有说明归一化 | magician_protocol_facts.md:85 | 论文表 2：MAGICIAN 0.721 / 0.919（15 个场景） |

- MAGICIAN 的测试器（`macarons/testers/*.py`、`test_magician_planning.py`）只把逐步 coverage 写进 LMDB，**本身不计算 AUC**：在根目录和 e697009 克隆中 grep `trapz` / `auc`，都只命中无关的 `_beta_taucol`。AUC 只由 `evaluation_lmdb.py` 离线计算。
- **Final 的定义**：
  - 我们：Final = c[100]（第 100 次移动之后，`cov_final` / `cov_*_final`）；
  - exp1：c[-1]，对 101 值等同于 c[100]；
  - `evaluation_lmdb.py`：报告的是 max(c[0..99])，不是 c[100]。

### 5.2 同一数据在不同定义下的数值
| 数据（场景均值） | A | B | C（不归一化） | C / 99 | C / 100 | Final c[100] |
|---|---|---|---|---|---|---|
| MAG-B，官方口径，Dev-3 + Test-3 | 0.6687 | 0.6665 | 66.00 | 0.6667 | 0.6600 | 0.8679 |
| B2，官方口径，Dev-3 + Test-3 | 0.7100 | 0.7075 | 70.12 | 0.7083 | 0.7012 | 0.8792 |
| MAG-B，单调口径 | 0.6853 | 0.6831 | — | — | — | 0.8907 |
| MAGICIAN LMDB，14 场景 × 5 | 0.7111 | 0.7087 | 70.20 | 0.7091 | 0.7020 | 0.9060（max(c[:100]) 0.9074） |

- 各模型在 A / B / C 下的值见 JSON `summary[*].scene_mean`。
- 在我们的数据上，定义不同带来的差别约 0.002–0.009；口径不同（官方 vs 单调）带来的差别约 0.02。后者比前者大一个数量级。
- 推断：MAGICIAN 论文的 0.721 与我们的复现 0.711 之间的差，用归一化方式解释不了。C/99 和 C/100 都更低；15 vs 14 个场景、不设种子的噪声更可能是原因（未验证）。

## 6. MAGICIAN 官方 LMDB（`MAGICIAN/results/scene_exploration/magician_lmdb`）
- **内容**（用 magician 环境只读打开，`readonly=True, lock=False`）：
  - 70 个键 = 14 个场景 × 起点 0–4：alhambra、bannerman、barts、bridge、colosseum、dunnottar、eiffel、fushimi、liberty、neuschwanstein、pantheon、pisa、redeemer、sestino_museum，没有 sofia_church；
  - 每个键：`coverage`（101 个值）、`X_cam_history`（102 × 3）、`V_cam_history`（102 × 2）、`points`、`points_color`。
- **怎么产生的**：
  - 2026-06-15 用上游 `test_magician_planning.py` 跑出（MAGICIAN/run.sh：sbatch，`-p gpu-l40`，4 × L40；MAGICIAN/jiahui/PROJECT_SUMMARY.md:185-196）；
  - 设置：作者权重 `trained_macarons.pth`（sha `34c5247f…`，与 MAG-B 相同），`use_perfect_depth_map = true`，beam 10 × 10（MAGICIAN/jiahui/stage_1/exp1/README.md:10-16）；
  - 不设种子（magician_protocol_facts.md:41-42）；
  - 这是我们自己的复现运行，不是作者上传的结果。CLEANUP 文档称其为“作者基线”，指的是用作者权重（MAGICIAN/jiahui/CLEANUP_2026-08-29.md:56）。
- **数值**（定义 A）：

  | 范围 | AUC | Final |
  |---|---|---|
  | 14 个场景 | 0.711 | 0.906 |
  | Dev-3 + Test-3 共 6 个场景 | 0.671 | 0.875 |
  | 我们 MAG-B 同 6 场景，官方口径 | 0.669 | 0.868 |

  - 逐场景值：alhambra 0.627 / 0.841，bridge 0.749 / 0.959，neuschwanstein 0.577 / 0.836，barts 0.822 / 0.997，pantheon 0.551 / 0.743，colosseum 0.701 / 0.872；其余场景 0.60–0.80。
  - 同码重跑时，单个起点可差到 ±0.2（MAGICIAN/jiahui/stage_2/table.md T4 节）。
- **为什么不能直接当作 MAG-B 的冻结协议 episode**：
  - 它来自 MAGICIAN 原生循环：观测采样不设种子，规划与覆盖率都在 MAGICIAN 自己的进程里算，不经过共享的服务记录；
  - 我们的 MAG-B 走 Task 2 适配层（与官方循环逐位等价的 `magician_core`，T0.10 测试 E1），种子键为 `exp5-T2|MAG|<tag>|<t>`，模拟器为 M 协议 EpisodeSim。
  - `experiments/exp3b_macaronspp_transfer/m_magician_replay.py` 可以按 LMDB 存储的位姿在 M 协议下重放，得到带种子的覆盖曲线（EXP3b T5 V1）。但轨迹本身仍是原生循环规划出来的，只能称为“原生 MAGICIAN 轨迹 + M 协议覆盖率”。
  - 建议在 Table 1 中只作参考列（与 magician_protocol_facts.md:123 的结论一致）。

## 7. 运行时间（每个 episode，历史记录，非独占）
- **数据来源**：
  - 墙钟时间：模拟器进程从启动到写出 episode 的时间（`seconds_this_process`，v2_sim.py:167）。所有 episode 都没有续跑，所以是整条 episode 的时间，包括模拟器建场景和规划器启动；
  - 每步决策时间：规划器 `decision_s` 的均值；
  - 覆盖率重放：`t25_out_cov.py` 的 `seconds`。
- **运行环境**：都在 L40S 上与其他作业共卡。
  - MAG-B 的 outdoor worker 每卡 1 个（_sbatch/eval_out_gpu114.sh:3-4），但同一批 GPU 同时有室内评估 worker（logs/eval_indoor_worker_gpu114_*.log 的时间重叠）；
  - B2 每卡 2–3 个 worker（chain_b2.sh:6,9）。

| 模型 | 墙钟中位 s（最小–最大） | 每步决策 s | 覆盖率重放 s | 规划器显存峰值 GB | 8 个新场景 × 5 起点的估计（GPU-h，闭环 + 覆盖） |
|---|---|---|---|---|---|
| MAG-B | **1706**（1232–2450） | **15.5**（Dev-3 14.3，Test-3 16.8） | 77 | 2.4 | ≈ 19 + 0.9 |
| MAG-G | 367（219–559） | 3.0 | 80 | 2.4 | ≈ 4.1 + 0.9 |
| V2O | 130（108–157） | 0.6 | 72 | 8.8 | ≈ 1.4 + 0.8 |
| B0 | 131（107–158） | 0.6 | 58 | 8.8 | ≈ 1.5 + 0.6 |
| B2 | 278（182–420） | 2.0 | 88 | 12.4 | ≈ 3.1 + 1.0 |
| B1 | 191（152–239） | 1.0 | 111 | 8.8 | ≈ 2.1 + 1.2 |
| E2D | 237（199–341） | 1.6 | 99 | 12.4 | ≈ 2.6 + 1.1 |
| E2S | 188（149–232） | 1.0 | 99 | 8.8 | ≈ 2.1 + 1.1 |
| V3 | 155（119–202） | 1.0 | 55 | 12.4 | ≈ 1.7 + 0.6 |
| V3.1 | 161（127–205） | 1.0 | 58 | 12.4 | ≈ 1.8 + 0.6 |
| V4 | 165（133–212） | 1.0 | 57 | 12.4 | ≈ 1.8 + 0.6 |
| V4C | 168（140–211） | 1.0 | 69 | 12.4 | ≈ 1.9 + 0.8 |
| V5 | 252（161–294） | 1.6 | 70 | 12.4 | ≈ 2.8 + 0.8 |

- **MAG-B 的耗时拆解**（Task 2 报告，TASK2_FINAL_REPORT.md:247-258）：
  - 每步展开约 740–770 个 beam 节点，beam 搜索 10–13 s；
  - SCONE 占据预测 2–3 s；
  - novelty 需要从全部历史位姿重新渲染，随步数增长。
  - 上游原生循环在 L40 上约 8–11 分钟 / 条（magician_protocol_facts.md:120）。
- **估计的前提**：最后一列 = 40 × 历史中位时间，假设新场景与 Dev-3 / Test-3 规模相近（未验证）。
  - 主表 8 个模型（MAG-B、V2O、B0、B2、E2D、V3、V4、V5）合计约 34 GPU-h 闭环 + 约 6 GPU-h 覆盖重放，其中 MAG-B 一项约占一半；
  - 5 个可选模型再加约 12 + 5 GPU-h。

## 8. 新 episode 的启动方式（每个模型一条；完整命令见 MODEL_REGISTRY.yaml `launch`）
**公共前缀**：
```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
T1W=$R/work_dirs/cvpr_results/table1/g3; S=<scene>; K=<start>; G=<gpu>
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; unset T3_CKPT V4_P1_MODE V3_CKPT V4_CKPT
```

| 模型 | 闭环命令（都在 magician 环境中运行模拟器，规划器子进程自动启动） | 覆盖率 |
|---|---|---|
| MAG-B / MAG-G | `CUDA_VISIBLE_DEVICES=$G $MG experiments/exp5_future_view/task2/scripts/t24_out_sim.py --group MAG-B --scene $S --start $K --root $T1W/closed_loop/outdoor/MAG-B`（MAG-G 同理） | `$MG …/task2/scripts/t25_out_cov.py --group MAG-B --tag MAG-B_${S}_s${K}_r0 --root $T1W/closed_loop/outdoor/MAG-B --out $T1W/metrics/outdoor` |
| V2O | `… t24_out_sim.py --group V2O … --root $T1W/closed_loop/outdoor/V2O` | `t25_out_cov.py --group V2O --tag V2O_${S}_s${K}_r0 …` |
| B0 | `… t24_out_sim.py --group V2M … --root $T1W/closed_loop/outdoor/B0/V2M` | `DA_G3_ROOT=$T1W $MG experiments/exp_density_ab/scripts/da_out_cov.py --cfg B0 --scene $S --start $K` |
| B2 / B1 | `$MG experiments/exp5_future_view/task3_full_depth_p1/scripts/t33_out_sim.py --p1 b2|full --group V2M … --root $T1W/closed_loop/outdoor/B2|B1/V2M` | `da_out_cov.py --cfg B2|B1` |
| E2D / E2S | `T3_CKPT=$R/work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100|s10/ema_005000.pt … t33_out_sim.py --p1 b2|sparse --group V2M … --root $T1W/closed_loop/outdoor/E2D|E2S/V2M` | `da_out_cov.py --cfg E2D|E2S` |
| V3 / V3.1 | `$MG experiments/exp6_v3/shared/sim_out_v3.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt（V3.1：$R/work_dirs/exp6_v3_1/checkpoints/final.pt）--group V2M … --root $T1W/closed_loop/outdoor/V3|V3.1/V2M` | `da_out_cov.py --cfg V3|V3.1` |
| V4 / V4C | `$MG experiments/exp6_v4/scripts/sim_out_v4.py --v4-ckpt $R/work_dirs/exp6_v4/checkpoints/final.pt（V4C：$R/work_dirs/exp6_v5/checkpoints/v4c_final.pt）--group V2M … --root $T1W/closed_loop/outdoor/V4|V4C/V2M` | `da_out_cov.py --cfg V4|V4C` |
| V5 | `$MG experiments/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $R/work_dirs/exp6_v5/checkpoints/v5_final.pt --group V2M … --root $T1W/closed_loop/outdoor/V5/V2M` | `da_out_cov.py --cfg V5` |

- **一定要给 `--root`**：不给时所有入口都写入 Task 2 历史目录（t24_out_sim.py:70）。
- **覆盖率重放的临时目录**：
  - `da_out_cov.py` 把临时回放帧放在 `$DA_G3_ROOT/cov_tmp/<cfg>`，不同模型之间没有冲突（da_out_cov.py:1-3,18）；
  - 历史的 `out_cov_v4.py` 用同一个 `V4_EVAL_ROOT/eval/outdoor/replay_frames/V2M_V2M_<tag>`，不同模型同一 tag 并行时会冲突；
  - `out_cov_v3.py` 把根目录写死在 `work_dirs/exp6_v3/eval`（out_cov_v3.py:11）；
  - MAG-* / V2O 的 tag 不是 `V2M_…`，只能直接用 `t25_out_cov.py`，它的临时帧写在 Task 2 目录下（`task2/eval/outdoor/replay_frames/<group>_<tag>`，结束删除，t25_out_cov.py:48-49,68）。
- **运行后核对**：
  - episode：101 个 `executed_keys`、`planner_bye.bye`；
  - 我们的模型：`planner_hello.ckpt_sha256` 等于注册表；规划器 stderr 第一行的 P1 模式与注册表一致；
  - metrics：`official_equal_stored = true`。
- **建议（未执行）**：批量运行前，对每条代码路径用 `MOVES=2` 在一个新场景上做 smoke，并对一个历史 episode 做 2 步前缀逐位回放。exp6_v2/shared/g3_queue.sh 和 da_g3_queue.sh 都支持 `MOVES`。

## 9. 无法在冻结协议下评估的情况
1. **sofia_church（所有模型）**：
   - 模拟器端（已实测）：冻结模拟器从 MAGICIAN_clean_9521d69 导入 `load_scene`，它的多纹理分支含 `torch.tensor()`（macarons_utils.py:494）；sofia_church 是唯一有多个 map_Kd（4 个）的场景。同目录 `load_test.log`:15 在 EpisodeSim 阶段报 `TypeError: tensor() missing 1 required positional arguments`。
   - 规划器端（读代码）：我们的 V2 系列规划器（含 V3 / V4 / V5）在 v2_planner.py:112 → t5_diagnostic.py:110-112 读取 `em2_windows.json`。该文件只有 14 个场景，没有 sofia_church，会抛出 KeyError。MAGICIAN 规划器不读这个文件。
   - 修复需要同时改模拟器和窗口表，属于协议变更。
2. **MAGICIAN LMDB 的 14 场景结果**：不是冻结协议产生的数据（§6），只能作参考列，不能当 MAG-B 的 episode 使用。
3. **V5.2 / V5.2-M**（未登记）：只有 pilot / smoke 闭环，冻结门槛未通过。
4. 除以上外，13 个登记模型都有可运行的冻结启动链；没有发现场景白名单。
   - 另外 8 个 MAGICIAN 测试场景都在 `em2_windows.json` 中；
   - z0 按 settings.json 计算（abl_common.z0_of，适用于任何场景）。
   - **未验证项**：新场景若明显更大，整场景规划区域的显存与耗时是否会超出历史范围（历史显存峰值 12.4 GB）。
