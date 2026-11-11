# EXP7 审计记录（2026-11-10，启动前）

任务书：`docs/tasks/cvpr_mp3d_UAVScenes_PARALLEL_EXECUTION.md`。主方法冻结为 V3-T0@40K + D1，不改权重和规划算法。本文件只记录启动前核对过的事实；结果见 `MP3D_150_RESULTS.md`，作业见 `JOBS.md`。

## 1. 代码与资源
- git：分支 `seen2scene`，启动时 HEAD 为 `28873d5`。
- 检查点：`work_dirs/exp6_v3/task0_vae/checkpoints/final.pt`，sha256 `7ed0ddd7…5145ab6`。该模型 `xattn.enabled = false`，没有 `unet.xattn` 参数，共 23,539,622 个参数。
- Slurm（QOS `gpu`：每用户最多 8 张 GPU，超出直接拒绝；`vscode-dev` 会话也算在内）：
  - Pool A：分配 3489889，gpu114，4 × L40S，60 CPU，500 GB 内存，到 11-14。
  - Pool B：sbatch 作业 3494056，gpu115，3 × L40，36 CPU，240 GB 内存，时限 14 h。启动前我已占用 5 张卡，按 QOS 只能再申请 3 张。
  - 3491984 是 gpu107 上另一个会话的分配，没有动。

## 2. MP3D 现有基础设施（全部复用，旧协议代码未改）
| 组件 | 文件 | 说明 |
|---|---|---|
| 模拟器 | `exp5_future_view/task0/magician_comparison/scripts/t12_sim.py` + `adapters/grid_env.py` + `task0/scripts/mp3d_sim.py` | T0.12 网格：1.0 m，5 个俯仰 × 8 个方位，官方 `get_neighboring_poses` 邻居，GT 单线段碰撞判定（只在模拟器里用）。传感器为冻结的 MAGICIAN 设置：456 × 256，60°，量程 70 runtime 单位（10 m）。每条 episode 150 步，共 151 帧 |
| 评价器 | `mp3d_sim.Coverage`（C_obs） | 固定 30 万个 GT 表面点（楼层框内面片，按面积加权采样），可见集固定，ε = 0.08 模型单位（室内约 6–13 cm）。分母与方法无关，不是 Ours 的局部 P1 窗口 |
| MAGICIAN 基线 | `task2/scripts/t23_in_sim.py --group MAG-B` → `adapters/magician_planner_grid.py` | 官方权重 `trained_macarons.pth`，sha256 `34c5247f…dbb59c`，未在 MP3D 上微调；束搜索 [10, 10] |
| Random | `t12_sim.py --method RAND-NV` | 在未访问的合法位姿中按种子均匀抽取；不需要子进程 |
| V3 室内规划器 | `exp6_v3/shared/sim_in_v3.py` / `v3_in_planner.py` | 即 V3 + H0（一步 DistP1）：t06 的 V2K4 路径（K = 4、R10）、n_win = 5、优先未访问，在 `p1_full` b2 下运行 |

现有室内结果，均为 Task 2 的 10 条 episode，150 步：
- MAG-B 10 条：Task 2 与 T0.12 的两次独立运行逐位相同。
- G-RAND 10 条。
- V3-T0 + H0 10 条（EXP6）。

此前不存在室内 D1。

## 3. Episode 列表（冻结为 `manifests/mp3d150_manifest.csv`）
- **规则**：沿用 EXP5 Task 0.3 存档规则，见 `work_dirs/exp5_future_view/task0/manifests/indoor_scenes.json`。
  - 范围是 11 个 MP3D 官方 val 房屋里所有带全景站点图的楼层，每层按种子选 2 个起点。
  - 规则在任何模型输出之前固定。
  - 合计 22 层 × 2 起点 = 44 条 episode。
  - 起点吸附到最近的、线段不被挡住的网格位置，朝向为俯仰 0°、方位 0°（T0.12 规则）。
- **划分**（依据 `task2/scripts/t2_common.py`）：
  - **HO-12**：Task 2 评估楼层所在的 5 个房屋，共 12 层、24 条 episode。从未用于 V2M / V3 的训练或 checkpoint 选择。
  - **VALSEL-10**：其余 6 个 val 房屋，共 10 层、20 条 episode。它们是 V2M 的 checkpoint 选择数据和 V3 的验证监控数据（V3-T0 取固定 40K 步的 final.pt），不是训练数据。报告中单列。
  - **T2-5**：Task 2 的 5 层 × 2 起点，是 HO-12 的子集，可与以前的结果直接对照。注意这 5 层是 T0.11 为可视化挑的，挑选时参考过 T0.10 的结果。
  - V3 的训练数据是 61 个 MP3D train 房屋的 114 个楼层；MP3D test 的 18 个房屋从未读取。
- **方法**（共 176 行）：
  - OURS：44 条，全部新跑。
  - MAG-B：34 条新跑，10 条复用。
  - RANDOM：34 条新跑，10 条复用。
  - OURS-H0（规划深度消融）：34 条新跑，10 条复用。
- **复用条件**：
  - 151 个覆盖值、150 步、网格尺寸相同、起点相同。
  - 模型方法还要求 checkpoint sha 相同、P1 模式为 b2。
  - 在 smoke 中，MAG-B 与 Random 的前 5 步曲线和执行动作都与存档逐位相同（§5）。
- **协议性质**：150 步是本研究固定的短预算协议，不等同于 MAGICIAN 论文 MP3D 的 5,000 步设置；论文数字不进入本表。

## 4. 室内 D1 封装（`scripts/sim_in_d.py`、`scripts/d_in_planner.py`、`configs/d_config_mp3d.yaml`）
- **复用链路**：与 V3 + H0 室内链完全相同（`t23_v2_in_planner` + `p1_full` b2 + `planner_v3.install`）。之后用冻结的 D-v1 `d_policy.install('d1')` 替换打分：两步前瞻，λ = 0，value_cov 的 eps 1.0、cell 1.0，`beam = null`。
  - 文件 sha：`d_policy.py` c8418279…，`h_policy.py` ccfdf4af…。
- **配置**：`d_config_mp3d.yaml` 是 D-v1 `d_config.yaml` 的副本，只改了 `pose_tables` 一个字段，diff 已核对。
- **位姿表**：`scripts/build_mp3d_pose_tables.py` 为 22 层生成，存放在 `work_dirs/cvpr_results/exp7/pose_tables/`。它是场景卡（楼层框、地板、天花板）的纯函数，不读 GT。核对结果见 `scripts/check_mp3d_pose_tables.py`：
  - 在 5 层的历史服务记录中检查了 2,523 个合法位姿，R / T / C 与表逐位相同。
  - 抽查 89,520 个位姿，D1 用的邻居规则 `h_policy.neighbours` 与模拟器的 `grid_env.neighbours` 一致。
  - 结果为 PASS。
- **室内特有细节（不改变决策）**：
  - 室内规划器每次 refresh 都会在区域坐标系里重建 P1 并重放记录 0..t。若不处理，`h_policy.capture` 会重复追加同一批 S10 点。
  - 封装改为每条记录只捕获一次。value_cov 只做"最近观测点是否在 eps 内"的判断，重复点不改变结果。
- **选择规则**：室内链在 D1 分数上使用 Task 2 的优先未访问选择（`nv_choice`），与 H0 室内一致。室外 Table 1 用的是 `m_score.choose`。
- **单位**：value_cov 的 eps 和 cell 都是 1.0 runtime 单位，室内约 0.143 m（runtime = 7 × 米）；评价器 ε 约 6–13 cm。保留冻结值，没有按室内尺度改。
- **未保存的内容**：t06 规划器只返回各候选的分数，所以 `planner.jsonl` 没有 D1 逐根节点的最优路径。D1 分数在 `scores` 字段，`timing.policy_s` 大于 0 可以证明 D1 生效。

## 5. Round-1 smoke（oLBMNvg9in8_L1_S36，5 步，三种方法同时在 gpu114 各占一张卡）
| 方法 | rc | C[0..5] | 墙钟 | GPU 峰值 | 核对 |
|---|---|---|---|---|---|
| Ours（T0 + D1） | 0 | 0.029 0.062 0.090 0.144 0.172 0.207 | 69 s（含加载）；单步决策 4.3–11.1 s，其中 D1 0.4–6.7 s | 24.3 GB | 前 3 步与 V3-T0 + H0 存档相同，第 3 步起 D1 走了不同的动作；planner_info 中 sha、d1、配置路径均正确 |
| MAG-B | 0 | 0.029 0.051 0.066 0.094 0.133 0.148 | 63 s | 3.8 GB | 覆盖与动作前缀和存档逐位相同 |
| Random | 0 | 0.029 0.048 0.058 0.061 0.062 0.062 | 25 s | 1.8 GB | 覆盖与动作前缀和存档逐位相同 |

判定为 GO，随即进入完整运行。

## 6. 调度依据
- **历史数据**：
  - V3-T0 + H0 室内每条约 370–470 s，规划器峰值 12.7–19.6 GB。
  - MAG-B 每条 1,490–1,990 s。
  - Random 每条约 15 s（不含加载）。
- **调度方式**：每张卡跑 1 个 A 槽 worker（OURS，之后 OURS-H0）和 1 个 B 槽 worker（MAG-B，之后 RANDOM）。Pool A 和 Pool B 共享两份任务清单，用 `mkdir` 认领，跨节点是原子操作。每个单元最多试 3 次；失败那次的目录移到 `raw/_failed/`，不删除。已有的 `episode.json` 不会被覆盖。
- **启动后实测**：Ours 约 6 s/步，MAG-B 约 12 s/步，单卡显存峰值不超过 25.2 GB（L40S 为 46 GB）。

## 7. UAVScenes
见 `UAVSCENES_FEASIBILITY.md`。只用了 CPU，下载约 166 MB。数据条件满足 GO-DEPTH；GO-NBV 不成立，因为还没有闭环 adapter，尺度/切块协议也未定。按任务书，Pool B 转为跑 MP3D。
