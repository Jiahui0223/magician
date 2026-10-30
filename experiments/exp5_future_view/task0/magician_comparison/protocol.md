# EXP5 Task 0 扩展：MAGICIAN 室内对照 —— 协议 v1.0（T0.8）

> 2026-10-24，依据你 2026-10-24 的扩展任务书（T0.8–T0.11）。
> **本版本只冻结 T0.8**。T0.9（官方复现）和 T0.10（同协议对照）的规则在各自运行前另写增补 v1.1 / v1.2，正文不改。偏差只追加到 `DEVIATIONS.md`。
> T0.1–T0.7 的结果、轨迹、模型和协议一律只读，不覆盖、不重算、不改写。

`W` = `work_dirs/exp5_future_view/task0`，`MC` = `W/magician_comparison`。

## 1. 路径（2026-10-24 核实）
- **MP3D 数据**：物理路径 `/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans`，与 `/bsuscratch/jiahuizhang/data/matterport3d/v1/scans` 是同一位置（`/bsuhome/jiahuizhang/scratch` → `/bsuscratch/jiahuizhang`）。
  - 90 个场景目录，T0.2 已逐个校验 sha256；
  - 当前节点 gpu115 可读；
  - 不移动、不复制。
- **项目**：`/bsuscratch/jiahuizhang/projects/occupancy/seen2scene`（物理路径相同）。
- **MAGICIAN 官方代码**：`MC/external/MAGICIAN`。
  - 2026-10-24 用 `git clone --recursive` 获取，HEAD `e697009b3f9360adfe65e8db1e5c27421158e388`（2026-08-29）；
  - 子模块：RaDe-GS `5f5cd3b0…`、simple-knn `44f76429…`、glm `5c46b9c0…`。

## 2. T0.8 范围
- **轨迹**：T0.6 的 22 条 episode × 5 组，共 110 条，每条 101 帧（t = 0..100）。
  - 随机参考：`W/preprocessing/traj/<tag>/`；
  - Frozen / T1-C / V2K4 / V2K1：`W/closed_loop/<组>/<tag>/`。
- **只复算评估量，不做任何模型推理或新的环境交互。**
- **观测来源**：T0.6 的服务记录 `svc_t<t>.npz`。
  - 内容：执行位姿处 M 协议（GT 网格渲染）的 256 × 456 zbuf、mask、R、T，以及 10% 采样点。
  - 报告中一律标为"M 协议执行位姿处的网格渲染观测（T0.6 原始记录）"，不是真实传感器数据。
- **复现检查**：用冻结的 `mp3d_sim.Coverage`，由保存的 10% 采样点重算每条 episode 的 101 个 C_obs，与保存的曲线比较。
  - 要求 110 × 101 个值全部一致，|Δ| ≤ 1e−12；
  - 不一致就 STOP，不进入后续计算。

## 3. 单位与转换测试（正式计算前必须全部通过）

**约定**：
- MP3D 网格：z 向上的米。
- 运行坐标：y 向上，由 z 向上的米 (x, y, z) 得到 rt = 7 · (x, z, −y)（`mp3d_sim.zup_m_to_rt`）。
- zbuf 是运行单位的 z 深度；量程 70 = 10 m，近裁剪 0.5 = 7.1 cm。
- 模型单位 = k 个运行单位 = k / 7 m。k = 层高（运行单位）/ 2.816；22 个楼层上 k 为 5.28–11.19。

**由此推出**：
- 覆盖率容差 ε = 0.08 **模型单位** = 0.08 k 运行单位 = 0.08 k / 7 m。22 个楼层上为 6.03–12.78 cm，中位数 7.80 cm。
  - 这与 T0 报告的"约 6–13 cm"一致；
  - 只有误把 0.08 当成运行单位，才会得到 1.14 cm。
- 体素 = 0.011 模型单位 = 0.83–1.76 cm。
- 5 cm = 0.35 运行单位 = 0.35 / k 模型单位。

**测试**（脚本 `tests/t08_units_test.py`，结果写入 `MC/metric_replay/units_test.json`）：

| 编号 | 内容 | 期望 |
|---|---|---|
| U1 | 随机点沿 x / y / z 各移动 0.05 m 后，运行坐标中的距离 | 0.35，误差 < 1e−9；往返变换误差 < 1e−9 m |
| U2 | 3 帧完整深度的反投影在 `sample_pixel_flat` 处的值，与保存的 `sample_points` 比较 | 最大差 < 1e−4 运行单位（14 µm） |
| U3 | 3 帧各 2,000 个完整深度点到 GT 网格的距离（网格渲染深度，期望约为 0） | 中位数 < 0.5 cm，95 分位 < 2 cm |
| U4 | 每条 episode t = 0 的相机中心高度（m）减去卡片地面高度 | 在 [1.3, 1.7] m 内（三脚架高度 1.43–1.61 m，T0.3 F10） |
| U5 | `2azQ1b91cZZ_L0` 上 `Coverage.eps_rt / 7` | 0.08 × 7.40057 / 7 = 0.084578 m，误差 < 1e−9 |

## 4. 指标

### 第一组：原 Task 0 协议
- **覆盖率**：Cov@10 / 20 / 50 / 100、AUC@100（trapz / 100）和完整曲线。用通过复现检查的已存曲线。
- **观测**：
  - 有效观测帧数：至少有 1 个有效像素（mask 且 z < 70）的帧；
  - 每帧有效像素数。
- **站点访问**：
  - 首次访问的站点数：不同站点数，含起点；
  - 站点回访：移动到已访问站点的次数，以及它占移动次数的比例。
- **位姿回访**：执行位姿 (站点, 俯仰, 方位) 此前已执行过的步数，以及它占 100 步的比例。
- **移动距离**（m）：每次移动的站点间直线距离之和。
- **停滞**：
  - 覆盖率零增长的步数（ΔC_obs ≤ 1e−12）；
  - 最长连续零增长长度；
  - 位姿回访步上获得的覆盖增量之和。
- **区域**：访问过的 `.house` 区域数和区域切换次数。区域标注只在评估侧使用。
- **时间**：每步决策时间均值和 episode 墙钟时间，来自 `episode.json`。随机参考没有规划时间，记为"—"。

### 第二组：MAGICIAN 论文的 MP3D 指标
论文只给了文字定义；官方发布代码里没有实现，`evaluation_lmdb.py` 只计算覆盖率和 AUC。所以下面是**按论文文字定义的本项目实现**，数值不与论文表格直接比较。

- **GT 点**：textured `matterport_mesh` 中被面片引用、且位于楼层框 `box_m` 内的顶点。
  - 楼层框与 C_obs 相同；
  - 每层 1.7 万 – 23.8 万个，22 层共 2,479,003 个；
  - 用顶点，不用面采样点，与论文"GT vertices"一致。
- **观测点云**：
  - **主口径**：执行帧 t = 0..T 中所有有效像素（mask 且 z 深度 < 70 运行单位 = 10 m）的完整反投影，换算成米。
  - **次口径**：只用 10% 采样点，即规划器和覆盖率实际使用的观测集合，相当于 MAGICIAN 的 `full_pc`。
  - 两者都只来自实际执行的位姿，不使用任何模型生成的深度。
- **Comp@5cm(T)**：GT 顶点中，到观测点云最近距离 ≤ 0.05 m 的比例，分母是全部框内顶点。
- **Comp distance(T)**（cm）：全部框内 GT 顶点到观测点云最近距离的平均值，不截断，与论文定义相同。另报中位数作为参考。
- **可见归一化的次口径**：分母改为"可观测顶点"，即在任一合法位姿（全部站点 × 3 个俯仰 × 8 个方位）的完整深度反投影 5 cm 以内的顶点。
  - 这是 C_obs 可见性归一化的对应物；
  - 用冻结传感器渲染全部位姿得到，属于 M 协议。
- **报告时刻**：T = 10、20、50、100，加上完整曲线（每步）。

## 5. 统计与报告
- **聚类单位**：房屋。11 个房屋聚类 bootstrap，10,000 次，seed 0；方括号为 95% CI。
- **配对差**：同一 episode 上各组相对 V2K4 的差，另加 Frozen − 随机参考、V2K4 − 随机参考。
- **其他**：每个房屋、每条 episode 的完整表，异常 episode 单列。
- **随机参考的特殊性**：它带有"优先未访问位姿"的规则，DistP1 规划器没有。这一差异照原样保留，不当作模型能力差异。

## 6. 隔离与产出
- 只读 T0.6 和 `W/preprocessing/` 的已有文件；输出只写 `MC/metric_replay/` 和 `MC/analysis/`。
- 反投影和全位姿渲染用 magician 环境（PyTorch3D，`env -u LD_PRELOAD`）。GT 网格只在评估侧使用。
- 报告：`reports/T0.8_metric_report.md`。
- **预注册范围**：上面列出的都是预注册指标。任何新增分析都标为"事后"。

---

# 协议 v1.1 增补：T0.9 MAGICIAN 官方 MP3D 复现（2026-10-24，在 T0.9 任何运行之前冻结）

## 7. 已核实的事实
**代码与权重**：
- 官方仓库 `e697009`。
- EXP3b 用过的 `MAGICIAN_clean_9521d69` 是你的 fork（`Jiahui0223/magician`）中的提交，不是官方版本。
  - 区别：多了一组环境变量控制的诊断钩子，以及 `occupied_pose`；
  - 本轮一律用官方 `e697009`。
- 官方 Google Drive 的 9 个权重文件（2023-05-16）下载到 `MC/weights/official_drive/`，sha256 见 `MC/weights/official_drive_sha256.txt`。
  - 与 EXP3b 所用的同名文件逐字节相同；
  - MAGICIAN 用 `pretrained_macarons.pth` + `trained_macarons.pth`（sha `34c5247f…`）。
- **环境**：现有 magician 环境，Python 3.9、torch 2.6.0+cu118、PyTorch3D 0.7.8。
  - 其中的 RaDe-GS 扩展来自同一子模块提交 `5f5cd3b`；
  - 官方代码可以直接 import，不安装任何东西。

**数据**：
- `mp3d_magician.zip`：来自 HuggingFace `sli016/Macarons-plus-plus`，104,848,109 B，sha256 `6ec157d5…`，与 HF 记录的 ETag 一致，zip 校验无误。
- 存放在 `/bsuhome/jiahuizhang/scratch/data/mp3d_magician/`（原始 zip 和解压后共 254 MB），与 MP3D 原始数据分开。
- **场景**：
  - `GdvgFV5R1Z5`、`gZ6f7yhEvPG`、`HxpKQynjfin`、`YmJkqBEsHnH` 属于官方 train；
  - `pLe4wQe7qrG` 属于 val，也是本轮 22 个楼层之一；
  - **没有场景属于封存的 test**。
  - 这 5 个场景在本轮只用于复现外部方法，不用于我们任何模型的训练、选择或调参。
- **网格**：来自同一份 `matterport_mesh`（贴图 UUID 相同，顶点和面数几乎一致），但经过再处理。
  - 字节不同，面数相差 0–1%，`HxpKQynjfin` 的顶点数不同；
  - 坐标为 y 向上，并按场景均匀缩放，非米制。外接框比例为：`pLe4wQe7qrG` 每米 0.270 个 OBJ 单位，`GdvgFV5R1Z5` 每米 0.588 个。
  - 所以不能直接复用我们的 MP3D 文件。

**论文协议与发布代码协议的差别**：

| 项目 | 论文（Table 1 / 4） | 发布代码（README 的 mp3d 设置） |
|---|---|---|
| 动作 | 无人机：6 DoF；轮式：前进 6.5 cm、转 10° | 场景自带的 6 DoF 网格（如 10 × 7 × 10 个位置 × 5 个俯仰 × 8 个方位），每次移动到相邻网格点 |
| 步数 | 无人机 5,000 步；轮式 1,000 或 2,000 步 | `n_poses_in_trajectory = 150`：150 次网格移动，每次 1 帧（`n_interpolation_steps = 1`） |
| 图像 | 无人机 1200 × 680，60° × 90° | 1200 × 680（README 要求手改） |
| 起点 | 每个场景 5 条轨迹 | 每个场景 `settings.json` 里只有 1 个起点 `[3, 3, 3, 2, 3]` |
| 指标 | Comp.(%)（5 cm）和 Comp.(cm) | `evaluation_lmdb.py` 只算覆盖率曲线前 100 个值的 AUC 和最大值，没有 Comp 实现 |

- 论文的 5,000 步是连续小步的运动步数，沿用 ActiveGamer / NARUTO 的无人机设定；发布代码的"位姿"是网格移动。**两者不能互换**，也不把 150 解释成 5,000。
- **结论**：用发布代码无法严格复现论文的 MP3D 协议。本阶段的目标是"发布代码协议"的可复现基线；论文数值（无人机 96.83% / 2.11 cm）只作为核查参照，不据此调整任何参数。

## 8. T0.9 运行设置（冻结）
**配置**：放在 `configs/t09_official/`，官方仓库的文件一个不改。用绝对路径传给官方入口 `test_magician_planning.py -c`。
- 测试配置：官方 `test_in_default_scenes_config.json` 只改三项：
  - `dataset_path` 指向本次运行的数据工作副本；
  - `test_scenes` 为 5 个 mp3d 场景；
  - `lmdb_dir_name` 指向本次运行的输出目录。
  - 其余不变：`trained_macarons.pth`、`beam_width = 10`、`beam_steps = 10`、`test_resolution = 0.05`、`use_perfect_depth_map = true`、`compute_collision = false`、`numGPU = 0`（配合 `CUDA_VISIBLE_DEVICES`）。
- MACARONS 配置：官方 `macarons_default_training_config.json` 只按 README 的 mp3d 设置改三项：`image_width = 1200`、`image_height = 680`、`n_poses_in_trajectory = 150`。

**数据与权重**：
- 每次运行有独立的数据工作副本：`MC/official_reproduction/<run>/data/<scene>/`，里面是指向原始文件的符号链接。tester 写出的 `test_memory_0/` 帧只落在工作副本里。
- `MC/external/MAGICIAN/weights` 是指向 `MC/weights/official_drive` 的符号链接。

**重复次数**：
- 发布代码不设随机种子（proxy 点等），重复运行之间会有差异。
- 每个场景 × 唯一的起点 × **3 次重复**（rep0–rep2），配置完全相同。全部报告，不挑选。

**冒烟**：在 `GdvgFV5R1Z5` 上做一次 `n_poses_in_trajectory = 5` 的冒烟，只验证能运行、能写出 LMDB、记录耗时和显存。冒烟的输出不计入结果。

## 9. T0.9 指标
**发布代码口径**（与官方 `evaluation_lmdb.py` 相同，覆盖率已除以 visibility_ratio = 1.0）：
- 覆盖率曲线（151 个值）；
- 前 100 个值的 AUC（trapz，不除以长度，与官方脚本一致）和最大值；
- 另报全部 151 个值的最终覆盖率和归一化 AUC。

**Comp@5cm 与 Comp distance**（按论文文字定义，换算成米）：
- **米制换算**：每个场景的均匀缩放 s（每米多少 OBJ 单位）和平移，由官方网格与我们 MP3D 网格的外接框确定，轴对应 (x, y, z)_官方 ↔ (x, z, ±y)_我们。
  - 两种符号都试，选顶点最近邻距离中位数 < 1 cm 的那种；
  - 都达不到就不报告米制指标，只报覆盖率。
- **GT 点**：官方网格中被面片引用、且位于 `settings.json` 场景框内的顶点。
- **观测点云**：
  - 主口径：在执行位姿（LMDB 里的 `X_cam_history` / `V_cam_history`）上，用官方相机设置和官方网格重新渲染 1200 × 680 的完整深度，取 z < `sensor_range` 的全部像素。标为 replayed / rendered；
  - 次口径：LMDB 里保存的 `points`，即官方 `full_pc`，累积的 10% 采样点；
  - 检查：保存的 `points` 到重渲染点云的最近邻距离，中位数必须 < 1e−3 OBJ 单位，否则只报次口径。
- 指标定义与 T0.8 相同：Comp@5cm = 距离 ≤ 5 cm 的顶点比例；Comp distance = 平均最近距离（cm）。

**成本**：每条轨迹的墙钟时间和峰值显存（L40S）。

**报告**：`reports/T0.9_magician_reproduction.md`。

---

# 协议 v1.2 增补：T0.10 同协议对照（2026-10-24，在 T0.10 任何正式 episode 之前冻结）

## 10. 接口测试（冻结前已完成）

| 编号 | 内容 | 结果 |
|---|---|---|
| E1 | **原生模式等价**（`tests/t10_native_equiv.py`）。在官方 `pLe4wQe7qrG` 上用同一种子和确定性算法比较：A 为官方 `compute_magician_trajectory`；B 为官方采集 + `adapters/magician_core.py` + 用官方网格邻居实现的候选生成器。beam 10 × 10 和 1 × 1 各 5 个位姿 | **逐位相同**：执行位姿、X / V 历史、覆盖率曲线全部一致 |
| E2 | **模拟器副本等价**（`scripts/t10_sim.py`，`replay:V2K4`）。按 T0.6 存档的动作重放 `8194nk5LbLH_L1_S2` | 101 帧服务记录和覆盖率曲线**逐位相同** |
| E3 | **冒烟**：同一 episode 12 步，MAG-B / MAG-G / V2K4-NV | 全部跑通，guard 拒绝 0 次，回退 0 次，选择都合法。每步决策约 6.6 / 1.9 / 0.8 s |

- 冒烟后的一项修正：MAG-G 和 MAG-B 改为共用种子（公共随机数），见 §11。冒烟输出不计入结果。

## 11. MAGICIAN 组（G3 / G4）

**组别**：
- **G3 MAG-G**：官方 MAGICIAN，beam 宽 1、深 1，即单步选点（EXP3b 的 magician_greedy 约定）。
- **G4 MAG-B**：官方 MAGICIAN，发布的 beam search，宽 10、深 10，每步重新规划（N_f = 1）。

**不变的部分**：
- 官方权重（`trained_macarons.pth`）；
- 官方默认配置 `macarons_default_training_config.json`。其中 456 × 256、60°、sensor_range 70（室内 10 m）、zfar 750 与 Task 0 的冻结传感器相同；
- 占用场到想象高斯（occ > 0.5，尺度 0.7154 / 2）、新颖度、增益公式，以及深层的占用碰撞检测（阈值 1 运行单位 = 14 cm）；
- 每 20 步重填一次表面；
- 官方的有效性规则：只保留真实轨迹没访问过的位姿，都访问过时才保留全部。
- 以上全部通过 `magician_core.py` 调用官方函数，E1 已证明与官方循环逐位等价。

**适配的部分**（`adapters/station_env.py`、`adapters/magician_planner.py`）：
- **观测**：每步只读服务记录。
  - zbuf → depth = clamp(zbuf, 0.5, 750)（官方 `apply_perfect_depth_simple`）；
  - mask、R、T 照用；
  - 10% 采样点就是官方 `compute_partial_point_cloud` 的同一规则，T0.3 已逐位验证。
  - 相机状态按官方 `update_camera` 的记账方式更新，R 与服务记录核对误差 < 1e−5。
- **场景设置**：
  - 场景框 = 卡片楼层框（运行单位 / 10，即官方 settings 的单位）；
  - 竖直方向 `grid_w = 1`，与官方 mp3d 相同；
  - 水平方向 `grid_l / grid_h = max(1, round(边长_m / 1.40))`。1.40 m 是官方 5 个 mp3d 场景水平格子边长的中位数，见 `manifests/official_mp3d_scale.json`；
  - `cell_capacity = 1000`、`cell_resolution = 0.05`，与官方 mp3d 相同；
  - `n_proxy_points = 800,000`（官方默认），不随楼层大小调整。每层的 proxy 密度作为诊断量报告。
- **候选动作**：
  - 第 0 层 = 当前服务记录的合法列表，即环境规则：站点图移动（GT 视线）加原地转向。它对应官方第 0 层的 GT 线段检测，所以与所有组的观测预算和动作集合完全相同。
  - 更深层 = 原地转向（俯仰 ±1，在 3 档之内；方位 ±1，模 8）加保持朝向、移动到 3.0 m 以内的任一站点。只用卡片里的站点坐标，不用由 GT 视线检查生成的站点图，也不读网格。之后由核心做官方的占用碰撞检测。
  - 每层的候选按键 (站点, 俯仰, 方位) 的字典序排列，对应官方的 `torch.unique`。不包含"原地不动"；官方只在网格边界被截断时才会出现这种情况。
- **选择**：最佳 beam 的第一步映射回合法列表的下标。没有可选动作时用带种子的均匀回退（预计不会发生）。全部增益为 0 时，按官方行为选排在最前的候选。
- **种子**：两组共用。sha256("exp5-T0.10|MAG|<episode>") 用于场景初始化（proxy 点）；sha256("exp5-T0.10|MAG|<episode>|<t>") 在每步开始时设置。
- **隔离**：
  - 规划器是 magician 环境的子进程，只收到 {t, 服务记录路径}；
  - 模型加载之后装 LeakGuard：EXP3b 规则，加上 MP3D 数据集、`mp3d_magician`、覆盖率 / 网格缓存、标签、eval3d、metric_replay、`traj.json`、`.house`、`episode.json`；
  - 任何一次拒绝都判 INVALID。

## 12. 消融 A1：V2K4-NV（单独命名，不是 V2K4 的修改）
- 冻结的 T0.6 V2K4 规划器（分数不变），只在选择时加上 MAGICIAN 的有效性规则：候选集合为未执行过的合法位姿；都执行过时取全部。
- 在候选集合上取 DistP1 的 argmax，并列取最小下标；最大值 ≤ 0 时，在候选集合上带种子均匀回退（sha256 "exp5-T0.10|NV|<episode>|<t>"）。
- **目的**：在相同的回访规则下比较 V2 与 MAGICIAN 的预测和表示（A1 对 G3），并检验 V2 的停滞有多少来自"缺少回访限制"（A1 对 V2K4）。

## 13. Episode 与组
- **场景与预算**：T0.6 的 22 个楼层、相同起点、100 步、1 次重复；模拟器、传感器、种子、覆盖率和合法动作与 T0.6 相同（E2）。
- **已有组**：G0 随机参考、G1 V2K4、G2 V2K1，以及 Frozen、T1-C，直接用 T0.6 的冻结轨迹，不重跑。
- **新组**：G3 MAG-G、G4 MAG-B、A1 V2K4-NV，每组 22 条。
- **输出**：`MC/shared_protocol/closed_loop/<组>/<episode>/`。
- **失败处理**：崩溃的 episode 重跑 1 次（结果可复现）；仍失败就记为 failed 并报告，不剔除。

## 14. 指标
- **主要指标**：与 T0.8 完全相同，用同一份 `t08_metrics.py` 代码计算新轨迹。包括 Cov@10 / 20 / 50 / 100、AUC、Comp@5cm、Comp distance、可见归一化的 Comp、回访、停滞、区域切换、移动距离、时间。
- **MAGICIAN 诊断**：每步展开的节点数、高斯数、占用 / 新颖度 / beam 三部分耗时、峰值显存。
- **事后分析**（标为事后）：beam 选择的第一步不是单步增益最高候选的比例，以及这类步骤的真实覆盖增益（"跨越低即时收益"）。

## 15. 比较（三类独立报告，不合并成一个结论）
1. **完整系统**：V2K4 对 MAG-B。
2. **尽量消除规划器差异**：V2K4 对 MAG-G。另报 A1 对 MAG-G，两者的回访规则相同。
3. **MAGICIAN 长时程搜索的贡献**：MAG-G 对 MAG-B。
- **补充**：A1 对 V2K4，即回访规则本身的作用。
- **统计**：同一 episode 配对相减，按房屋做聚类 bootstrap（10,000 次，seed 0）；同时报告每条 episode 的结果和方向一致的 episode 数。
- **限制**：只有 11 个房屋、1 次重复，属于探索性结论。

## 16. 成本与计时
- 所有组都在 L40S 上运行。正式运行时 GPU 可能被同节点其他任务共享，所以墙钟时间只作参考。
- **独占计时**：正式运行结束后，在空闲 L40S 上，对 2 条 episode（最小层 `8194nk5LbLH_L1_S2`、最大层 `2azQ1b91cZZ_L0_S60`）的前 20 步，分别对 V2K4、MAG-G、MAG-B 测每步决策时间。只用于成本比较。
- **GPU-h**：墙钟时间之和。

## 17. 报告
- `reports/T0.10_fair_comparison.md`；
- `reports/T0.11_final_comparison.md`；
- 在 `TASK0_FINAL_REPORT.md` 末尾追加新章节，原文与 §13 不改。

---

# 协议 v1.3 增补：T0.12 网格动作空间对齐（2026-10-25，在 T0.12 任何正式 episode 之前冻结）

## 18. 目的与决定
- **目的**：T0.10 用的是 MP3D 全景站点图，每次移动平均 1.92 m，与 MAGICIAN 发布代码的网格动作不同。本阶段按发布代码的思路，把动作空间换成规则的三维相机网格，再比较 V2 与 MAGICIAN。
- **你 2026-10-25 的决定**：

| 项目 | 决定 |
|---|---|
| 动作空间 | 三维网格、6 自由度：每步沿一个轴 ±1 格，或俯仰 / 方位 ±1 档 |
| 步长 | 物理 1.0 m（官方 MP3D 场景换算成米后的中位数，范围 0.5–1.6 m） |
| 朝向 | 俯仰 5 档、方位 8 档，与官方 mp3d 设置相同 |
| 合法性 | 官方规则：两个相机中心之间的线段被 GT 网格挡住就不能走，不额外过滤位置 |
| 步数 | 150 步，同时报告 @100 |
| 规模 | 5 个已做可视化的楼层 × 2 个起点 = 每组 10 条 episode |
| 传感器 | 保持 456 × 256、60°、10 m |

- **说明**：官方代码没有固定的步长规则。自定义场景的教程只要求把场景缩放到与示例场景相近，再为每个场景手工设定 `pose_l / pose_w / pose_h`。论文的 5,000 步来自 NARUTO 的无人机设定：每步平移 ≤ 10 cm、旋转 ≤ 10°。发布代码没有实现这一设定。

## 19. 网格（`adapters/grid_env.py`，只依赖场景卡片）
- **相机框**：楼层框水平各内缩 0.25 m；高度从地面以上 0.5 m 到天花板以下 0.5 m。
- **每个轴的位置数** = max(1, round(长度 / 1.0 m))。
- **位姿公式**与官方 `Camera.__init__` 相同：
  - 位置 = x_min + (0.5 + i) · 长度 / n；
  - 俯仰 = −90 + 180 (1 + i) / 6，即 −60° 到 +60°；
  - 方位 = 45° · j。
- **邻居**：官方 `get_neighboring_poses`，下标截断到边界，方位循环。环境的合法动作中不包含"原地不动"。
- **合法性**：官方 `line_segment_mesh_intersection`，只在模拟器侧用 GT 网格判断，规划器看不到。转向永远合法。
- **起点**：每个楼层沿用 T0.3 选出的两个起点站点（第一个就是 T0.6 用过的）。吸附到距离最近、且从站点到该点的线段没有被挡住的网格位置；朝向为俯仰 0°、方位 0°。
- **5 个楼层的网格**（位置数为长 × 高 × 宽）：

| 楼层 | 位置数 | 实际步长 (m) |
|---|---|---|
| Z6MFQCViBuw_L0 | 33 × 4 × 24 | 1.01 / 0.88 / 0.99 |
| oLBMNvg9in8_L1 | 12 × 2 × 12 | 1.04 / 0.87 / 0.99 |
| QUCTc6BB5sX_L1 | 29 × 2 × 45 | 0.99 / 1.17 / 1.00 |
| 2azQ1b91cZZ_L0 | 43 × 2 × 26 | 0.99 / 0.99 / 1.00 |
| TbHJrupSAjP_L1 | 15 × 2 × 15 | 1.01 / 1.02 / 1.01 |

## 20. 组（`scripts/t12_sim.py`；输出 `MC/grid_protocol/closed_loop/<组>/<episode>/`）

| 组 | 内容 |
|---|---|
| G-RAND | 随机参考：在未执行过的合法位姿中均匀随机选（都执行过时在全部合法位姿中选），带种子 |
| G-V2K4 | 冻结的 T0.6 规划器 `t06_planner.py --method V2K4`，一行不改 |
| G-V2K4-NV | `adapters/v2nv_planner.py`，与 T0.10 相同 |
| G-MAG-G / G-MAG-B | `adapters/magician_planner_grid.py`：模型、核心、种子规则、LeakGuard 与 T0.10 相同 |

- **MAGICIAN 网格规划器**：
  - 第 0 层候选 = 服务记录里的环境合法列表；
  - 更深层 = 官方 `get_neighboring_poses`，包括被截断时出现的"原地"；
  - 每层都用官方的有效性规则（只保留真实轨迹没访问过的位姿，都访问过时保留全部），更深层再做官方的占用碰撞检测；
  - 场景网格与 T0.10 相同（水平每格 1.40 m，竖直 1 格）；
  - 种子 sha256("exp5-T0.12|MAG|<episode>")，每步加 "|<t>"，MAG-G 与 MAG-B 共用。
- **传感器、采样种子、C_obs 评估器和服务记录格式**都与 T0.6 / T0.10 相同。
- **每条 episode**：150 次移动，151 帧。

## 21. 接口测试（`tests/t12_grid_tests.py`，冻结前已完成）

| 编号 | 内容 | 结果 |
|---|---|---|
| E4 | 用官方 Camera 构造 `oLBMNvg9in8_L1` 的 11,520 个位姿，与 `grid_env` 逐个比较 | 位置最大差 4.4e−6 运行单位，角度 0；随机 200 个位姿的邻居与官方 `get_neighboring_poses` 全部相同 |
| E5 | 10 个起点的吸附 | 吸附距离 0.47–0.76 m；每个起点有 3–6 个合法移动 |
| E6 | 冒烟：`oLBMNvg9in8_L1_S36` 上 5 组各跑 12 步 | 只确认能跑通、guard 没有拒绝、选择合法；不计入结果 |

## 22. 指标
- **覆盖率**：C_obs，评估器和分母与 T0.6 相同。报告 Cov@10 / 20 / 50 / 100 / 150、AUC@100（前 101 个值 trapz / 100）和 AUC@150（151 个值 trapz / 150）。
- **其他**：Comp@5cm 和 Comp distance（与 T0.8 同一实现，扩展到 151 帧）、回访、停滞、移动距离、区域、时间，以及 MAGICIAN 的规划诊断。
- **描述性参照**：同 5 层第一个起点上 T0.6 / T0.10 的站点图结果，即站点图对网格的比较；不做配对检验。

## 23. 比较与统计
- 与 T0.10 相同的三类比较（V2K4 − MAG-B、V2K4 − MAG-G、MAG-B − MAG-G），加上消融（V2K4-NV − MAG-G、V2K4-NV − V2K4），在同一批 10 条 episode 上配对相减。
- 聚类单位为房屋（5 个），bootstrap 10,000 次，seed 0。**只有 5 个聚类，CI 会很宽**；报告每条 episode 的值和方向一致的条数。
- **失败处理**：崩溃的 episode 重跑 1 次，仍失败就写 `failed.json` 并报告。

## 24. 报告
- `reports/T0.12_grid_alignment.md`；
- 在 `TASK0_FINAL_REPORT.md` 末尾追加 §15。
