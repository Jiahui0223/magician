# UAVScenes 闭环协议 UAV-v0（EXP7，2026-11-10）

目标：把 UAVScenes 的 Terra 重建 mesh 切块，转换成 Macarons++ 的场景目录格式。转换后，冻结的室外闭环管线可以原样运行；Ours、Random、MAGICIAN 用同一个模拟器、同一个官方覆盖率评价器。这些都是确定性规则，对所有方法一致，不使用任何模型输出。按用户 2026-11-10 的决定：切块参考 Macarons++，先跑 Ours 和 Random。

## 1. 切块与坐标（`scripts/uavscenes/uav_build_tile.py`）
| 项 | 规则 | 依据 |
|---|---|---|
| 尺度 | s = 0.5 runtime 单位/米，即 1 单位 = 2 m。OBJ 单位 = runtime / 10 | Macarons++ 各场景约 1.2–3.5 m/单位；传感器量程 70 单位 = 140 m |
| 坐标系 | runtime 用 y-up：x = s(E − cx)，y = s(U − 地面)，z = −s(N − cy) | MAGICIAN 的 y-up 约定 |
| 原点 | 块中心 (cx, cy)；"地面" = 块内 2 m 最大高度图的 1% 分位 | — |
| mesh 范围 | 面片质心落在块框外扩 60 m 以内的保留（外扩部分是上下文，类似 Macarons++ 延伸到场景框外的地形） | 避免块边缘的视线直接看到"虚空" |
| 降采样 | 0.3 m 顶点聚类（取位置和颜色的均值），去掉退化面和重复面，使每块面数与 Macarons++ 相当（约 150 万） | 0.3 m 小于模型体素（约 0.4 m） |
| 纹理 | 每个面一个纹素，颜色 = 三个顶点颜色的均值（只影响 RGB，深度和覆盖率不受影响） | — |
| 场景框（GT 表面和覆盖域） | 水平方向 = 块的矩形。竖直方向：下沿 = 地面 0.5% 分位 − 2 m；上沿 = max(顶部 99.9% 分位 + 20 m, 下沿 + 最长边/3)，保证宽高比在 Macarons++ 范围内（bridge 为 5.6） | — |
| 相机框 | 水平方向 = 场景框；竖直方向从场景下沿 + 5 m 到场景上沿；位置步长约 10 单位（20 m）；5 个俯仰角（−60/−30/0/30/60°）× 10 个方位角 | 与全部 12 个 Macarons++ 室外场景相同 |
| 覆盖网格 | 每格约 20 单位；`cell_capacity` 1000；`cell_resolution` 0.05，即 eps = 1 单位 = 2 m | Macarons++ 官方值 |

## 2. 相机预筛、起点与可见率（`scripts/uavscenes/uav_tile_gpu.py`）
- **可达位置**：
  - 种子是顶层中竖直向上通往天空的线段不被遮挡的位置。
  - 从种子做 ±1 位置移动的广度优先搜索；每一步移动用官方线段碰撞规则判定。批量实现与官方 `line_segment_mesh_intersection` 抽查 300 条，结果一致。
  - 地下位置和建筑内部位置因此不可达。
  - 闭环中的合法性仍然只由模拟器的线段规则决定，与 Macarons++ 相同。
- **离表面距离**：取到 300 万个按面积加权的表面采样点的最近距离。
- **`occupied_pose.pt`**：标记不可达或离表面不足 5 m 的位置。只作记录，MAGICIAN 的测试循环不读这个文件。
- **起点**：
  - 候选：可达且离表面至少 10 m 的位置。
  - 用种子 sha256("EXP7-UAV-v0|starts|<tile>") 均匀抽取 10 个互不相同的位置。
  - 俯仰索引取 {0, 1, 2}（−60/−30/0°），方位索引取 0..9。
- **可见率 `visibility_ratio`**：
  - GT 点：场景框内（三个顶点都在框内）的面片上按面积加权采样 10 万个点，带种子。
  - 判定：一个 GT 点在 eps = 1 单位内有某个反投影点，就算可见。反投影点来自冻结传感器，取 z < 70、每个视角 10% 的像素样本，遍历所有可达位置的全部 50 个视角。
  - 可见点所占比例写入 `settings.json`，作用与 Macarons++ 官方给出的数值相同。

## 3. 运行（`scripts/uavscenes/uav_sim_out.py`，所有进程都设置 `MACARONS_PLUSPLUS_ROOT` = UAV 数据根目录）
- **模拟器与评价器**：冻结的室外链 `t24_out_sim.py`（底层是 EXP3b EpisodeSim），100 步，官方覆盖率除以 `visibility_ratio`。
- **Ours**：V3-T0@40K + D1。
  - 规划器子进程与 Table 1 的 V3-T0-D1 完全相同（D-v1、d1）。
  - `configs/d_config_uav.yaml` 只把 `pose_tables` 改为 UAV 切块的位姿表，位姿表由冻结的 `build_pose_tables.py` 生成。
  - `uav_hooks.install_frame_hook()` 对 `uav_*` 场景按冻结的 E-M2 规则计算 k_M（k_M = 场景高度 / 2.816）。该规则对 14 个 Macarons++ 场景逐一复现了冻结表。
- **Random**：冻结的 EXP3b `m_t7_sim.py --method random --seed 0`，与 CVPR Table 1 的 Random 行完全相同。
- **MAGICIAN**：之后用同样的数据根目录接入 `t24_mag_out_planner.py`。该文件把数据根目录写死了，需要用同样的 hook 方式重定向。

## 4. 零样本与 GT 边界
- T0 和 D1 全部冻结，不在 UAVScenes 上做任何训练、微调或调参。
- 切块按几何规则预先确定，不按覆盖结果挑选。
- GT mesh 只给模拟器（渲染、线段合法性）、可见率计算和官方覆盖评价器使用。规划器的 LeakGuard 只允许读取 `settings.json`。
- 许可证为 CC BY-NC-SA 4.0：切块文件留在 `work_dirs/`，不入库、不再分发。

## 5. 切块清单
| 切块 | 来源 | ENU 框（m） | 面数 | 场景框（runtime） | 相机网格 | 说明 |
|---|---|---|---|---|---|---|
| uav_HKisland_S | HKisland | E 0–260，N −200–60 | 1,560,550 | 130 × 48.8 × 130 | 13 × 5 × 13 × 5 × 10 | Pilot：块内有两座约 40 m 高的大型结构 |
