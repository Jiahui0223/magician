# FisherRF 复现 · 协议适配设计（PROTOCOL_ADAPTATION）

日期 2026-11-11。这是 Stage F0 的设计文档，**还没有写适配器代码**。下文引用格式为 `文件:行号`，`AM/` 指 FisherRF-active-mapping 的克隆，参见 `SOURCE_AUDIT.md`。

## 0. 原则

1. **打分必须是 FisherRF 本身**。我们不另写打分，而是直接调用官方的 `GaussianSLAM.track_rgbd`（在线 3DGS）和 `GaussianSLAM.pose_eval`（Fisher EIG），二者都在 `AM/models/SLAM/gaussian.py`，代码一行不改。不用 frontier、TSDF、可见性等启发式替代或混合。
2. **只替换导航胶水**，即 `SOURCE_AUDIT.md` 中 5、6 节的 B、C 层：Habitat、占据图、frontier、环采样、A*、原子动作。替换后：
   - 候选位姿 = 我们服务记录里的 legal 列表；
   - 每步只走一步。
3. **GT 边界不变**。GT 网格、覆盖率、碰撞检测、legal 列表的生成，都只存在于仿真端和评测端。规划子进程只能拿到当前视角的观测：深度，若 D1 选 (a) 再加上 RGB；以及已执行位姿和 legal 候选。
4. 其余协议要素一律沿用冻结版本，不做修改：
   - 起点、步数（outdoor 100 / indoor 150）、hover 规则；
   - 评测器：MAGICIAN 官方 coverage / C_obs；
   - JSON-lines 子进程协议。

## 1. FisherRF 每步需要的输入，以及我们的记录能否提供

服务记录的白名单见 `experiments/exp3b_macaronspp_transfer/m_common.py:166-172`。

| # | FisherRF 输入 | 用在哪里 | 我们的服务记录 | 结论 / 适配 |
|---|---|---|---|---|
| 1 | 当前视角 RGB，(H, W, 3) uint8 | 建图 loss（权重 0.5，`yaml:48-50`）；新高斯的颜色（`gaussian.py:92-94`）；间接影响 H，因为 ∂RGB/∂opacity 依赖颜色 | **没有**。outdoor 仿真端的 frame 里有 `rgb`（`m_t6_sim.py:83, 89`；MAGICIAN 纹理渲染，`macarons_utils.py:4208-4211`）；indoor 仿真端只做 rasterize，不加载材质（`mp3d_sim.py:84, 190-227`） | **需要在仿真端补 RGB**，见 §4 和决策 D1 |
| 2 | 当前视角深度，(1, H, W)，z-depth，0 表示无效 | 初始化、建图、新增高斯 | `zbuf`（256×456，-1 表示无命中）和 `mask` | 有。已在 CPU 上验证 PyTorch3D 的 zbuf 是 z-depth（§3.2）。换算方法见 §3.3 |
| 3 | 当前视角位姿 w2c，OpenCV 约定 | 写入 GT 位姿（`gaussian.py:636-644`） | `R`、`T`，PyTorch3D 行向量约定 | 有。约定转换见 §3.1 |
| 4 | 内参 K | 反投影；光栅器相机 | 记录里没有，由冻结的传感器常数推出 | 用 `fx = fy = 221.703`、`cx = 227.5`、`cy = 127.5`，已在 CPU 上验证（§3.2，决策 D6） |
| 5 | 候选位姿 c2w 列表 | `pose_eval(poses)` | `legal_R`、`legal_T`、`legal_center` | 有。用同一套转换；候选可以有俯仰，打分本身不限制位姿 |
| 6 | 历史关键帧（含 RGB-D 和位姿） | `compute_H_train`；建图时的窗口 | 规划子进程自己保存 3DGS 状态和关键帧；续跑时按顺序重放 svc 0..T-1，与 MAGICIAN 适配器做法一样（`t24_mag_out_planner.py:88-89`） | 有，但续跑**不能做到逐位一致**（§8 风险 R1） |
| 7 | 占据图、frontier、navmesh、habitat 的 pathfinder | 官方的候选生成和 A*（B、C 层） | — | **不需要**，用 legal 列表替代 |
| 8 | GT 网格、未来深度、覆盖率 | 官方只在评测时用 | — | **禁止进入规划端**，仍由白名单校验器把关（`m_common.py:179-192`） |

## 2. 规划子进程的设计（`fisherrf_planner.py`，F1 阶段再写）

在 `fisherrf` env 中运行，协议与现有子进程相同：
- 启动后输出一行 `{"ready": true, "info": {...}}`；
- 每步收到 `{"t", "svc"}`，回复 `{"t", "pick", "fallback", "decision_s", "timing": {"decision_s"}, "scores", "diag", "guard_denied"}`；
- 收到 `{"cmd": "end"}` 或 `{"bye": true}` 时回复 `bye`。

依据：`m_t6_sim.py:131-166`、`t24_mag_out_planner.py:66-107`、`t12_sim.py:176-194`。仿真端读取的字段见 `v2_sim.py:140-146` 和 `t12_sim.py:179-194`。

每步的伪代码（`AM` 只 import，不改）：

```
cfg = get_cfg_defaults(); cfg.merge_from_file(AM/configs/mp3d_gaussian_FR_eccv.yaml)   # 官方 FisherRF 配置
cfg 覆盖项（只有下面这些，见 §5）
slam = GaussianSLAM(cfg)                       # gaussian.py:383-437

on (t, svc):
    rec = MC.load_service_record(svc)          # 白名单校验；m_common 只依赖 numpy，可在 fisherrf env 里 import
    rgb = load_rgb_sidecar(svc, rec)           # §4；核对 R、T、t 与 rec 一致
    w2c = to_fisherrf_w2c(rec.R, rec.T, s)     # §3.1
    depth = to_fisherrf_depth(rec.zbuf, rec.mask, s)   # §3.3
    if t == 0:
        slam.init(rgb, depth, inv(w2c))        # 与官方 tester.py:243 相同
        slam.track_rgbd(rgb, depth, w2c)       # 官方把同一个初始观测再送一次（tester.py:253, 263-292），见 D10
    else:
        slam.track_rgbd(rgb, depth, w2c)       # GT 位姿 + 建图 + 加关键帧（gaussian.py:548-807）
    if n_legal == 0: pick = None (hover)       # 与其它方法相同
    else:
        C = [inv(to_fisherrf_w2c(legal_R[j], legal_T[j], s)) for j]   # c2w
        scores, _ = slam.pose_eval(torch.stack(C))   # 官方 EIG：Σ H_c ⊙ (H_train + 0.1)^-1（gaussian.py:1312-1335）
        pick = tie_break(scores)               # D5
```

**为什么一步打分就等价于官方的路径打分**：官方的路径总分 = `Σ(...)/len + 30·final_EIG`。路径只有一个动作时，point 项不累加（`(1+1) % 5 ≠ 0`），pose 项恒为 0（`tester.py:863-882`），于是总分是 final_EIG 的单调函数，按它取 argmax 就等于按 `pose_eval` 取 argmax。官方截断为 top-20 再做 A*，在我们这里没有影响，因为 legal 候选只有 4–10 个，全部参与打分。

## 3. 坐标、内参与单位

### 3.1 位姿约定

- 我们的记录：PyTorch3D 行向量约定 `X_view = X_world · R + T`（`m_common.py:11-12, 80-92`）。view 系为 +x 左、+y 上、+z 前；世界系为 MAGICIAN 运行时坐标（y-up）。
- FisherRF：OpenCV 相机系（+x 右、+y 下、+z 前），4×4 的 w2c 作用在列向量上（`gaussian.py:61-77`，`tester.py:63-69`）。Habitat 的世界系也是 y-up，与我们的运行时坐标一致。
- 转换公式。记 `F = diag(-1, -1, 1)`，场景缩放为 s（§3.4）：
  - `w2c_FR = [[F·Rᵀ, F·T / s], [0, 1]]`，`c2w_FR = inv(w2c_FR)`；
  - 自检：相机中心 = `-(F Rᵀ)ᵀ (F T) / s = -R T / s`，与 `m_common.camera_center`（`:89-92`）给出的 `-T Rᵀ` 一致（再除以 s）。F1 中对每条记录断言 `|c2w_FR[:3, 3]·s − cam_center| < 1e-4`。

### 3.2 内参（已在 CPU 上验证）

在 CPU 上用 PyTorch3D 0.7.8 实测 `FoVPerspectiveCameras(fov = 60)` 在 256×456 下的投影。探针脚本为 `/bsuhome/jiahuizhang/.claude/jobs/fde6a1ae/tmp/p3d_probe/probe_intrinsics.py`，结果：
- `transform_points_screen` 拟合得到 `fx = fy = 221.7026`，主点（以像素角为原点）为 (228.0, 128.0)。换成以像素中心为整数坐标的约定，就是 **cx = 227.5、cy = 127.5**。这等价于 `f = (1 / tan 30°) · min(H, W) / 2`、`cx = (W−1)/2`、`cy = (H−1)/2`。
- 光栅化一个正对相机的平面 z = 10，zbuf 恒为 10，说明 **zbuf 是 z-depth，不是射线长度**。
- 光栅化一个倾斜平面后，逐像素对比（u = 10, 227, 440）：
  - 上面的 PyTorch3D 内参预测与 zbuf 完全一致（11.08776 / 10.00226 / 9.12534）；
  - MAGICIAN 反投影表的约定（`m_common.py:73-77`：`f = S·(m−1)/2 = 220.836`，`cx = W(m−1)/(2m) = 227.109`）有 0.02–0.05% 的深度偏差，焦距差约 0.4%。

**建议（D6）**：FisherRF 用真实的 PyTorch3D 内参（221.703 / 227.5 / 127.5）。原因是 FisherRF 的建图要逐像素对比渲染结果和观测，而观测本身就是 PyTorch3D 光栅器的输出。MAGICIAN 表约定只用在我们的 10% 采样点上，与 FisherRF 无关。
- 写法：`Calibration.fx = fy = 221.7025`、`cx = 227.5`、`cy = 127.5`、`width = 456`、`height = 256`（`base_config.py:168-181`）。
- 官方 tester 要求正方形图像（`tester.py:127`），但我们不用 tester。`setup_camera` 分别处理 w 和 h（`recon_helpers.py:4-32`），所以非正方形应该可行。这一点作者没有测过，F1 中需要用渲染和观测的重合度来验证。

### 3.3 深度

- 转换：`depth_FR = zbuf / s`，满足 `mask & (zbuf < 70)` 的像素保留，其余置 0。这里的 70 是协议的传感器量程（`m_common.py:26`）；`m_t6_sim.py:92` 和 `mp3d_sim.py:239` 对 10% 采样点也用同样的规则。
- FisherRF 把深度 ≤ 0 视为无效：建图的 mask 为 `depth > 0`（`gaussian.py:195`）；新增高斯要求 `depth > 0.01`（`:308`）；初始化要求 `depth > 0.5`，即 10 × cell_size（`:471`）。
- 无命中的像素（天空、户外的远处）既不参与深度监督，也不参与颜色监督，因为两者共用同一个 mask（`gaussian.py:201`）。这与官方在 Habitat 里的处理一致。
- 是否保留 ≥ 70 的深度，见 D11。

### 3.4 尺度 s

FisherRF 中有几处使用绝对单位：
- means3D 的学习率为 0.001（`yaml:56`）；
- 初始化门限 0.5（`gaussian.py:471`）；
- 近平面 0.01、远平面 100（`recon_helpers.py:4`）；
- 剪大高斯的门限 0.1（实际不会触发）。

官方的工作区间是 Habitat 室内，单位为米，深度量程约 10 m。

| 协议 | runtime 单位 | 建议的 s | FisherRF 中的量程 |
|---|---|---|---|
| MP3D-150 | 1 m = 7 runtime 单位（`mp3d_sim.py:3-9, 31`） | **s = 7**，即米制、严格一致 | 70 / 7 = 10 m，与官方 Habitat 量级相同 |
| Macarons++ | OBJ × 10，非米制（`m_common.py:33`） | **s = 7**（D4）：传感器量程同样变成 10 个 FisherRF 单位，和室内及官方同一量级 | 10 |

备选：s = 10（OBJ 单位，量程 7）或 s = 1（runtime 单位，量程 70）。两者都会让 means3D 学习率、0.5 门限等相对场景的含义发生变化，需要用户决定。

## 4. 仿真端需要补的东西（只补观测通道，不改协议）

### 4.1 RGB 怎么传给规划端（D9）

**推荐 (b) 旁路文件**：仿真包装脚本在每个 `svc_*.npz` 旁边写一个 `svc_*_rgb.npz`，内容为 `{t, rgb: uint8 (256, 456, 3), R, T}`。规划端加载后核对 t、R、T 与 svc 完全一致，否则报错。
- 冻结的 `m_common.py` 白名单与其它方法的服务记录保持逐位不变。
- RGB 只来自当前视角，与深度来源相同，属于合法观测。

备选 (a)：在 `SERVICE_KEYS_OPTIONAL`（`m_common.py:169`）中加入 `"rgb"`。只需改一行，但这改动了冻结模块。

### 4.2 Outdoor（Macarons++）

- 不需要新的渲染。`EpisodeSim.observe` 已经载入了 MAGICIAN 的 frame（`m_t6_sim.py:83`），其中的 `frame["rgb"]` 与 zbuf 来自同一次 `capture_image` 渲染（`macarons_utils.py:4207-4252`，SoftPhong + 环境光 + 对比度调整）。
- 实现方式：仿照 `t24_out_sim.py` 写一个新的包装脚本。它覆盖 `VS.Planner`（`t24_out_sim.py:31-56`），对 `EpisodeSim.observe` 做猴子补丁以记下最近一次的 `frame["rgb"]`，然后在 `ask(t, svc_path)` 里先写旁路文件，再转发请求。
- `v2_sim.run`（`task1/scripts/v2_sim.py:100-150`）以及冻结的 `m_t6_sim` 都不改。

### 4.3 Indoor（MP3D-150）

需要**新增纹理渲染**。现在的 `mp3d_sim.Sensor` 只有 `MeshRasterizer`，网格以 `skip_materials=True` 载入（`mp3d_sim.py:84, 190-227`）。做法：
- 在包装脚本中，用 MP3D `matterport_mesh/*/*.obj` 自带的纹理，构造一个 PyTorch3D MeshRenderer。由于每个场景有多张贴图，要用 TexturesAtlas 或者先合并纹理。
- 使用与 zbuf **完全相同**的 R、T、RasterizationSettings，渲染出 RGB，与 zbuf 逐像素对齐。
- 包装方式与 EXP7 `cvpr_results/exp7/scripts/sim_in_d.py` 相同：t23 / t12 一律不改，只覆盖 `child()`。

风险：MP3D 贴图的显存占用和渲染时间，以及 OBJ 的多材质解析。

## 5. FisherRF 配置覆盖项（全部列出；其余保持 `mp3d_gaussian_FR_eccv.yaml` 原值）

| 键 | 官方值 | 适配值 | 原因 |
|---|---|---|---|
| `SLAM.Dataset.Calibration.{fx, fy, cx, cy, width, height}` | 400 / 400 / 400 / 400 / 800 / 800 | 221.7025 / 221.7025 / 227.5 / 127.5 / 456 / 256 | 我们的传感器（§3.2） |
| `map_every` | 10（`base_config.py:43`） | **1** | 我们每步都是一次大幅位姿跳变，每个新视角都应该建图（D2）。仓库里 `configs/global.yml:3-4` 也有 `keyframe_every: 1`、`map_every: 1` |
| `keyframe_every` | 4（`base_config.py:42`） | **1** | 同上 |
| `mapping.num_iters` | 60 | **60**（D2 (a)） | 与官方每次建图的迭代数相同 |
| `num_frames` | 2000 | 2000（不改） | 只决定相机参数数组的大小，大于 151 即可 |
| `workdir` / `run_name` | `experiments/GaussianSLAM` | 每个 episode 一个 scratch 目录 | GaussianSLAM 会写参数和图片（`gaussian.py:410-413, 767, 789-805`） |
| `use_wandb` | false | false | — |
| `report_iter_progress` | true | false | 只影响日志输出（`gaussian.py:741-750`），不改变参数；关掉可以省时间 |
| `explore.*`、`policy.*`、`path_*` | — | 不使用 | 属于 B 层 |

λ = 0.1、只用 means3D + opacity 的 H、建图的 loss 和学习率、剪枝、新增高斯的规则，全部保持官方实现。

## 6. GT 边界

| 一侧 | 持有的内容 |
|---|---|
| 仿真端（magician env） | GT 网格、纹理渲染、GT 线段碰撞（legal 列表）、覆盖率评测器、起点 |
| 规划端（fisherrf env） | 当前 svc，经白名单校验：zbuf、mask、R、T、cam_center、10% 采样点、已执行位姿、legal 候选；当前视角的 RGB 旁路文件；自己维护的 3DGS 和关键帧 |
| 规划端绝对不能拿到 | GT 网格、GT 点、覆盖率、未来位姿或观测（`m_common.py:170-172`）、frontier 或 oracle 增益 |

规划端还沿用 MAGICIAN 适配器的 Guard 机制（`t24_mag_out_planner.py:90, 101-102, 105`），禁止打开 GT 路径；`guard_denied` 非空时，仿真端判定该 episode 无效（`t12_sim.py:182-183`）。

## 7. 每步预算（按代码数调用次数；F0 不许用 GPU，没有计时）

**Hessian pass** 指一次 RGB 渲染加上 `backward_power = 2` 的反传（`gaussian.py:1509-1510`）。**建图迭代**指 RGB 渲染 + 深度/轮廓渲染 + 反传 + Adam（`gaussian.py:693-750`）。

| | 官方（每个 episode） | 我们 outdoor（100 步） | 我们 indoor（150 步） |
|---|---|---|---|
| 分辨率 | 800×800 = 640k 像素 | 456×256 = 117k 像素（官方的 18%） | 同左 |
| 建图 | 2000 / 10 = 200 次 × 60 = 12,000 次迭代 | 100 × 60 = 6,000 次（D2 (a)） | 150 × 60 = 9,000 次 |
| 关键帧 | 约 500 个（每 4 帧一个） | 第 t 步时有 t+1 个 | 同左 |
| H_train | 每次重规划 2×\|KF\|（`pose_eval` 一次，`tester.py:812` 再一次） | 每步 t+1 次，合计 Σ = 5,050 | 合计 11,325 |
| 候选 H | 每次重规划 ≤256 个，加上 ≤21 条 × ≤30 个动作 = ≤630 | 每步 n_legal 个，实测 5–9，合计约 700 | 实测 4–9、均值 7.7，合计约 1,160 |
| 重规划次数 | ≥ 2000/30 ≈ 67 次（turn、卡住时还会更多） | 100 次（每步一次） | 150 次 |

结论：在 D2 (a) 下，我们每个 episode 的 Hessian pass 和建图迭代次数**都不多于官方**，而且单次的像素量只有官方的 18%。真实墙钟时间要到 F1 冒烟测试才能测，计划在 1 个 L40S 上跑 1 个 episode 的前 10 步。若要和官方的总建图量对齐（D2 (b)），outdoor 每步 120 次、indoor 每步 80 次，两者总量都是 12,000。

## 8. 风险

- **R1 不可逐位复现**：光栅器反传使用浮点 `atomicAdd`（`backward.cu:1095-1137`），建图时的关键帧选择也是随机的（`gaussian.py:697`，`keyframe_selection.py:58`）。所以同一 episode 重跑，或从断点续跑，都无法逐位一致。对策：
  - 每个 episode 固定 torch / numpy 种子，种子由 scene、start 派生；
  - 不中断地跑完；
  - 记录重复运行之间的方差（可选）；
  - 续跑只作为备用手段，并在结果中标注。
- **R2 没有 frontier 引导**：官方的候选集来自 frontier 环采样（`astar.py:704-746`）。而 Fisher EIG 只对已经存在的高斯有响应：如果一个视角里没有高斯，它的 H 为 0，EIG 也为 0（`compute_Hessian` 不使用随机 frontier 高斯，见 `SOURCE_AUDIT.md` 8 节第 2 条）。因此纯 EIG 加 legal 邻居可能在"已经看完的局部"里打转，或者出现全零平局。这是方法本身的特性，不是适配错误，但会影响结果（见 D3、D5）。
- **R3 RGB 来源不同**：outdoor 是 MAGICIAN 的 SoftPhong 渲染，indoor 是我们新加的纹理渲染，都不是 Habitat 的渲染。H 依赖学到的颜色，因此纹理质量会影响得分。
- **R4 非正方形图像和户外尺度**：作者只跑过正方形图像和室内场景。户外场景里相机常在场景盒外（见 memory geometry-roi-audit），远处几何靠近量程上限，3DGS 的高斯规模和显存占用都需要在 F1 中观察。
- **R5 SLURM 信号**：`ClusterStateManager`（`AM/cluster_manager.py:11-65`）在 SLURM 作业里会接管 SIGTERM / SIGINT / SIGUSR1，并且可能执行 `scontrol requeue $SLURM_JOB_ID`。触发条件是 `SLURM_JOB_ID` 存在、QOS ≠ kd-high、作业名 ≠ bash。对策：在规划子进程的环境变量里设 `SLURM_JOB_QOS=kd-high`，这样它就不生效（只改环境变量，不改代码）。
- **R6 import 副作用与 open3d**：
  - import `models.SLAM.gaussian` 会连带 import habitat_sim / habitat / open3d / wandb / torchmetrics（`SOURCE_AUDIT.md` 6 节），并在克隆里写 `__pycache__`。对策：设 `PYTHONDONTWRITEBYTECODE=1`（`.gitignore` 本来也忽略它）。
  - **open3d 在 fisherrf env 中装不上**（ENVIRONMENT.md B1），所以按原样 import `gaussian.py` 会报 ModuleNotFoundError。`gaussian.py` 里 open3d 只被 import、从未使用。
  - 此外 `eval_helpers.py:19` 在 import 时就 `.cuda()`，所以 import 本身就需要 GPU（B2）。
  - 以上都要在 F1 前由用户定夺，见 D8。
- **R7 每次建图存一张 PNG**（`gaussian.py:1129-1131`）：IO 开销小，接受，或者把 workdir 放到 /tmp。
- **R8 indoor 纹理渲染**是新增的仿真端代码，需要单独验证：RGB 与 zbuf 逐像素对齐，渲染不影响 coverage 评测，且 zbuf 与原仿真逐位一致。

## 9. 需要用户决定的事项

| # | 问题 | 选项 | 建议 |
|---|---|---|---|
| D1 | RGB | (a) 仿真端补 RGB：outdoor 直接用现有 frame，indoor 新增纹理渲染；(b) 只用深度的 FisherRF（颜色设为常数、RGB loss 权重为 0），**不忠实**，只能作消融 | (a) |
| D2 | 每步建图迭代数 | (a) 每步 60 次，与官方每次建图相同；(b) 与官方总量对齐，outdoor 120 / indoor 80；(c) 完全照搬官方节奏（map_every 10、keyframe_every 4），整个 episode 只建图 10–15 次，不推荐 | (a)，F1 再测时间 |
| D3 | 候选集 | (a) legal 邻居 + 纯 EIG；(b) 自行复刻官方 frontier 引导，需要新写占据图和 frontier，属于胶水而非 FisherRF 核心，工作量大且容易引入偏差 | (a)，并在论文中注明 |
| D4 | 尺度 s | 两个协议都用 7 / outdoor 用 10 / outdoor 用 1 | 都用 7 |
| D5 | 平局和空 legal 列表 | 平局：argmax 取最低下标（`m_common.argmax_lowest`），或在并列最高中按种子随机选；空列表时 hover（pick = None） | 并列最高中按种子随机选（种子 sha256("fisherrf\|scene\|start\|t")），并记录平局次数 |
| D6 | 内参约定 | PyTorch3D 实测值（221.703 / 227.5 / 127.5），或 MAGICIAN 表的约定（220.836 / 227.109 / 127.5） | PyTorch3D 实测值 |
| D7 | λ | 官方 `pose_eval` 写死为 0.1，路径和评测里是 1e-6 | 保持 0.1，即官方终点打分，不改 |
| D8 | 环境 / open3d | env 已经装好：真实的 habitat-sim 0.2.4 来自 aihabitat 频道，conda-forge 上没有。剩下的问题是 **open3d（B1）**，可选：(a) 在适配器进程中放一个空的 `open3d` 桩模块，官方代码不改，而 open3d 在核心路径上从未被调用；(b) 用 apptainer 容器加 pip open3d 0.18；(c) 从源码编译 open3d | 建议 (a)，影响最小、可审计：在适配器里明确记录"open3d 是桩模块"，并断言核心路径上没有任何 `o3d.` 调用。**需要用户批准** |
| D9 | RGB 通道 | 旁路文件，或修改白名单 | 旁路文件 |
| D10 | 第一帧 | 与官方 tester 一样，同一观测先 init、再 track 一次；或者只 init，此时 t = 0 没有关键帧，`pose_eval` 会因 H_train 为 None 而报错 | 照官方做法 |
| D11 | 量程外深度 | 置 0，与协议的 70 量程一致；或保留 zbuf（远到 750） | 置 0 |

## 10. 后续阶段（每一阶段都要等用户批准）

- **F1**：写适配器和两个仿真包装脚本，做冒烟测试。
  - outdoor 跑 1 个场景 × 1 个起点 × 10 步，indoor 跑 1 个层 × 1 个起点 × 10 步；
  - 只确认能端到端运行，并记录每步耗时、高斯数量、显存；
  - 检查 §3.1 的位姿自检、RGB 与深度对齐、渲染与观测的 PSNR。
- **F2**：Dev 子集的闭环。
- **F3**：outdoor 14×5、indoor 22×2 的全量运行，接入 Table 1 / EXP7 的同协议主表。

---

## F1b · FisherRF-full：官方规划循环在 Macarons++ 位姿网格上的适配（2026-11-11）

**用户决定**：保留 FisherRF 的 frontier 目标和路径评估。目标是复现 MAGICIAN 论文 Table 2 的 FisherRF 行（15 场景 AUC 0.546 / Final 0.786）。D3a（只对一步邻居打分）只作为消融。
实现：`scripts/fisherrf_full_planner.py`，用 `fisherrf_sim_out.py --planner fisherrf_full_planner.py` 运行。

**参考依据**
- MAGICIAN 论文脚注："We adapt the released FisherRF code to outdoor scenes by modifying its frontier selection and adjusting its action space"；正文："selects viewpoints along the frontier and generates a set of shortest paths from the current pose. It then evaluates these paths using Fisher Information"。
- MAGICIAN 仓库的历史提交 e4e08fe：
  - `macarons/testers/fish_planning.py` 并**不计算 Fisher 信息**。它在整个位姿空间采样 300 个位姿，用 MACARONS 的覆盖增益打分，再在位姿网格的位置上用 Dijkstra 规划路径，用 GT 网格做碰撞检测，然后沿路径行进。
  - `frontier_planning.py` 中实际生效的代码用的是 GT 表面点，不能放在规划端；被注释掉的设计是：proxy 点的三态占据（未知 / 空 / 占据，由深度 carving 得到）→ 采样朝向未知点的位姿 → 在位姿网格位置上做 Dijkstra（与占据点做碰撞检测）。
  - 我们只采用这个三维结构：三维占据、在网格位置上求最短路、动作空间换成 settings 网格。**打分全部来自 FisherRF 官方代码**，没有用 MACARONS 增益，也没有用 GT。

**不变的部分**：官方 GaussianSLAM 每步 `track_rgbd`（60 次迭代）、官方 `pose_eval`（λ = 0.1）、D12（docstring init）、D13、RGB 旁路文件、open() 守卫、尺度 s = 7。
**规划端输入**：只有当前步的服务记录、RGB 旁路文件，以及该场景的 settings.json（相机网格和场景盒，守卫安装前读一次；MAGICIAN 规划器也读这个文件）。

| 官方（`AM/` = FisherRF-active-mapping 8b196cd） | 本适配 | 说明 |
|---|---|---|
| 主循环：动作队列为空才重规划（`tester.py:337-386`） | 队列为空，或队首位姿不在仿真器的 legal 列表中时，重规划 | legal 列表是执行动作的唯一依据 |
| 无路径时转一次、`expansion += 1`；`expansion > 10` 抛 `NoFrontierError`，episode 结束（`tester.py:344-371, 535`） | 转一次方位角（azim + 1，不合法则 −1，再不行用种子随机）；`expansion > 10` 记事件后重置为 1，**继续跑满 100 步** | 协议要求固定步数 |
| 撞墙卡住：标障碍，清空队列（`tester.py:497-518`） | 当前位姿的官方邻居中，未访问却不在 legal 列表里的（GT 线段检测失败）→ 记为封锁边 | 仿真器先保留全部未访问邻居（`get_valid_neighbors`），再做 GT 检测，所以这一推断成立；只用 legal 列表这一观测 |
| `update_occ_map`（`astar.py:114-211`）：2D 三通道累加（未知先验 1 / 占据 ×100 / 射线上的空闲 = 1），逐格归一化后 argmax；高度窗 [0.1, 1.3] m；深度 < `pcd_far_distance` = 7 m；下采样 4 | 同一累加器放在 **3D 体素**上，并列时归为未知，与官方 argmax 一致；不加高度窗；`pcd_far_distance` = 7 FR 单位；**用全部像素的射线** | 2D 投影会把下采样 4 的射线压密，3D 中要用全分辨率才有相当的密度 |
| 栅格 `cell_size` = 0.05 m（yaml `explore.cell_size`） | 体素 0.05 FR = 0.35 runtime；范围为场景盒 ∪ 相机盒 | 与 MAGICIAN 的 800k proxy 点密度量级相同（barts 约 0.33） |
| `build_connected_freespace`（`astar.py:213-259`）：去掉高斯 > 25 的格、3×3 开运算、取最大连通域 | 去掉高斯 > 25 的体素、3×3×3 开运算、取最大 26 连通域 | 3D 推广 |
| `build_frontiers`（`astar.py:307-410`）：frontier = dilate(free) − free ∩ unknown，连通域面积 > 10，"combined" = count / (平均距离 + 20) | 同样的定义和打分，换成 26 连通、体素距离；centering 取簇均值 | — |
| `generate_candidate`（`astar.py:704-746`）：K = 256，在半径 [0.5, 1.0 × expansion] m 的水平环上，相机高度固定，只有 yaw 且朝向中心；腐蚀空闲（10×10）过滤，当腐蚀后像素数 > 40 时才启用（`astar.py:575-594`），没有候选则 expansion × 1.5 | 256 个点放在三维球壳上，半径 [0.5, 1.0 × expansion] FR；**吸附到最近的 settings 网格位置**，俯仰 / 方位取最接近"朝向中心"的网格角度；去重后再打分。过滤用 11³ 腐蚀；**只有当腐蚀空闲中至少有一个网格位置时才启用**（同时要求腐蚀后体素数 > 40），扩张循环最多 30 次 | 动作空间的适配：自由飞行相机没有"高度"，所以用球壳代替水平环 |
| 没有 frontier 时，在腐蚀空闲中随机取 1/4 的格子、随机 yaw（`astar.py:471-526, 597-603`） | 位置落在（腐蚀）空闲体素中的网格位置取 1/4，朝向随机 | — |
| `pose_eval`，取 top-20（`astar.py:615-662`） | 官方 `slam.pose_eval`，取 top-20 | — |
| `setup_start` + A*（`astar.py:261-305, 749-`）：障碍 = 占据格（膨胀 3×3）+ 高斯 > 50 的格，未知视为空闲；起点格置空 | **v3：在 5D 位姿图上做 BFS 最短路**（单位代价，即 Dijkstra）。邻接为官方 `get_neighboring_poses`（沿一个位置轴 ±1，或沿一个角度档 ±1）；平移需要线段畅通，旋转总是可行；**已执行过的位姿不可经过**，除非当前位姿的所有邻居都已执行过。（v2 只在位置上做 BFS，再补转向，见下文）边与障碍体素相交则不可走；障碍 = 占据体素（3³ 膨胀）+ 高斯 > 50 的体素，未知视为空闲；还要排除封锁边。**膨胀之后，把每个已执行相机位置周围 5³ 的体素清空** | 与 MAGICIAN 被注释掉的 frontier 设计一致（`generate_Dijkstra_path` 在位置网格上，与占据点做碰撞） |
| `action_planning`（`tester.py:904-1030`）：前进 0.05 m / 转 5°，最多 `planning_queue_size` = 30 个动作；终点转向目标 yaw；动作序列相同的路径去重 | 动作 = 沿路径逐格平移（朝向不变），再逐档转俯仰、方位到目标（方位走较短方向）；最多 30 个；相同序列去重 | 我们的网格可以在任意轴上平移而不必转向 |
| `plan_best_path`（`tester.py:763-893`）：最多 21 条路径；每个动作调一次 `compute_Hessian`；point_EIG = log Σ H/(H_path + `H_reg_lambda` = 1e-6)；每 `acc_H_train_every` = 5 个动作累加一次；pose_EIG = log det(I) = 0；总分 = total/len + `path_end_weight` = 30 × final_EIG；final_EIG 来自**排好序的 EIGs 与有效路径的 zip** | 公式和常量完全相同，从官方 cfg 读取，官方函数直接调用；**保留官方 zip 的错位**：某个目标没有路径时，后面路径配到的是排序列表中下一个 EIG | 官方 bug 照原样保留 |
| 空路径（目标就是当前位姿）：`total / len(curr_action)` 会除以 0（官方会崩溃） | 跳过空路径 | 避免崩溃 |
| 随机数：全局 torch / numpy | 候选采样用 sha256("fisherrf-full\|scene\|start\|t\|expansion") 的 numpy RNG | 只换随机源，便于复现 |

**每次重规划记录**：原因、frontier 统计（free / unknown / occ 体素数、簇数、所选簇大小和距离、中心）、候选（采样数、去重数、是否启用过滤）、top-20 的 EIG、路径（可达位置数、每条长度）、路径 EIG 的总分和 Hessian 次数、最优目标和路径长度，以及各阶段计时（synchronize 后的 frontier / candidates / pose_eval / paths / path_eig）。

**v1 试跑发现的两处修正（2026-11-11 16:20，v1 的 8 条 episode 已中止并移到 `full/_aborted_v1/`，不计入结果）**
1. **起点被围死**（`n_reachable_positions = 1`）。
   - 原因：官方 `get_pointcloud` 用 max-pool 后的 mask，深度却取 4×4 块左上像素。看到天空时，左上像素深度为 0 的块会在相机中心生成零尺度高斯。
   - 后果：高斯 > 50 的障碍规则再加 3³ 膨胀，把相机四周全部封住。
   - 修正：官方把机器人所在的 3×3 格设为空闲（`astar.py:125`），并在膨胀后清空起点格（`astar.py:290`）。3D 推广为：清空所有已执行位置周围 5³ 的体素。
2. **腐蚀空闲过滤永远满足不了**。
   - 官方的扩张循环（`astar.py:575-594`）在 2D 中，环半径变大后总能落进腐蚀空闲区。
   - 在稀疏的 3D 网格上，可能根本没有网格位置位于腐蚀空闲中，扩张多少次都满足不了。
   - 修正：只有当至少一个网格位置位于腐蚀空闲中时才启用过滤；启用后扩张循环一定能结束。
3. **v3：路径必须服从环境的动作规则**（2026-11-11 16:50，v2 的结果保留在 `full/outdoor/closed_loop/` 作为记录，不计入结论）。
   - 现象：v2 在 redeemer s0 / s1 上分别有 26 次和 53 次 `first_move_not_legal_turn`，表现为原地转圈、重规划后再转圈。
   - 原因：仿真器的 legal 列表在还有未执行邻居时，不提供已执行过的位姿（`get_valid_neighbors`）。v2 的路径常常以"保持朝向平移到一个已执行位姿"开头，这一步不合法。
   - 修正：在 5D 位姿图上规划，并排除已执行位姿。这相当于官方 A* 在机器人自身的动作空间里规划；规划端知道这条环境规则，也知道自己的执行历史。
4. **F1c 变体（只做过 20 步冒烟，没有试跑；2026-11-11 暂停）**。
   - 设 `FRF_FRONTIER_MODE=magician` 时，frontier 和候选环节换成 MAGICIAN e4e08fe 中 `sample_valid_poses_in_space_for_frontier` 的设计：在整个位姿空间均匀采 256 个位姿，保留位置未被占据、且视野（传感器量程内）含未知体素的；未知体素用规划端自己的，最多抽样 5 万个。
   - 之后仍用官方 `pose_eval` 取 top-20；路径和路径 EIG 与 v3 相同。默认（不设这个变量）就是 v3。
5. **v4 / f1c4：打分前的可达性过滤**（2026-11-12，用户恢复后的预注册变体；用 `FRF_REACH_FILTER=1` 开启，默认关闭，所以 v3 / F1c 仍可原样复现）。
   - 官方依据：`GaussianSLAM.global_planning`（`AM/models/SLAM/gaussian.py:1244-1257`）只对"`follower.pathfinder.is_navigable(target_pos)` 为真、且 `follower.find_path(target_pos)` 能找到路径"的候选计算 Hessian 和 EIG，其余候选跳过。
   - 本适配在 5D 位姿图上做同样的检查，在 `pose_eval` **之前**丢掉三类候选：
     - 已执行的位姿：环境动作规则，除非当前位姿的所有邻居都已执行；
     - 位置落在障碍体素上（膨胀后的占据体素，或高斯 > 50 的体素）；
     - 从当前位姿 BFS 不可达。
   - BFS 图与 v3 求路径用的是同一个，每次重规划只建一次（`reach_graph`），之后的 top-20 路径也从它取出。
   - 打分（`pose_eval`，λ = 0.1）、top-20、路径 EIG 的公式与常量、主循环都**不变**。
   - 日志：每次重规划的 `info.cand.reach_filter` 记录 `n_before`、`n_after`、`drop_executed`、`drop_obstacle`、`drop_unreachable`。
   - 动机：v3 中 top-20 全部不可达，于是转一次（redeemer s2 55 次，s0 19 次），见 `PILOT_FULL.md` §4。
   - 两个预注册变体：
     - **v4** = v3 的官方 frontier 模式 + 可达性过滤，输出根 `full/outdoor/closed_loop_v4/`；
     - **f1c4** = `FRF_FRONTIER_MODE=magician` + 可达性过滤，输出根 `full/outdoor/closed_loop_f1c4/`。
   - 两者都通过共享 GPU 队列（`work_dirs/cvpr_results/gpu_queue`）运行，任务由 `scripts/frf_queue_tasks.py` 生成。
