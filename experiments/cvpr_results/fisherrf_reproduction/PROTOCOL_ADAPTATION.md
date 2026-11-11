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
