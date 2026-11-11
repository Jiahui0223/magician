# FisherRF 复现 · Stage F0 源码审计（SOURCE_AUDIT）

日期 2026-11-11。本阶段只读代码、只用 CPU，没有用 GPU，也没有提交 Slurm 作业。克隆里的文件一个都没改（两个仓库 `git status --porcelain` 都为空）。
下文的路径缩写：
- `AM/` = `/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping`
- `FR/` = `/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF`

引用格式为 `文件:行号`。

---

## 0. 结论摘要

1. **FisherRF 打分的核心**很小，全部在 `AM/models/SLAM/gaussian.py` 和改造过的光栅器 `AM/thirdparty/diff-gaussian-rasterization-modified` 里：
   - 在候选位姿上渲染 RGB，反传时把"每像素对每个高斯的 Jacobian"平方后再累加，得到对角 Fisher 矩阵 H：每个高斯 4 个数，即 means3D 3 个、opacity 1 个；
   - 对历史关键帧求和，得到 H_train；
   - 视角得分为 `Σ H_cand ⊙ (H_train + 0.1)^-1`。
2. **在线 3DGS** 采用 SplaTAM 式建图：
   - 使用 GT 位姿，不做跟踪；
   - 每 10 帧建图一次，每次 60 次 Adam 迭代，loss = L1 depth + 0.5 × L1 RGB；
   - 新增高斯的来源是 silhouette 不足或深度误差大的像素。
3. **官方系统的其余部分**都依赖导航方式，可以替换，但不属于 Fisher 打分本身，包括：
   - 2D 占据栅格；
   - frontier 聚类；
   - 在 frontier 质心周围的环上采样 256 个候选；
   - 保留 top-20；
   - A* 路径与路径 EIG；
   - 离散动作（前进 0.05 m / 转 5°）；
   - 初始转 360°。
4. **官方实验**：
   - 场景为 Habitat 0.2.4 上的 Gibson 9 个、HM3D 5 个；
   - 2000 个原子动作，传感器为 800×800 RGB-D、90° 视场、GT 位姿；
   - 指标为随机视角上的 PSNR / SSIM / LPIPS / 深度 MAE，加上 2D 地面 fog-of-war 覆盖率（m² 和 %）。
5. **官方实验不是我们的协议**：
   - 官方仓库里没有 Macarons++（14×5，100 步）的任何代码，也没有 MP3D-150（22 层 × 2 起点，150 步）的任何代码；
   - MAGICIAN 论文里 FisherRF 在 Macarons++ 上的数字，不能从这两个仓库复现出来。
6. **复现路径**：保留第 1、2 点的官方实现（直接调用 `GaussianSLAM.track_rgbd` 和 `GaussianSLAM.pose_eval`），只把第 3 点换成我们的 legal 候选列表。详见 `PROTOCOL_ADAPTATION.md`。

---

## 1. 仓库、版本、子模块

| 仓库 | URL | commit | 子模块 |
|---|---|---|---|
| FisherRF（NeRF 侧） | https://github.com/JiangWenPL/FisherRF | `b74732812b295189f230a192418375f56cec3bd6`（2024-12-26，"Update README.md (#23)"） | `submodules/diff-gaussian-rasterization` 59f5f77e…；其中 `third_party/glm` 5c46b9c0…；`submodules/simple-knn` 44f76429…；`diff/third_party/glm` 4eb3fe1d…。全部解析成功 |
| FisherRF-active-mapping | https://github.com/JiangWenPL/FisherRF-active-mapping | `8b196cd22b1f4030f205778ad14b23fbd796c3b7`（2024-12-27，"Update README.md"） | `thirdparty/diff-gaussian-rasterization-modified/third_party/glm` 33b4a621…，解析成功；`thirdparty/simple-knn` 86710c2d…，**见下方说明** |

关于 `thirdparty/simple-knn`：
- `.gitmodules` 里的 URL 是 `https://gitlab.inria.fr/bkerbl/simple-knn`，不带 `.git`。用系统 git 1.8.3.1 做 `--recursive` 克隆时，返回 `HTTP code = 422` 而失败。
- 改用 `https://gitlab.inria.fr/bkerbl/simple-knn.git` 手工克隆到同一路径，并 checkout 到锁定的 commit 86710c2d（2024-07-18，"Compatibility"）。
- 之后 `git submodule status` 显示的就是锁定 commit。**不算 blocker。**
- 另外，active-mapping 的 Python 代码里没有任何地方 import `simple_knn`（全仓 grep 只命中 `thirdparty/simple-knn/setup.py`）。README 要求安装它（`AM/README.md:65`），但运行时用不到。

其它说明：
- active-mapping 仓库只有 2 个 commit，FisherRF 有 8 个。
- FisherRF 远端还有一个 `fix-10` 分支，我们用的是 `main`。
- 机器可读版本见 `source_versions.json`。

## 2. 许可

- `FR/LICENSE.md`：Gaussian-Splatting License（Inria / MPII），只允许非商业研究和评测使用。GitHub API 显示 license 为 `NOASSERTION`。
- `AM/` 仓库根目录**没有 LICENSE 文件**，GitHub API 显示 license 为 `null`。它包含的两个目录 `thirdparty/diff-gaussian-rasterization-modified/LICENSE.md` 和 `thirdparty/simple-knn/LICENSE.md` 都是 Inria 的 Gaussian-Splatting License。
- 结论：用于学术对比实验没有问题；如果要再分发改动过的代码，需要遵守 Inria 许可。我们的适配器只 import，不改也不分发它们的代码。

## 3. 仓库结构（与我们相关的部分）

**`AM/`**

| 文件 | 内容 |
|---|---|
| `main.py` | 入口；只跑 `scenes_list[0]`（`AM/main.py:37`） |
| `tester.py` | `NavTester`：Habitat 主循环、规划、评测 |
| `models/SLAM/gaussian.py` | `GaussianSLAM`：SplaTAM 式 3DGS，包含 `compute_Hessian`、`compute_H_train`、`pose_eval` |
| `models/SLAM/utils/*` | SplaTAM 的辅助函数（loss、相机、关键帧选择、剪枝） |
| `planning/astar.py` | `AstarPlanner`：占据图、frontier、候选生成、A* |
| `configs/base_config.py` | yacs 默认配置 |
| `configs/mp3d_gaussian_FR_eccv.yaml` | FisherRF 主实验配置 |
| `configs/mp3d_gaussian_FR_frontier.yaml`、`configs/mp3d_gaussian_UPEN_fbe.yaml` | 两个基线的配置 |
| `datasets/dataloader.py` | `HabitatDataScene`，负责 Habitat 配置和场景路径 |
| `visualization/habitat_viz.py` | fog-of-war 覆盖率 |
| `thirdparty/diff-gaussian-rasterization-modified/` | **FisherRF 改造过的光栅器**，`backward_power` 参数 |
| `scripts/{gibson,mp3d}.sh` | 跑实验的脚本 |

**`FR/`**（NeRF 侧，不是 active mapping）

| 文件 | 内容 |
|---|---|
| `active_train.py` | 主训练脚本 |
| `active/H_reg.py` | `HRegSelector`：NeRF 侧的 Fisher 选视角 |
| `diff/` | `modified_diff_gaussian_rasterization`，使用 `atomicAddSquare` |

## 4. FisherRF 真正的核心（A 层，复现时必须原样保留）

### 4.1 Fisher 信息 H（对角近似）

`GaussianSLAM.compute_Hessian(rel_w2c)`（`AM/models/SLAM/gaussian.py:1463-1527`）的做法：
1. 用候选的 w2c 把高斯中心变换到相机系（`:1482-1487`），构造 means3D / colors / rotations / opacities / scales 的 rendervar（`:1497-1504`）。
2. 渲染 **RGB**，`Renderer(raster_settings=self.cam, backward_power=2)`（`:1509`），然后 `im.backward(gradient=ones*1e-3)`（`:1510`），也就是每个像素、每个通道的上游梯度都是 1e-3。
3. 改造后的反向核 `renderCUDAFused` 对每个像素、每个高斯，把局部梯度做 `powf(·, grad_power)` 再 `atomicAdd`。依据：
   - `AM/thirdparty/diff-gaussian-rasterization-modified/cuda_rasterizer/backward.cu:1093-1137`；
   - 由 `rasterizer_impl.cu:392-433` 调用；
   - `power` 由 `diff_gaussian_rasterization/__init__.py:154-157,87,120` 传入。

   因此在 power=2 时，得到的是"逐像素 Jacobian 的平方和"，即对角 Fisher / Laplace 近似。
4. 返回 `H = [∂/∂means3D (3), ∂/∂opacity (1)]`，形状 (N, 4)（`:1512-1514`）。颜色、尺度、旋转的平方梯度虽然也算了，但被丢弃。
5. `pose_H` 恒为 `torch.eye(6)`（`:1527`）。
6. H 只来自 **RGB 渲染**，深度不参与。但 ∂RGB/∂opacity 依赖高斯颜色，而颜色是从观测 RGB 学来的，所以 **RGB 观测会间接影响 H**。

### 4.2 EIG 视角得分

- `compute_H_train()`（`gaussian.py:1296-1306`）：H_train = Σ_{关键帧} compute_Hessian(关键帧 est_w2c)。**每次调用都从全部关键帧重新算**，因为建图后高斯会变。
- `pose_eval(poses)`（`gaussian.py:1312-1335`）：对每个候选 c2w 计算 `view_score = Σ cur_H ⊙ reciprocal(H_train + 0.1)`（`:1314-1315, 1323, 1327`）。正则项 **λ = 0.1 写死在代码里**，得分不取对数。每个候选还会多调一次 `render_at_pose`，结果没有使用（`:1325`）。
- 路径层面的得分在 `NavTester.plan_best_path`（`AM/tester.py:830-893`）：
  - 沿 A* 动作序列逐步算 `point_EIG = log Σ cur_H ⊙ (H_path + H_reg_lambda)^-1`，其中 `H_reg_lambda = 1e-6`（`configs/mp3d_gaussian_FR_eccv.yaml:12`，`tester.py:856-861`）；
  - 每 5 个动作把 cur_H 累加进 H_path，并计入 `path_point_weight × point_EIG`（`tester.py:869-871`；`acc_H_train_every = 5`，`path_point_weight = 1.0`，见 `configs/base_config.py:15-18`）；
  - `pose_EIG = log det(I6) = 0`（`tester.py:863`），所以 `path_pose_weight` 这一项恒为 0；
  - 最终得分 = `total / len(actions) + 30 × final_EIG`（`path_end_weight = 30`，`yaml:18`，`tester.py:879-882`）。`final_EIG` 就是 4.2 第二条中 pose_eval 的终点得分。
- **单步特例**：路径只有一个动作时，`(1+1) % 5 ≠ 0`，point 项不累加；pose 项为 0。于是总分 = 30 × final_EIG，与 pose_eval 的排序完全一致。这一点是我们离散一步协议的依据，见 `PROTOCOL_ADAPTATION.md`。

### 4.3 在线 3DGS 更新（SplaTAM 式）

- **初始化** `GaussianSLAM.init`（`gaussian.py:439-496`）：
  - 第一帧中深度大于 `10 × cell_size`（0.5 m，`gaussian.py:471`，`cell_size` 见 `yaml:108`）的像素，按 `downsample_pcd = 4`（`yaml:29`）反投影成高斯；
  - 尺度按 projective 规则取 depth / f（`gaussian.py:83-86`）；
  - 不透明度 logit 设为 0（`:124`）；
  - 默认各向异性（`base_config.py:56`）。
- **每帧** `track_rgbd(color, depth, gt_w2c)`（`gaussian.py:548-807`）：
  - `use_gt_poses: true`（`yaml:95`）→ 直接写入 GT 位姿（`:636-644`），**不做跟踪优化**；
  - 当 `(time_idx+1) % map_every == 0` 时建图：`map_every` 默认 10（`base_config.py:43`），eccv yaml 没有覆盖；
  - 建图第一步 `add_new_gaussians`（`:287-378`）：对 silhouette < 0.5，或"渲染深度 > 观测深度 且 深度误差 > 10 × 中位数"（`yaml:45`）的像素，反投影成新高斯；
  - 建图第二步：`mapping.num_iters = 60`（`yaml:59`）次 Adam。每次从重叠关键帧（≤ `mapping_window_size - 2 = 30`，外加最后一个关键帧和当前帧，`base_config.py:45`，`gaussian.py:672-681`）里随机取一帧，loss = L1 depth × 1.0 + L1 RGB × 0.5（`yaml:48-50`，`gaussian.py:693-750`）；
  - 学习率：means3D 0.001、log_scales 0.01、opacity 0.05、rgb 0.0025、rot 0.001（`yaml:51-58`）。**means3D 的学习率是绝对场景单位**，对尺度敏感；
  - 剪枝：每 40 次迭代删除 opacity < 1e-4 的高斯（`yaml:61-69`，`slam_external.py:196-220`）。删"大高斯"要求 `it ≥ 100`，而每次建图只有 60 次迭代，所以永远不会触发；
  - 关键帧：当 `(time_idx+1) % keyframe_every == 0` 时加入，`keyframe_every` 默认 4（`base_config.py:42`，`gaussian.py:770-786`）。
- **每次建图的副作用**：
  - 都会存一张 `mapping/mapping_iter_XXXX.png`（`gaussian.py:767, 1129-1131`）；
  - 当 `time_idx % checkpoint_interval == 0` 时保存参数，`checkpoint_interval = 1000`（`yaml:14`，`gaussian.py:789-805`）。
- **第一帧的处理**：官方 tester 先对初始观测调用 `slam.init`（`tester.py:232-243`），然后**在同一位姿上再取一次观测**，送进第一次 `track_rgbd`（`tester.py:253, 263-292`）。也就是说，初始观测被送了两遍：第一遍做 init，第二遍作为第 1 帧。
- **相机约定**：
  - OpenCV 相机系（x 右、y 下、z 前），Habitat 的 c2w 要右乘 `diag(1,-1,-1,1)`（`tester.py:63-69, 237`）；
  - 光栅器相机由 `setup_camera(w, h, K, I, near=0.01, far=100)` 生成（`models/SLAM/utils/recon_helpers.py:4-32`），view matrix 是单位阵，高斯在 Python 里手动变换到相机系；
  - 内参来自 `SLAM.Dataset.Calibration`，默认 `fx = fy = cx = cy = 400`，图像 800×800（`base_config.py:168-181`）；
  - tester 断言图像必须是正方形（`tester.py:127`）。

### 4.4 NeRF 侧（`FR/`）作对照

- `HRegSelector.nbvs`（`FR/active/H_reg.py:24-90`）做的事情相同：
  - 对训练视角渲染并 `backward(ones)`，平方累加由 `FR/diff/cuda_rasterizer/backward.cu:750-751, 1097`（`atomicAddSquare`）完成；
  - 参数是除 rotation 以外的全部高斯参数（`FR/active_train.py:321`，`filter_out_grad = ["rotation"]`）；
  - 得分 = `Σ H_cand ⊙ (H_train + reg_lambda)^-1`，`reg_lambda = 1e-6`（`FR/active_train.py:315`）；
  - 选多个视角时用贪心，每选一个就把它的 H 加进 H_train（`H_reg.py:79-88`）。
- 区别：这是在固定候选集上的离线选视角。数据为 NeRF-synthetic / Mip-NeRF360，例如 `FR/scripts/blender_seq1.sh`，30k 迭代，每次加 1 个视角，直到 20 个（`FR/active/schema.py`）。**不是 active mapping**，与我们的协议无关，只用来交叉核对公式。

## 5. 官方 active-mapping 系统中的规划部分（B 层：FisherRF 系统专用，但不是 Fisher 打分本身）

主循环见 `AM/tester.py:284-533`，`policy.name = astar_greedy`（`yaml:6-8`）。

1. **开局**：72 次左转，共 360°（`tester.py:1113-1116`）。
2. **占据图** `AstarPlanner.update_occ_map`（`planning/astar.py:114-211`）：
   - 用深度做 2D 栅格，`cell_size = 0.05 m`；
   - 高度窗口为 [0.1, 1.3] m（`base_config.py:35-36`）；
   - 自由空间通过沿射线采样和画线得到。
3. **frontier**（`astar.py:307-423`）：
   - frontier = 自由空间边界 ∩ 未知区域；
   - 取连通分量，面积 > 10；
   - 用 "combined" 规则选一个簇：`count / (mean_dist + 20)`（`astar.py:366-389`，`yaml:112`）。
4. **候选生成** `generate_candidate`（`astar.py:704-746`）：
   - 在所选 frontier 簇的质心周围（centering，`astar.py:574-575`，`yaml:109`），按半径 ∈ [0.5, 1.0 × expansion] m 在环上随机采 `sample_view_num = 256` 个位姿（`yaml:105-107`）；
   - 朝向指向质心，高度为相机高度，**只有 yaw，没有俯仰**；
   - 用腐蚀后的自由空间过滤（`astar.py:584-594`）；
   - 没有 frontier 时，在自由空间里随机采位姿（`astar.py:597-603`，`471-526`）。
5. **打分与截断**：用 `slam.pose_eval` 打分（`tester.py:789-792`，`astar.py:615-618`），只保留 **top-20**（`astar.py:658-662`）。
6. **A* 与动作**：
   - 对每个目标做 A*，并离散成最多 `planning_queue_size = 30` 个动作（`tester.py:904-1029`，`yaml:8`）；
   - 动作为前进 0.05 m、转 5°（`yaml:16-17`，`slam_external.py:44-65`）；
   - 最多评估 21 条路径（`tester.py:835`）；
   - 取路径 EIG 最大的一条（4.2 节），把它的动作入队，执行完再重新规划（`tester.py:340-397`）。
7. **随机高斯**：`explore.add_random_gaussians: true`（`yaml:111`）会在 frontier 上生成随机高斯（`astar.py:557-561, 669-691`）。但 `compute_Hessian` 完全不用 `random_gaussian_params` 这个参数（`gaussian.py:1464-1527`），`pose_eval` 也传入 False（`:1301, 1323`）。**所以这些随机高斯对得分没有任何贡献。**
8. **碰撞**：撞墙卡住后，在占据图里标障碍并清空动作队列（`tester.py:497-518`）。

这一层依赖"连续导航 + 原子动作"的设定，在我们的离散位姿栅格协议里没有对应物，要换成 legal 候选列表。需要注意：官方的候选集是由 frontier 引导的，这是一个有实质影响的系统设计，见 `PROTOCOL_ADAPTATION.md` 的决策 D3。

## 6. Habitat / 数据集胶水（C 层）

- **依赖**：habitat-sim 和 habitat-lab 均为 v0.2.4（`AM/README.md:57`），官方推荐 docker `wen3d/agslam:latest`（`AM/README.md:31-48`）。
  - `tester.py` 顶层 import 了 `habitat_sim`、`habitat` 和 `models.UPEN`。后者经过 `planning/ddppo_policy.py:6-7`，import 了 `habitat_baselines` 和 `gym`。
  - `models/SLAM/gaussian.py` 本身也在顶层 import 了 `habitat_sim`（`:26`），以及 `open3d`（`:7`，文件里实际没用到）。它还经由 `eval_helpers → test_utils` 间接 import 了 `habitat.utils.visualizations.maps`（`test_utils.py:10-11`）。**所以即使只用打分核心，也必须能 import habitat_sim / habitat / open3d**，或者在适配器里预先注入桩模块。
  - 另外，`models/SLAM/utils/eval_helpers.py:19` 在模块顶层执行 `LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()`。所以 **import gaussian.py 本身就需要可见的 GPU**；首次 import 还会下载 alexnet 权重。2026-11-11 的 CPU import 检查（ENVIRONMENT.md E6）证实了这一点。
- **场景与传感器**（`datasets/dataloader.py:76-139`）：
  - 场景路径：`<root>/data/scene_datasets/<scenes_dir>/<scene>.glb`；
  - 传感器尺寸 800×800（`:101-104`），hfov 取 habitat-lab 默认 rgbd_agent 的值（`:175`）；
  - 内参为 `f = (W/2) / tan(hfov/2)`（`:207-211`）；
  - 必须存在 pointnav 的 episode 文件 `data/datasets/pointnav/{mp3d,gibson}/v1/val/val.json.gz`（`:124-133`）。它只用来按场景过滤 episode，但缺了会直接报错。
- **起点**：`self.test_ds.sim.sim.reset()`（`tester.py:229`），之后不再设置位姿，所以起点由 habitat-sim 的 reset 决定，`sim.seed(0)` 见 `dataloader.py:118-119`。**仓库里没有显式定义起点**（未运行 habitat-sim，无法确认 reset 的具体采样方式）。
- **深度**：直接读 `sim.sim.get_sensor_observations()`，即 habitat-sim 原始观测（`tester.py:232-233`）。注释说深度在 [0, 1]、应乘 10（`tester.py:491`），但代码路径上没有做反归一化。只读代码无法确认实际深度单位；按 habitat-sim 的惯例应为米，**未运行验证**。

## 7. 官方实验具体做了什么

| 项 | 官方设定 | 依据 |
|---|---|---|
| 数据集 | Gibson 9 个场景（Greigsville、Denmark、Cantwell、Eudora、Pablo、Ribera、Swormville、Eastville、Elmira）；"HM3D" 5 个场景（DBjEcHFg4oq、mscxX4KEBcB、QKGMrurUVbk、oPj9qMxrDEa、CETmJJqkhcK） | `AM/README.md:5-9` |
| 数据集备注 | README 把 HM3D 的下载链接写成了 Matterport 主页。这 5 个 ID 不在我们本地 MP3D 的 90 个 scan 中（已核对 `~/scratch/data/matterport3d/v1/scans`），与 HM3D 一致。跑 HM3D 时代码走的是 `dataset_type = mp3d` 分支（`scripts/mp3d.sh`，`dataloader.py:84-86`） | — |
| 脚本 | `scripts/gibson.sh` 默认只跑 Ribera，其余场景被注释掉（`:5-6`）；`scripts/print_result.sh` 汇总 9 个 Gibson 场景 | — |
| 步数 / 动作 | `num_frames = 2000` 个原子动作（`yaml:13`）；前进 0.05 m，转 5°（`yaml:16-17`）；开局 360° 扫描（72 步） | — |
| 传感器 | RGB-D 800×800（`yaml:27-28`），fx = fy = 400，hfov 90°，GT 位姿（`yaml:95`） | — |
| 规划 | 5 节全部内容；每次重规划最多入队 30 个动作 | — |
| 指标（只在结束时算，`eval_every = -1`，`base_config.py:49`） | 在 2000 个随机水平位置中可导航的那些、随机 yaw 的视角上，算 PSNR / SSIM / LPIPS / 深度 MAE（`tester.py:552-573, 577-662`）；coverage(m²) 与 coverage(%) 为 2D fog-of-war（fov 90°、10 m，`habitat_viz.py:110-132`）∩ GT 可导航俯视图（`tester.py:668-683`） | — |
| 基线 | Frontier（`configs/mp3d_gaussian_FR_frontier.yaml`）、UPEN（`configs/mp3d_gaussian_UPEN_fbe.yaml`） | — |
| 预训练 | README 提供了"Pretrained Gaussians"的 Google Drive（`AM/README.md:83`），只用于从 `params*.npz` 续跑（`tester.py:246-250`）。我们的在线复现不需要 | — |

## 8. 官方代码中的不一致和问题（复现时照原样保留，不修复）

1. `pose_eval` 里 λ = 0.1 是写死的（`gaussian.py:1315`）；路径 EIG 和评测用的是 `H_reg_lambda = 1e-6`（`tester.py:599, 856`）。
2. 随机 frontier 高斯生成了却从未参与打分（5 节第 7 条）。
3. `pose_H` 恒为单位阵，所以 `pose_EIG` 恒为 0（`gaussian.py:1527`，`tester.py:863`）。
4. 评测时 `cur_H` 被乘了两次 `H_train_inv`（`tester.py:618-619`）。只影响结果文件名和可视化，不影响指标。
5. `pose_eval` 里每个候选多做一次渲染，结果没有使用（`gaussian.py:1325`），只是白白耗时。
6. tester 只支持正方形图像（`tester.py:127`）。
7. **两个仓库的光栅器 Python 包同名**，都叫 `diff_gaussian_rasterization`：FR 的 `submodules/` 里是 graphdeco 原版，AM 里是改造版。同一个环境只能装 AM 的改造版。FR 的改造版叫 `modified_diff_gaussian_rasterization`，不冲突。
8. README 用 `pip install -e .` 在源码目录里就地编译，会往克隆里写 build 产物。我们改为先复制再编译，见 `ENVIRONMENT.md`。

## 9. 官方实验与我们协议的差异（明确声明：**官方实验不是我们的协议**）

| 维度 | 官方 FisherRF active mapping | 我们的 Macarons++（MAGICIAN 协议） | 我们的 MP3D-150（EXP7） |
|---|---|---|---|
| 场景 | Gibson 9 个 / HM3D 5 个（Habitat） | Macarons++ 14 个场景 × 5 个官方起点 | MP3D 22 个 val 层 × 2 个起点 |
| 动作空间 | 连续导航；原子动作 0.05 m / 5°；navmesh 碰撞 | settings.json 的离散位姿栅格（位置 × 5 俯仰 × 10 方位）；官方邻居 + GT 线段碰撞（`m_t6_sim.py:119-121`） | 1 m 栅格 × 5 俯仰 × 8 方位；邻居 + GT 线段碰撞（`t12_sim.py:111-117`） |
| 步数 | 2000 个原子动作（含 72 步开局扫描） | 100 步 | 150 步 |
| 每步候选 | ≤256 个环采样 → top-20 → ≤21 条 A* 路径 | legal 邻居，实测 5–9 个（T7a 的 40 条记录） | legal 邻居，实测 4–9 个、均值 7.7（EXP7 前 60 个 episode，1200 步） |
| 传感器 | RGB-D 800×800，hfov 90° | 仅深度 zbuf，456×256；PyTorch3D fov 60°（短边），量程 70 runtime 单位（`m_common.py:22-27`） | 同左；1 m = 7 runtime 单位（`mp3d_sim.py:3-9, 31`） |
| RGB | 有 | 服务记录里**没有**；仿真端 frame 里有 `rgb`（MAGICIAN 纹理渲染，`m_t6_sim.py:83, 89`） | 服务记录里**没有**；仿真端只做 rasterize，网格用 `skip_materials=True` 加载（`mp3d_sim.py:84, 190-227`） |
| 位姿 | GT | 执行过的位姿 R、T（服务记录） | 同左 |
| 指标 | 新视角 PSNR / SSIM / LPIPS / 深度 MAE + 2D 地面 fog-of-war 覆盖率 | MAGICIAN 官方表面覆盖率，AUC@20 / @100 / Final | level-scale 表面覆盖率 C_obs，AUC@150 / Final |
| 代码 | 有 | **官方仓库没有** | **官方仓库没有** |

补充：
- MAGICIAN 论文附录表 6 / 表 7 给出了 FisherRF 在 Macarons++ 上的数字（AUC@100 0.542 / Final 0.766，均值的不一致见我们的审计）。这些数字我们只当参照列，见 `experiments/cvpr_results/OVERNIGHT_AUDIT.md:59` 与 `CVPR_RESULTS_AUDIT_AND_VISUALIZATION_REPORT.md:63, 117`。
- 这两个官方仓库里都没有生成这些数字的代码，所以 MAGICIAN 是怎么运行 FisherRF 的，**无法从源码确定**。

## 10. 不能确定的事项（需要 GPU 运行或作者确认）

- habitat-sim 0.2.4 在 `reset()` 后的起点采样方式，以及原始深度的单位（6 节）。
- 官方实验的单场景墙钟时间。仓库里没有给出；F0 不允许用 GPU，所以没有测。
- MAGICIAN 论文中 FisherRF 一列的运行设置（9 节）。
