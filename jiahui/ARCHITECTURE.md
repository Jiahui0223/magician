# MAGICIAN 代码 ↔ 架构 对照分析

> 目的：把代码和论文（arXiv 2603.22650）的 architecture 对应起来——哪段是 volume occupancy network、每步用哪个核心模块、整体 pipeline。
> 一句话定位：**MAGICIAN = MACARONS 的三个网络（深度+占据+可见性，预训练）+ 一个全新的"测试时规划器"（Imagined Gaussians + tree/beam search）。** 训练训的是 MACARONS 骨干；MAGICIAN 的创新全在推理规划里。论文摘要原话：occupancy network → Imagined Gaussians(3DGS) → volumetric rendering 算 coverage gain → tree-search 规划。

---

## 1. 三个核心网络（`macarons/networks/`，由 `Macarons.forward(mode=...)` 分发）

| 架构块 | 代码 | 输入 → 输出 |
|---|---|---|
| **Depth Network** | `ManyDepth.py`（`DepthDecoder`/`CostVolumeBuilder`/`PoseDecoder`），`mode='depth'` | RGB+相邻帧+位姿 → 深度图 → 反投影成**观测部分点云**。自监督（光度重投影），无需深度 GT |
| **Volume Occupancy Network** ⭐ | `SconeOcc.py: SconeOcc.forward`，`mode='occupancy'` | (部分点云 pc, 查询点 x, 相机历史球谐 view_harmonics) → 每点**占据概率**。多尺度邻域 Transformer(MSN)+MLP |
| **Coverage-Gain / Visibility Network** | `SconeVis.py`，`mode='visibility'` | (点+占据概率 dim=4, view_harmonics) → 可见性增益球谐。**MACARONS 的 NBV 网络；MAGICIAN 推理不用它**（见 §3） |

`MacaronsWrapper` 拆成 `macarons.depth` 与 `macarons.scone`(occ+vis)，配 `MacaronsOptimizer`(可分别 freeze) 分开优化。

## 2. 场景表示 / 记忆（`macarons_utils.py: Scene, Camera`）

- `surface_scene`：已观测表面点云　`proxy_scene`：体积代理点 + view_state(相机历史) + supervision_occ(深度图**空间雕刻**得到的占据监督)　`covered_scene`/`gt_scene`：算覆盖率
- 占据网络的输入(pc, view_harmonics) 来自 surface_scene / proxy_scene

## 3. MAGICIAN 创新：Imagined Gaussians + Beam Search（`testers/magician_planning.py: compute_magician_trajectory`）

1. 占据场 → **想象高斯**（占据>0.5 的代理点 → 高斯，opacity=占据概率）`L521-537`
2. **RaDe-GS 渲染**（`render_gaussian_depth`，子模块 `diff_gaussian_rasterization`）从任意候选相机渲染想象高斯
3. **Novelty**：先从所有已访问相机渲染，标记已看过的想象点 `L546-589`
4. **Beam Search**（`beam_width`×`beam_steps`）：扩展到合法邻居位姿→渲染→算新看到多少想象点=coverage_gain(`L692`)→保留 top-beam_width；带碰撞检测 `L591-720`
5. 选最优下一步移动，约 100 步
> 覆盖增益 = 想象高斯渲染出的新颖像素加权和，**不是** SconeVis 预测的可见性。这是 MAGICIAN 的核心 idea。

## 4. 整体 Pipeline（文件级）

```
test_magician_planning.py                      ← 推理入口(run.sh 跑的)
  └─ run_magician_test()  [magician_planning.py:742]
       ├─ setup_test()                          载入 dataloader + 模型(depth+scone) + memory
       ├─ open LMDB
       └─ for 每个 test 场景:
            ├─ setup_test_scene() / setup_test_camera()   gt/surface/proxy/covered scene
            └─ for 每个 start_position(5个):
                 └─ compute_magician_trajectory()   ← 核心循环(§3)
                      → coverage_evolution 存 LMDB
evaluation_lmdb.py                             ← 读 LMDB 算 AUC + Max/Final Coverage
```
每步循环：拍帧→[Depth]深度→部分点云→更新 surface/proxy(空间雕刻)→算覆盖率→[SconeOcc]占据场→想象高斯→[RaDe-GS]novelty→[Beam Search+RaDe-GS]选下一最佳视角→移动。

训练（`train_macarons_run.py → train_macarons.py: loop()`）：训 MACARONS 骨干（depth 自监督 + SconeOcc + SconeVis），MAGICIAN 复用这些训练好的网络。

## 5. ⭐ 关于输入模态：RGB-only vs 我们实际跑的"完美深度"（重要澄清）

**问题**：论文不是 RGB-D 吗？为什么还有 depth network、还要"生成"深度？

**答**：分两层，两者都对——

**(A) 方法设计上 = RGB-only**：MAGICIAN 基于 MACARONS（"…with RGB Online Self-supervision"），卖点就是**只用普通 RGB 相机**做主动建图（如无人机），**不需要深度传感器**。所以才有 depth network——它从 RGB **自监督预测**深度（ManyDepth：相邻帧+cost volume+位姿，光度重投影损失，无深度 GT）。"生成深度"正是 RGB-only 建图的关键能力。论文摘要也强调用的是 **pre-trained** occupancy network（感知网络预训练好，MAGICIAN 只管规划）。

**(B) 但我们跑的发布配置 = 完美深度（≈ RGB-D）**：
- 我们用的所有配置都 `use_perfect_depth=true`（test 是 `use_perfect_depth_map=true`）
- `apply_perfect_depth_simple` 用的是 `zbuf`——**渲染已知网格得到的 GT 深度**（clamp 到 [0.5,750]），不是预测深度
- `magician_planning.py` 推理代码**从不调用 depth 网络**（只在打印参数量时提了一下 `macarons.depth`）→ 深度网络被完全旁路
- README 原话：`use_perfect_depth` —— "If True, uses perfect depth maps rather than predicted depth maps. **Should be False.**"

**所以你的直觉对应的是发布实验设置**：我们复现的 MAGICIAN 实验用 GT 深度（等于 RGB-D），depth 网络没用上（我们的训练里 depth 也被 freeze、不训，深度网络由 `pretrained_macarons.pth` 提供）。

**为什么发布实验用完美深度？** MAGICIAN 的贡献是**规划器**（Imagined Gaussians + tree search），不是感知。用完美深度去掉深度估计误差这个干扰项，覆盖率/AUC 才纯粹反映规划质量。完整 RGB-only 管线（`use_perfect_depth=false` + 训练好的 depth 网络）是从 MACARONS 继承的通用能力；规划实验把它关掉了。

> 小结：depth network 是架构的一部分（为 RGB-only 服务），但我们复现的发布配置用 `use_perfect_depth=true` 假设深度已知（类 RGB-D），把它旁路了。

## 6. 代码定位速查

| 架构块 | 文件:函数 |
|---|---|
| 深度网络 | `networks/ManyDepth.py`；`utility/depth_model_utils.py: apply_depth_model`（推理被 perfect-depth 旁路）|
| **体积占据网络** | `networks/SconeOcc.py: SconeOcc.forward`；入口 `utility/macarons_utils.py:1571 compute_scene_occupancy_probability_field` |
| 可见性网络(仅训练) | `networks/SconeVis.py: compute_coverage_gain` |
| 场景/代理点/空间雕刻 | `utility/macarons_utils.py: Scene, Camera, update_proxy_supervision_occ` |
| 想象高斯 + RaDe-GS 渲染 | `testers/magician_planning.py: SimpleGaussianModel, render_gaussian_depth` |
| Beam search 规划 | `testers/magician_planning.py: compute_magician_trajectory (L591-720)` |
| 训练每步流程 | `trainers/train_macarons.py: loop()` |
