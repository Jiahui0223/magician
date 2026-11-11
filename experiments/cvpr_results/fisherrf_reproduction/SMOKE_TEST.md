# FisherRF 复现 · Stage F1 冒烟测试（SMOKE_TEST）

日期 2026-11-11，节点 gpu114，只用 `CUDA_VISIBLE_DEVICES=3`（L40S）。没有提交 Slurm 作业，同一时刻只跑一个冒烟进程。
**GPU 3 同时被另一个实验的进程占用**：显存约 7 GB，利用率一直是 100 %。因此下文的耗时都是在共享 GPU 上测的，偏慢。
官方克隆没有改动：两个仓库 `git status --porcelain` 都为空，也没有写入 `__pycache__`。冻结脚本没有改动：exp5、exp3b、table1 目录下 `git status` 都为空。
本文所有数字都来自下文列出的运行和日志。

## 0. 结论

| 项 | 结果 |
|---|---|
| 1 规划子进程 + 仿真端包装脚本 | **完成**（§1） |
| 2a RGB-D 送到规划端 | **PASS**（outdoor / indoor） |
| 2b 3DGS 每步在线更新 | **PASS**：每步 `track_rgbd`，60 次建图迭代，加一个关键帧 |
| 2c legal 候选用真实 Fisher EIG 打分 | **PASS**：逐候选得分已记录；独立重算与 `pose_eval` 一致，H ≥ 0 |
| 2d pick 合法、闭环推进 | **PASS**：10/10 步 pick 都在 legal 列表里，没有 hover |
| 2e Table 1 官方覆盖率回放与存储曲线逐位一致 | **PASS**（两个 init 变体都通过） |
| 3 Indoor（MP3D，纹理 RGB）10 步 | **PASS**：纹理渲染的 zbuf 与传感器 zbuf 逐位相同 |
| 4 本文档 | 完成 |

在 F2 / F3 之前**需要用户决定的两件事**（详见 §5）：

- **D12（新）**：第一帧 `init` 的位姿参数。官方 tester 把 w2c 传进了文档写明为 c2w 的参数位，这是官方 bug。在冒烟里它影响很大：barts s0 的 cov@10 为 0.125，而按 docstring 语义传参时为 0.337。这只是单个 episode 的数字，不是结论。
- **D13（新）**：深度全部无效的帧会让官方 `track_rgbd` 崩溃。Table 1 中 dunnottar s1 / s2 / s4 和 redeemer s1 的起点在 t = 0 就是这种帧，所以对 FisherRF 来说是**必然**发生的。

## 1. 代码（全部新写，放在 `experiments/cvpr_results/fisherrf_reproduction/scripts/`）

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `fisherrf_planner.py` | fisherrf env | 规划子进程，JSON-lines 协议。直接 import 官方 `models/SLAM/gaussian.py` 的 `GaussianSLAM`，依次调用 `init` → `track_rgbd`（GT 位姿 + `add_new_gaussians` + 60 次建图迭代 + 关键帧）→ `pose_eval`（Σ H_cand ⊙ (H_train + 0.1)^-1，backward_power = 2）→ argmax。平局时用种子随机（D5）；legal 列表为空时 hover。打开文件守卫（guard）拒绝读取 GT、仿真端和评测端文件。每步写一行 JSON 日志 |
| `fisherrf_sim_out.py` | magician env | Macarons++ 闭环。`t24_out_sim → v2_sim.run → m_t6_sim.EpisodeSim` 原样运行，只在内存中注册 `FisherRF` 组；包装 `EpisodeSim.service_record` 以记下当前帧，在 `ask` 时于服务记录旁写 `svc_<s>_s<k>_t<t>_rgb.npz`（D9） |
| `fisherrf_out_cov.py` | magician env | Table 1 的覆盖率回放：`t25_out_cov.py` 原样运行，回放帧和 metrics 都写在本目录的 work_dirs 下，不碰 table1 / task2 |
| `fisherrf_sim_in.py` | magician env | MP3D 闭环。`t23_in_sim.main` 原样运行；`--group` 在内存中加入 `FisherRF`，`child` 替换为 fisherrf 子进程。新增纹理 RGB 渲染（§3），通过包装 `t12_sim.service_record` 写 `svc_t<t>_rgb.npz` |
| `stubs/open3d/__init__.py` | 只放在规划进程的 sys.path 上 | D8 桩模块：访问任何属性都会抛 `RuntimeError("open3d stub: called, not allowed …")`；规划端用 `open3d.__file__` 断言加载的是桩 |

配置：官方 `configs/mp3d_gaussian_FR_eccv.yaml`，只覆盖 PROTOCOL_ADAPTATION §5 列出的键，实际值记在每个 episode 的 hello 行：
- 内参 fx = fy = 221.7025、cx = 227.5、cy = 127.5，图像 456 × 256（D6）；
- `map_every = 1`、`keyframe_every = 1`、`mapping.num_iters = 60`（D2 (a)）；
- `workdir` / `run_name` 指向本 episode；`use_wandb = false`；`report_iter_progress = false`。

坐标与深度：
- `w2c = [[F·Rᵀ, F·T/7], [0, 1]]`，F = diag(−1, −1, 1)，s = 7（D4）；
- 深度 = zbuf / 7，只保留 mask 内且 zbuf < 70 的像素，其余置 0（D11）。

## 2. 运行命令与日志

工作目录为仓库根目录，`R=work_dirs/cvpr_results/fisherrf_reproduction/smoke`，`MG="env -u LD_PRELOAD -u PYTHONPATH CUDA_VISIBLE_DEVICES=3 /bsuscratch/jiahuizhang/envs/magician/bin/python"`。

```bash
S=experiments/cvpr_results/fisherrf_reproduction/scripts
# outdoor：主冒烟（官方 tester 原样调用 init），以及 docstring 变体
$MG $S/fisherrf_sim_out.py --scene barts --start 0 --moves 10 --root $R/outdoor/init_tester/closed_loop    --init-pose tester    --verify-every 5
$MG $S/fisherrf_out_cov.py --root $R/outdoor/init_tester/closed_loop    --out $R/outdoor/init_tester/metrics    --scene barts --start 0
$MG $S/fisherrf_sim_out.py --scene barts --start 0 --moves 10 --root $R/outdoor/init_docstring/closed_loop --init-pose docstring --verify-every 5
$MG $S/fisherrf_out_cov.py --root $R/outdoor/init_docstring/closed_loop --out $R/outdoor/init_docstring/metrics --scene barts --start 0
# indoor：EVAL_LEVELS[0]
$MG $S/fisherrf_sim_in.py --scan Z6MFQCViBuw --level 0 --start 6 --moves 10 --out $R/indoor/init_tester/closed_loop --init-pose tester --verify-every 5
```

| 运行 | episode | 规划端逐步日志 | 其它 |
|---|---|---|---|
| outdoor tester | `$R/outdoor/init_tester/closed_loop/episodes/ep_FisherRF_barts_s0_r0.json` | `…/planner_logs/pl_FisherRF_barts_s0_r0.steps.jsonl`（stderr 在同目录 `.stderr`） | 服务记录和 RGB 旁路文件：`…/svc/FisherRF_barts_s0_r0/`；官方建图 PNG：`…/planner_work/barts_s0/mapping/`；回放结果：`$R/outdoor/init_tester/metrics/FisherRF/FisherRF_barts_s0_r0.json`；`sim_run.log`、`cov_run.log` |
| outdoor docstring | `$R/outdoor/init_docstring/closed_loop/…`（结构同上） | 同上 | 同上 |
| indoor tester | `$R/indoor/init_tester/closed_loop/FisherRF/Z6MFQCViBuw_L0_S6/episode.json` | `…/planner_fisherrf.steps.jsonl`、`planner.err`、`planner.jsonl` | `rgb_zbuf_check.json`、`svc/`、`planner_work/…/mapping/` |
| 第一次失败的尝试 | `$R/outdoor/init_tester.failed_try1_nograd/` | — | 见 §4 第 1 条 |

## 3. 验证了什么（证据都来自上面的日志）

**RGB-D 送达规划端**
- 每步加载记录后，会核对旁路文件的 t、R、T 与服务记录逐位相等，并检查 shape (256, 456, 3)、dtype uint8，不符就报错。
- outdoor 的 RGB 来自 MAGICIAN `capture_image`，浮点范围约 [0.0005, 0.79]，均值 93–102 / 255。
- indoor 的 RGB 来自新写的纹理渲染，均值 91–114 / 255。
- 官方 `visualize_frame` 的 PNG 中，"GT Image" 一栏就是送进去的 RGB，可以与深度逐像素对照；indoor 的门框、镜框边缘对齐。

**indoor 纹理渲染（新增的仿真端代码）**
- 渲染用的就是传感器的几何（缓存的 `v, f`）。OBJ 由 PyTorch3D `load_obj` 读入，只用来提供每个面的 UV 和材质。
- trimesh 会拆分 UV 接缝（368,183 个顶点，而 OBJ 中为 174,932 个），并按材质重排面。所以按面质心做 KD 树匹配；其中 261 个面质心重复，改用 k 近邻解决。三个角点的最大误差为 1.9e-6 m。
- 纹理用 PyTorch3D 的 `make_material_atlas` 生成，每个面 4 × 4 texel，175 张 2048² 贴图逐张处理。在 GPU 上构建耗时 11.1 s。
- 相机、`RasterizationSettings` 与 `mp3d_sim.Sensor` 是同一个对象。着色仿照 MAGICIAN outdoor：SoftPhong + AmbientLights 0.85，不调对比度。
- **11 帧全部满足：命中 mask 相同，zbuf 最大差 0.0。**

**位姿约定自检**
- 当前位姿：`|c2w[:3, 3]·7 − cam_center|` 的最大值，outdoor ≤ 1.2e-6，indoor ≤ 2e-6。
- 候选位姿：同一量的最大值，outdoor ≤ 1.2e-6，indoor ≤ 4e-6。
- 阈值为 1e-4，超过就报错。

**非正方形图像与内参**
- 每步建图后，在观测位姿上渲染当前 3DGS 并与观测比较（只在 valid 像素上）。这个渲染不改动 slam 的任何状态。
- outdoor docstring 变体：PSNR 26.0–31.1，深度 L1 0.02–0.34 runtime 单位。
- outdoor tester 变体：PSNR 23.0–33.8。
- indoor：PSNR 21.9–25.4，深度 L1 0.09–0.39 runtime 单位，即 1.2–5.6 cm。

**在线 3DGS**
- 每步都调用 `track_rgbd`：关键帧数每步加 1，outdoor 和 indoor 都是 1 → 10。
- 建图 60 / 60 次迭代（stderr 中 tqdm 的计数）。
- 高斯数：outdoor docstring 7,296 → 21,992；outdoor tester 12,882 → 16,092，t ≥ 6 后不再增长；indoor 13,908 → 26,347。
- 参数里没有 NaN 或 inf 的 means / 颜色 / 不透明度。例外是 indoor 有 log_scale = −inf 的高斯，见 §4 第 5 条。

**真实 Fisher EIG**
- 得分直接取自官方 `slam.pose_eval(stack(c2w_legal))` 的返回值，逐候选记录在日志的 `scores` 字段，episode 里的 `planner.scores` 字段也有一份。
- 在 t = 0 和 t = 5，用官方 `compute_H_train` 和 `compute_Hessian` 独立重算 Σ H_c / (H_train + 0.1)：
  - 与 `pose_eval` 的最大相对差为 1.1e-7 到 2.1e-6，来自 atomicAdd 的浮点非确定性；
  - argmax 完全相同；
  - H 的最小值为 0，说明 backward_power = 2 的平方累加非负。

**pick 合法、闭环推进**
- outdoor 两个变体、indoor 共 30 步：pick 全部在 legal 列表里，ties = 0，没有 hover；executed key 每步都变化。
- 仿真端的冻结检查（非法 pick 即报错，以及 guard_denied）全部通过。

**GT 边界**
- 规划进程的 open 守卫在三次运行中 `guard_denied` 都为空。
- outdoor 的 bye 行显示：服务记录目录之外只打开过 matplotlib 字体和自己的建图 PNG。

**官方覆盖率回放（Table 1 流程，t25 原样运行）**

| 变体 | official_equal_stored | AUC@10 official / mono | cov@10 official / mono |
|---|---|---|---|
| outdoor tester | **True** | 0.0923 / 0.0930 | 0.1250 / 0.1279 |
| outdoor docstring | **True** | 0.1912 / 0.1927 | 0.3369 / 0.3408 |

indoor 冒烟：MP3D 的 C_obs 由 t23 在闭环内计算。cov 从 0.021 到 0.035（10 步），不需要单独回放。

## 4. 与官方代码路径的差异，以及保留的官方行为

1. **调用上下文**：官方 tester 的整个循环在 `@torch.no_grad()` 下运行（`tester.py:227`），`get_loss` 和 `compute_Hessian` 自己用 `@torch.enable_grad`。第一次尝试没有加 no_grad，`init` 报 "a view of a leaf Variable…"。改成与 tester 相同的上下文后正常。这是对官方用法的还原，不算偏差。
2. **替换导航层**（D3，已决定）：没有 360° 开局、占据图、frontier、A*、原子动作。每步对 legal 列表做一次 `pose_eval`，取 argmax。
3. **配置覆盖**：见 §1。另外每个 episode 的 numpy / torch / random 种子取 sha256("fisherrf|scene|start")；官方 tester 在 import 时固定用 0。
4. **open3d 桩**（D8）：运行中没有任何 open3d 属性被访问，否则桩会抛错。
5. **官方行为照原样保留（需要知情）**：
   - `get_pointcloud` 用 max-pool 后的 mask，但深度取 4 × 4 块的左上像素。块内只要有一个 valid 像素就会生成高斯，左上像素深度为 0 时，这个高斯落在相机中心，log_scale = −inf。indoor 从 init 的 63 个累计到 t = 9 的 389 个；outdoor barts 深度全部有效，所以是 0。没有导致 NaN。
   - 从未被关键帧看见的高斯 H_train ≈ 0，权重为 1 / 0.1 = 10，会产生极大的得分：outdoor tester 在 t = 9 有 7,399，indoor 在 t = 2 有 24,336，而且都被选中了。
   - `pose_eval` 对每个候选多做一次渲染，结果不用；每次建图存一张 PNG（每步 0.4–0.7 s）。都保留了，耗时计入。
6. **测量用的附加物**（不影响状态，也不消耗随机数）：
   - 实例级计时包装：`compute_H_train`、`visualize_frame` 外面加了 synchronize + 计时；
   - 渲染与观测的对比（render check）；
   - `--verify-every 5` 的重算。F2 / F3 应设 `--verify-every 0`。
7. **半像素**（记录，未改）：
   - SplaTAM 的 `setup_camera` 把点投影到 fx·X/Z + cx − 0.5，而 `get_pointcloud` 反投影用的是 (u − cx)/fx，两者在官方代码里本来就差 0.5 px；
   - 官方 Habitat 配置的 cx = W/2 = 400，相当于以像素角为原点。我们按 D6 用 227.5，与官方习惯差 0.5 px。
8. **indoor RGB 是新的仿真端渲染**（R3）：每面 4 × 4 texel 的纹理比原始 2048² 贴图模糊，PNG 中可见块状纹理。如需更清晰可改 `--atlas 8`，显存约 250 MB。
9. **规划端代码在 outdoor tester 运行之后有过一次修改**，用于 indoor：
   - 新增 `--svc-name` / `--scene-key` 参数，默认值与原来相同；
   - 收到 `{"bye"}` 也会结束；
   - 应答的 `timing` 中多一个 `decision_s` 字段；
   - 守卫多禁止 indoor 的 `episode.json` 和 `planner.jsonl`。

   对 outdoor 行为没有影响。docstring 和 indoor 两次运行用的是最终版本。

## 5. 需要用户决定 / blocker

**D12 第一帧的 init 位姿参数**
- 事实：
  - `GaussianSLAM.init(color, depth, pose)` 的文档写的是 pose = c2w（`gaussian.py:446`），函数内部取 `w2c = inv(pose)`；
  - 官方 tester 却传入 `w2c_t`（`tester.py:243`；`get_cam_transform` 返回的是 c2w，见 `datasets/util/utils.py:86`）；
  - 结果是初始高斯被放到错误位置：outdoor 中位误差 11.1 runtime 单位，indoor 119 runtime 单位（约 17 m）；
  - 这里纠正一下：PROTOCOL_ADAPTATION.md §2 的伪代码 `slam.init(rgb, depth, inv(w2c))` 注明"与官方 tester.py:243 相同"，这一说法不对。
- 冒烟结果（barts s0，各 1 个 episode）：
  - tester 方式：AUC@10 0.092，cov@10 0.125；t ≥ 6 后高斯不再增加；
  - docstring 方式：AUC@10 0.191，cov@10 0.337；
  - indoor 的 tester 方式下，t = 2 出现 24,336 的异常得分并被选中，很可能来自错位的初始高斯。
- 选项：
  - (a) `--init-pose tester`：照搬官方，即当前默认值，保留官方 bug；
  - (b) `--init-pose docstring`：按函数文档的语义传参，与 PROTOCOL_ADAPTATION 原设计一致，在论文中注明。
- 建议：(b)，并在附录说明官方 tester 的这个错误。**需要用户批准。**

**D13 深度全部无效的帧**
- 原因：`keyframe_selection_overlap` 在有效深度像素上做 `torch.randint`，没有有效像素就会报错（`keyframe_selection.py:55-58`）。`init` 在 mask 全空时只打印 WARN，然后把所有点放在相机中心（`gaussian.py:103-107`）。这两点都是读代码推断的，没有实际运行验证。
- 现状：查了 `work_dirs/cvpr_results/table1/g3/closed_loop/outdoor/MAG-B` 下全部服务记录。这里只有部分场景，共 40 个 episode、4,000 条记录，其中 21 条深度全部无效：
  - dunnottar：s1 t0、s2 t0–t12、s2 t16、s4 t0；
  - redeemer：s1 t0–t1；
  - eiffel：s0 t1–t3。
  - 起点帧对所有方法都相同，所以 **dunnottar s1、s2、s4 和 redeemer s1 这 4 个 episode 在 t = 0 必然崩溃**。其余场景的起点没有检查。
- 选项：
  - (a) 无效帧跳过 `track_rgbd`；没有地图时按种子随机或 hover，有地图时照常打分；
  - (b) 把这些 episode 判为无效，在论文中注明。
- **需要用户决定。**

**其他风险**
- **耗时外推**：
  - 只测了前 10 步，而且是在共享 GPU（另一进程 100 % 占用）上测的；
  - 高斯每步增加约 1–2k，到第 100 步可能有 10–20 万个，建图和 Hessian 的单次耗时会随之上升（未测）；
  - 建议在 F2 / F3 先完整跑 1 个 episode，校准耗时后再全量运行。
- **R1**：结果不能逐位复现，断点续跑也不逐位一致，所以全量运行应该一次跑完。

## 6. 每步耗时（`torch.cuda.synchronize` 计时，单位 s，共享 GPU）

outdoor tester（barts s0）：

| t | 候选数 | 高斯数 | 建图（其中 PNG） | H_train | 候选 H | 打分合计 | core* | 总计** | 峰值显存 GB |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 10 | 12,882 | 4.36 (0.67) | 0.012 | 0.384 | 0.396 | 4.77 | 4.99 | 0.118 |
| 1 | 9 | 14,978 | 3.75 (0.40) | 0.015 | 0.322 | 0.337 | 4.10 | 4.20 | 0.121 |
| 2 | 9 | 15,198 | 3.76 (0.41) | 0.033 | 0.373 | 0.406 | 4.18 | 4.30 | 0.123 |
| 3 | 8 | 15,198 | 3.80 (0.41) | 0.044 | 0.318 | 0.363 | 4.18 | 4.32 | 0.124 |
| 4 | 8 | 16,086 | 3.76 (0.38) | 0.044 | 0.311 | 0.355 | 4.13 | 4.21 | 0.128 |
| 5 | 7 | 16,092 | 4.02 (0.59) | 0.068 | 0.283 | 0.350 | 4.38 | 4.79 | 0.129 |
| 6 | 7 | 16,092 | 3.89 (0.42) | 0.083 | 0.298 | 0.381 | 4.29 | 4.39 | 0.131 |
| 7 | 7 | 16,092 | 3.96 (0.40) | 0.140 | 0.276 | 0.416 | 4.38 | 4.48 | 0.132 |
| 8 | 6 | 16,092 | 4.06 (0.38) | 0.128 | 0.266 | 0.394 | 4.47 | 4.53 | 0.133 |
| 9 | 7 | 16,092 | 4.02 (0.38) | 0.186 | 0.313 | 0.498 | 4.53 | 4.62 | 0.136 |

\* core = 读记录 + 建图 + 打分，即只算官方路径。
\*\* 总计 = core 加上 render check，以及 t = 0、5 的重算。

各运行 10 步的均值：

| 运行 | core | 建图 | 打分 | t = 9 的 H_train（10 个关键帧） | 规划端峰值显存 |
|---|---|---|---|---|---|
| outdoor tester | 4.34 | 3.94 | 0.39 | 0.186 | 0.136 GB |
| outdoor docstring | 4.30 | 3.91 | 0.38 | 0.132 | 0.140 GB |
| indoor tester | 4.53 | 4.12 | 0.40 | 0.079 | 0.102 GB |

仿真端：
- outdoor：每步观测 0.85 s；启动约 73 s，含 EpisodeSim 建立和规划进程启动；仿真进程峰值显存 1.15 GB。10 步 episode 共 132 s，回放 72 s。
- indoor：10 步 episode 墙钟 64 s，其中启动加观测共 17.6 s；纹理网格构建另需 11 s。

## 7. F2 / F3 的 GPU 开销估计（由上面的实测外推，未验证）

模型：每步耗时 = 建图 + 候选 + 读取（约 4.2 s outdoor、4.5 s indoor）+ h_kf × (t + 1)。其中 h_kf 是每个关键帧的 Hessian 耗时，实测 8–19 ms，取 15 ms。

| | 步数 | 单 episode | episode 数 | 合计 |
|---|---|---|---|---|
| **F3 Macarons++** 14 × 5 × 100 | 100 | 启动 73 + 观测 101×0.85 ≈ 86 + 100×4.2 = 420 + H_train 5,050×0.015 ≈ 76（40–96）→ 约 **11 min**；加官方覆盖率回放约 2.5 min | 70 | 约 **13 GPU-h 闭环 + 3 GPU-h 回放 ≈ 16 GPU-h** |
| **F2 MP3D** 22 × 2 × 150 | 150 | 启动约 30 + 观测约 150×0.7 ≈ 105 + 150×4.5 = 675 + H_train 11,325×0.015 ≈ 170（90–215）→ 约 **16 min** | 44 | 约 **12 GPU-h** |

说明：
- 显存很小，规划端不超过 0.15 GB，仿真端约 1–3 GB，所以一张卡可以并行几个 episode。但计算本身已经占满，并行的加速有限。
- 在独占的 GPU 上应该更快，但没有测。
- 后期高斯数增多，建图和 Hessian 都会变慢，**上表可能偏低到 2 倍**。
- 以上都**不含** D13 的处理。
