# FisherRF 复现 — 最终报告（2026-11-11，2026-11-12 更新）

## 0. 当前结论（2026-11-12 22:00，FisherRF-full 全量完成）
**FisherRF-full 没有复现出 MAGICIAN 论文中的 FisherRF 数值。**
- 按论文的做法保留了 frontier 目标和路径评估，选用最接近论文的变体（f1c4）跑完 Macarons++ All-14 × 5，共 70 条。官方覆盖率重放 70 / 70 条逐位相同。
- 结果：AUC@100 0.367、Final 0.543；论文同 14 个场景的平均为 0.541 / 0.766。场景平均差 −0.174 / −0.222，只有 2 / 14 个场景在论文值 ±0.05 以内（pantheon、barts）。
- 在我们的协议下与 Random Walk 无显著差别：AUC@100 +0.014 [−0.023, +0.048]。
- 同一协议下，MACARONS（0.660，论文 0.647）和 MAGICIAN（0.711，论文 0.721）都与论文相差在 0.015 以内。所以差距来自 FisherRF 的适配方式，不是协议本身。MAGICIAN 的 FisherRF 适配代码没有公开，论文只写了"修改 frontier 选择、调整动作空间"。

**版本与试点**（dunnottar / redeemer / barts × 5，100 步，逐条重放核对）。v4 和 f1c4 两个变体在运行前预先登记，规则是选 3 场景 AUC@100 离论文更近的一个跑全量，此外没有任何调参：

| 版本 | 内容 | AUC@20 | AUC@100 | Final |
|---|---|---|---|---|
| 论文（这 3 个场景） | — | — | 0.633 | 0.854 |
| D3a（消融） | 官方 `pose_eval` 只对相邻一步的候选打分 | — | 0.429 | 0.651 |
| v3 | 官方 frontier 选簇 + 256 个候选 → 前 20 → 5D 位姿图最短路 + 官方路径 EIG | 0.207 | 0.491 | 0.678 |
| v4 | v3 + 可达性过滤（官方 `gaussian.py:1244-1257`） | 0.207 | 0.492 | 0.661 |
| **f1c4** | MAGICIAN 式全局位姿采样（候选位置不被占据、视野内有未知体素）+ 可达性过滤 | 0.181 | **0.510** | **0.722** |

**全量 All-14**（场景等权；与 Table 1 的行同协议，结果在 `results/full14_f1c4_table.md`）：

| 方法 | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Cov@50 | Final |
|---|---|---|---|---|---|---|
| FisherRF-full（f1c4） | 0.145 | 0.250 | 0.367 | 0.222 | 0.406 | 0.543 |
| FisherRF-score-1step（D3a，消融） | 0.130 | 0.219 | 0.301 | 0.197 | 0.329 | 0.439 |
| Random Walk | 0.144 | 0.248 | 0.353 | 0.226 | 0.384 | 0.517 |
| MACARONS | 0.352 | 0.540 | 0.660 | 0.538 | 0.741 | 0.810 |
| MAGICIAN | 0.307 | 0.544 | 0.711 | 0.537 | 0.806 | 0.906 |
| Ours | 0.420 | 0.633 | 0.763 | 0.646 | 0.854 | 0.911 |

- **配对差值**（按场景 bootstrap；括号后为 FisherRF-full 更高的场景数）：

  | 对比 | AUC@100 | Final |
  |---|---|---|
  | − D3a | +0.066 [+0.036, +0.098]，13/14 | +0.105 |
  | − Random | +0.014 [−0.023, +0.048]，9/14 | +0.027 [−0.028, +0.077] |
  | − MAGICIAN | −0.344，0/14 | — |
  | − Ours | −0.396，0/14 | — |

- **与论文逐场景的 AUC@100 对比**（括号内为论文值）：

  | 场景 | 本复现（论文） |
  |---|---|
  | dunnottar | 0.319（0.500） |
  | colosseum | 0.220（0.551） |
  | bannerman | 0.362（0.595） |
  | pantheon | 0.257（0.270） |
  | redeemer | 0.553（0.727） |
  | liberty | 0.353（0.553） |
  | pisa | 0.364（0.486） |
  | fushimi | 0.299（0.565） |
  | alhambra | 0.184（0.462） |
  | neuschwanstein | 0.278（0.375） |
  | eiffel | 0.441（0.616） |
  | bridge | 0.402（0.637） |
  | barts | 0.659（0.673） |
  | sestino_museum | 0.450（0.571） |

  逐场景与论文的相关系数为 0.71：哪些场景难、哪些容易，两者基本一致，但整体水平低一截。
- **行为：** 每条 episode 去过 31–53 个不同位置，每 100 步重规划 10–22 次。在独占 L40S 上，规划端每步平均约 2.2 s（含 3DGS 建图）。

**能否进入 CVPR 表格：**
- 不能写成"FisherRF（论文方法）的结果"。
- 可以写成同协议的适配版 FisherRF†，并注明三点：
  1. 按 MAGICIAN 论文描述适配；
  2. 在我们的协议下，与论文报告值相差 −0.17 AUC@100；
  3. 论文的适配代码没有公开。
- 另一种做法：引用论文报告的 FisherRF 数值，单独分组（论文报告值与本复现分开）。这与 `paper_tables/` 已有的"论文报告组 / 重新评估组"分法一致。
- 用哪种请用户决定。

**未完成：** MP3D 室内版的 FisherRF-full 还没有实现；室内只有 D3a 消融的结果（AUC@150 0.233）。

**文件：**
- 试点报告：`PILOT_FULL.md`（v3）。
- 运行记录与重跑命令：`FISHERRF_FULL_PAUSED.md` §6（队列用法）。
- 结果：`results/pilot_{v4,f1c4}_*`、`results/full14_f1c4_{episodes.csv,summary.json,table.md,table.json}`；汇总脚本 `scripts/frf_full14_table.py`。
- 原始数据：`work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_{v3,v4,f1c4}/`。
- 运行方式：全部在共享独占队列 `work_dirs/cvpr_results/gpu_queue` 上执行（任务 `frf_*`，11-12 19:30–21:51），没有用 Slurm。

---
以下 §1–§5 是 2026-11-11 的 D3a 版本原文，保留不改。

> **状态更正（2026-11-11）：** 本文件中标为 FisherRF 的数值来自 **D3a 版本**：官方 3DGS 建图加官方 `pose_eval` 评分，但候选只有相邻一步的合法动作，**没有官方的 frontier 目标和 A\* 路径评估**。它不是 MAGICIAN 论文所用的 FisherRF 适配版（论文 Table 2：AUC 0.546、Final 0.786），只能作为消融，记作 **FisherRF-score-1step**，不能作为 FisherRF 行进入 CVPR 表格。按 MAGICIAN 论文做法保留 frontier 目标和路径评估的 **FisherRF-full** 正在实现，见 `PILOT_FULL.md`（完成后生成）。以下原文保留不改。


**结论：**
- 在我们的两个冻结协议下，复现了官方 FisherRF 视角评分（Fisher 信息 EIG，`backward_power = 2`）加在线 3DGS 建图，并在与其他方法相同的统一候选和动作空间中运行完了全部闭环：Macarons++ 70 / 70 条、MP3D-150 44 / 44 条。
- 两个协议下都通过了同协议核对：室外官方覆盖率逐位重放，室内 C_obs 逐位重放加动作合法性检查。
- 在两个环境中，FisherRF 都显著低于 Random Walk，更低于 MAGICIAN 和 Ours。每步计算更便宜，约 1.5 s（独占 L40S）。

本复现的目的是忠实复现方法本身，好坏不是目标（用户 2026-11-11 的说明）。

## 1. 源码、环境与执行
- **官方源码：**
  - FisherRF `b74732812b29`（含 4 个子模块）；
  - FisherRF-active-mapping `8b196cd22b1f`；
  - 位置 `/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/`，未做任何修改。
- **环境：** `/bsuscratch/jiahuizhang/envs/fisherrf`（用户授权由 Claude 安装）。
  - torch 2.2.2+cu118（官方指定），habitat-sim 0.2.4；
  - 两个 CUDA 扩展用 sm_89 编译；
  - open3d 用桩模块替代：官方代码只 import 它，唯一调用它的 `o3d_knn` 从未被调用；桩模块一旦被调用会立即报错。
  - 详见 `ENVIRONMENT.md`。
- **作业：**

  | 作业 | 用途 |
  |---|---|
  | F1 smoke | gpu114 GPU 3 |
  | 校准 3495338 / 3495340 | 独占 L40S |
  | 全量 3495366 / 3495368 / 3495370 | 每卡 4 个进程 |
  | 覆盖率重放 3495384 | — |

  提交前都确认无需抢占。
- **日志：** `work_dirs/cvpr_results/fisherrf_reproduction/logs/`。

## 2. 适配（`PROTOCOL_ADAPTATION.md`、`SMOKE_TEST.md`）
保留的核心：
- 官方 GaussianSLAM 在线建图：每步跟踪，用 GT 位姿，每步 60 次迭代；
- 官方 `pose_eval` 打分：Σ H_cand ⊙ (H_train + 0.1)⁻¹。

替换的外围部分：
- Habitat、frontier、A* 改为本协议的合法候选和执行器；
- RGB 由我们的仿真端提供：室外用 MAGICIAN 的 RGB 帧，室内新增纹理渲染；
- 坐标尺度 s = 7，内参用 PyTorch3D 的实测值。

需要说明的两处偏离官方代码路径：
- **D12**：官方 tester 把 w2c 传给了要求 c2w 的 `init`，这是 bug，我们改用 docstring 语义；
- **D13**：当前帧没有有效深度时跳过建图，没有地图时按种子随机选择。

## 3. 结果
| 环境 | 指标 | FisherRF | Random | MAGICIAN | Ours |
|---|---|---|---|---|---|
| Macarons++ All-14（100 步） | AUC@20 / AUC@100 / Final | 0.130 / 0.301 / 0.439 | 0.144 / 0.353 / 0.517 | 0.307 / 0.711 / 0.906 | 0.420 / 0.763 / 0.911 |
| MP3D ALL-22（150 步） | AUC@20 / AUC@150 / Final | 0.078 / 0.233 / 0.338 | 0.121 / 0.309 / 0.436 | 0.176 / 0.434 / 0.540 | 0.237 / 0.539 / 0.699 |

- **配对差值：**
  - 室外 FisherRF − Ours：AUC@100 −0.463，14 个场景全部更低；
  - 室内 FisherRF − Ours：AUC@150 −0.307，22 个楼层全部更低；
  - 与 Random 相比：室外 AUC@100 −0.052 [−0.087, −0.016]，室内 −0.076 [−0.109, −0.044]。
- **行为：** 平均只到过约 19–21 个不同位置，大部分步数在原地转视角。EIG 偏好补拍附近已有 Gaussian，而统一动作空间里没有远距离目标。
- **运行时间（独占 L40S）：** 每步视角选择 0.27–0.40 s，3DGS 建图 1.06–1.20 s，合计约 1.46 s，显存 < 0.5 GB。详见 `RUNTIME_RESULTS.md`。

## 4. 能否进入 CVPR 表格
- 可以作为同协议的适配基线行（加 † 标注），放在 Table 1 和 MP3D 表中，与 Random / MACARONS 行同等对待。
- 必须注明三点：
  - 候选集是统一的相邻动作，没有官方的 frontier 目标引导；
  - D12 / D13 的处理；
  - 结果不是论文数值。官方实验使用 Gibson / HM3D、2000 步、渲染质量指标，与本协议不同，不可比较。
- 不建议写"FisherRF 比随机还差"之类的一般性结论，只限定在本协议下。

## 5. 文件
| 类别 | 文件 |
|---|---|
| 报告 | `SOURCE_AUDIT.md`、`ENVIRONMENT.md`、`PROTOCOL_ADAPTATION.md`、`SMOKE_TEST.md`、`OUTDOOR_RESULTS.md`、`INDOOR_RESULTS.md`、`RUNTIME_RESULTS.md` |
| 脚本 | `scripts/`：`fisherrf_planner.py`、`fisherrf_sim_out.py`、`fisherrf_sim_in.py`、`fisherrf_out_cov.py`、`frf_queue.sh`、`frf_job.sbatch`、`frf_cov_queue.sh`、`frf_collect.py`、`stubs/open3d` |
| 汇总数据 | `results/`：`summary.json`、`outdoor_episodes.csv`、`indoor_episodes.csv`、`runtime_summary.json` |
| 原始数据 | `work_dirs/cvpr_results/fisherrf_reproduction/{outdoor,indoor}/closed_loop/`，含逐步覆盖、动作、服务记录、规划器日志、RGB 旁路文件 |
