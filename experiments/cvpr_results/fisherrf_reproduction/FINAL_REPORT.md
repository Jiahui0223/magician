# FisherRF 复现 — 最终报告（2026-11-11）

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
