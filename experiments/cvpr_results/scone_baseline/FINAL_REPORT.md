# EXP9-SCONE — SCONE 基线：最终报告（2026-11-14）

## 0. 结论与 Table 1 填写摘要
- 在与 Seen2Map、MAGICIAN、MACARONS、FisherRF 相同的统一协议下，运行了完整的 SCONE：
  - 官方 MACARONS 仓库中更新版的 SCONE 代码；
  - 官方预训练的占据与可见性权重，只在 ShapeNet 上训练过，在两个基准上都是零样本，没有重训或微调；
  - 原版的贪心 NBV 规则。
- 全部完成：室外 70 / 70 条（官方覆盖率重放与存储曲线逐位相同）、室内 44 / 44 条（EXP7 验证全部通过）。
- 室外结果接近 MAGICIAN 论文报告的 SCONE：All-14 AUC@100 0.520，论文 0.535；Final 0.629，论文 0.673。

**Table 1 可直接填写的数值**（场景或楼层等权平均）：

| 数据集 | AUC@20 | AUC@50 | AUC@100 / AUC@150 | Cov@20 | Cov@50 | Cov@100 / Cov@150 |
|---|---|---|---|---|---|---|
| 室外 Macarons++（14 × 5，100 步） | 0.308 | 0.431 | 0.520（@100） | 0.441 | 0.565 | 0.629（@100） |
| 室内 MP3D（22 × 2，150 步） | 0.224 | 0.317 | 0.442（@150） | 0.315 | 0.430 | 0.559（@150） |

完整指标：
- 室外另有 AUC@80 0.494、Cov@80 0.618；
- 室内另有 AUC@80 0.370、AUC@100 0.395、Cov@100 0.507。

## 1. 与其他方法的对比
完整对比表和曲线见 `experiments/cvpr_results/fig/compare_full_v2/`。

| 室外 All-14 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | Cov@20 | Cov@50 | Cov@80 | Cov@100 |
|---|---|---|---|---|---|---|---|---|
| Random Walk† | 0.144 | 0.248 | 0.316 | 0.353 | 0.226 | 0.384 | 0.476 | 0.517 |
| MACARONS† | 0.352 | 0.540 | 0.625 | 0.660 | 0.538 | 0.741 | 0.789 | 0.810 |
| **SCONE†** | 0.308 | 0.431 | 0.494 | 0.520 | 0.441 | 0.565 | 0.618 | 0.629 |
| FisherRF†（完整） | 0.145 | 0.250 | 0.329 | 0.367 | 0.222 | 0.406 | 0.500 | 0.543 |
| MAGICIAN | 0.307 | 0.544 | 0.664 | 0.711 | 0.537 | 0.806 | 0.892 | 0.906 |
| Ours（Seen2Map） | **0.420** | **0.633** | **0.728** | **0.763** | **0.646** | **0.854** | **0.903** | **0.911** |

| 室内 ALL-22 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | AUC@150 | Cov@20 | Cov@50 | Cov@100 | Cov@150 |
|---|---|---|---|---|---|---|---|---|---|
| Random Walk† | 0.121 | 0.181 | 0.230 | 0.257 | 0.309 | 0.173 | 0.270 | 0.383 | 0.436 |
| **SCONE†** | 0.224 | 0.317 | 0.370 | 0.395 | 0.442 | 0.315 | 0.430 | 0.507 | 0.559 |
| FisherRF†（完整） | 0.110 | 0.189 | 0.248 | 0.279 | 0.339 | 0.162 | 0.308 | 0.423 | 0.490 |
| MAGICIAN | 0.176 | 0.290 | 0.358 | 0.387 | 0.434 | 0.264 | 0.437 | 0.512 | 0.540 |
| Ours（Seen2Map） | **0.237** | **0.360** | **0.432** | **0.471** | **0.539** | **0.353** | **0.508** | **0.645** | **0.699** |

**配对差**（场景或楼层 bootstrap 10,000 次，seed 0，95% CI；括号后为前者更高的场景或楼层数；完整表见 `compare_full_v2/compare_paired.csv`）：

| 对比 | 室外 AUC@20 | 室外 AUC@100 | 室外 Cov@100 | 室内 AUC@20 | 室内 AUC@150 | 室内 Cov@150 |
|---|---|---|---|---|---|---|
| Ours − SCONE | +0.112 [+0.095, +0.130]，14/14 | +0.244 [+0.190, +0.299]，14/14 | +0.282 [+0.201, +0.363]，14/14 | +0.013 [+0.005, +0.022]，16/22 | +0.098 [+0.062, +0.135]，20/22 | +0.141 [+0.090, +0.195]，20/22 |
| SCONE − MAGICIAN | +0.001 [−0.041, +0.047] | −0.191 [−0.248, −0.134]，0/14 | −0.278，0/14 | +0.048 [+0.024, +0.076]，21/22 | +0.008 [−0.031, +0.044] | +0.018 [−0.033, +0.066] |
| SCONE − MACARONS | −0.044 [−0.066, −0.016] | −0.141，1/14 | −0.181 | — | — | — |
| SCONE − FisherRF†（完整） | +0.163，14/14 | +0.153，13/14 | +0.085 | +0.114，22/22 | +0.103，20/22 | +0.068 |
| SCONE − Random | +0.164，14/14 | +0.167，14/14 | +0.112 | +0.103，22/22 | +0.133，22/22 | +0.123 |

**解读：**
- **室外：** SCONE 前 20 步与 MAGICIAN 相当，之后明显落后：AUC@100 低 0.191，Final 低 0.278。它也低于 MACARONS，但明显好于 FisherRF 和 Random。
- **室内：** SCONE 前 20 步优于 MAGICIAN（+0.048，21/22 层）；到 150 步两者无显著差异。
- **Ours：** 在室内外所有指标上都高于 SCONE。室内前 20 步差距较小（+0.013），后期差距明显（AUC@150 +0.098）。

## 2. 与 MAGICIAN 论文报告值的对照（室外，同样的 14 个场景）
| | AUC@100 | Final |
|---|---|---|
| 论文 SCONE（Table 6 / 7，去掉 St. Sofia） | 0.535 | 0.673 |
| 本复现 | 0.520 | 0.629 |
| 差 | −0.015 | −0.045 |

- 逐场景 AUC@100 与论文的相关系数为 0.74，8 / 14 个场景在 ±0.05 以内。
- 差异较大的场景：barts（本复现 0.802，论文 0.551）、dunnottar（0.483 / 0.366）、colosseum（0.442 / 0.589）、bannerman（0.366 / 0.559）、pantheon（0.271 / 0.465）。
- MAGICIAN 早期提交（`e4e08fe`）中的 SCONE 脚本与官方 MACARONS tester 有四处不同：每步 30% 随机动作、碰撞判定写反、默认加载 `trained_macarons.pth`、覆盖率不除以 visibility_ratio。论文到底用的是哪个配置无法核实。所以论文值只作参考；本复现严格按官方 MACARONS 代码运行。
- 与 FisherRF 不同（本协议下比论文低 0.174），SCONE 有官方场景级代码和权重，复现值与论文接近。

## 3. 方法来源与适配（详见 `AUDIT_AND_PLAN.md`、`PROTOCOL_ADAPTATION.md`、`SMOKE_TEST.md`）
**代码：**
- MACARONS 上游 `b23c180` 中的 `SconeOcc` / `SconeVis`、`compute_scene_occupancy_probability_field`、`predict_coverage_gain_for_single_camera`，以及 tester 的贪心 argmax。
- 实际执行的是 MAGICIAN_clean_9521d69（室外）和 MAGICIAN `e697009`（室内）中的同一批函数，已逐函数核对与 `b23c180` 相同。
- SCONE 原仓库 `24f1765` 只有 ShapeNet 的脚本；它的 README 指明完整版在 MACARONS 仓库。

**权重：**
- 来源：MACARONS 的 Google Drive，存于 `/bsuscratch/jiahuizhang/projects/occupancy/SCONE_official/weights_gdrive/`（只读）。
- `pretrained_scone_occ.pth`：sha256 `257674e17f7d693bfdd555eed50e4a2ce9e77d5f3c31e45183d68d0faf404bd9`，epoch 301。
- `pretrained_scone_vis.pth`：sha256 `72c9bfdccde7e50578a7c830f3c09e7653074e9df3f23b8241546210b1a315f0`，epoch 209。
- 与本地 `pretrained_macarons.pth` 中的 SCONE 部分 232 / 232 个张量逐位相同。

**保留原版：**
- 两个网络和官方权重；
- 80 万个 proxy point 的体积表示，用深度做 carving；
- 逐 cell 预测占据概率；
- 在视野内按占据概率做 Monte-Carlo 积分，得到覆盖增益；
- 对相邻候选取增益最大者。

**适配（与 Table 1 中 MACARONS 一行完全相同）：**
- 动作用统一协议的合法候选列表：单轴平移一格或单个角度转一格，优先选未访问过的位姿，由 GT 线段做碰撞判定。
- 只给 GT 深度，量程 70 runtime 单位，不给 RGB。
- 共享 10% 的点采样。
- 室内的设置与 EXP7 中 MAGICIAN 室内相同。

**不采用：** 30% 随机动作、beam search、手写启发式、重训。

**GT 隔离：**
- 规划器的表面点全部来自本条轨迹观测到的深度；
- 室内由 LeakGuard 检查，没有拒绝记录；
- GT 只用于模拟器生成观测、合法性判定和最终覆盖评估。

## 4. 运行与验证
**运行：**
- 全部在共享独占 GPU 队列上执行：gpu114 和 gpu115，每张卡一次一个任务。
- 2026-11-14 17:07–18:36，共 114 个任务，0 失败。
- 单条 episode 墙钟：室外平均 5.4 min（含官方覆盖率重放），室内 6.0 min。
- smoke 中单步决策约 3 s，显存峰值室外 4.0 GB、室内 2.5 GB。

**验证：**
- 室外：70 / 70 条官方覆盖率重放与存储曲线逐位相同（`official_equal_stored`）。
- 室内：44 / 44 条通过 V1–V3、与 MAG-B 同起点、来源和 guard 检查，C_obs 重放最大差为 0。
- 汇总脚本 `make_compare_full_v2.py` 会再次逐条断言上述条件，并核对 v1 各行的数值没有变化。

## 5. 文件
| 类别 | 位置 |
|---|---|
| 结果 | `results/scone_{outdoor,indoor}_curves.csv`（逐条逐步覆盖率）、`scone_{outdoor,indoor}_per_unit.csv`（逐场景或逐楼层）、`scone_summary.json` |
| 对比表和曲线 | `experiments/cvpr_results/fig/compare_full_v2/`（新目录，v1 的 `fig/compare_full/` 未改动） |
| 脚本 | `scripts/scone_common.py`、`scone_sim_out.py`、`scone_out_cov.py`、`scone_sim_in.py`、`scone_planner_grid.py`、`scone_validate_in.py`、`scone_smoke_check.py`、`scone_queue_tasks.py`、`inspect_weights.py` |
| 原始数据 | `work_dirs/cvpr_results/scone_baseline/{outdoor,indoor}/`；上游代码参考拷贝在 `reference/` |

**复现命令：**
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
M="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I"
bash -c "$M experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py full --write"   # 写入共享队列；已完成的任务会自动跳过
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_compare_full_v2.py   # 汇总、出表、画曲线
```

## 6. 尚未完成 / 待决定
- **Table 1 正式流水线：** SCONE 还没有登记进 `cvpr_lib` 和 `manifest/episodes.csv`，那里仍标为 BLOCKED。登记后可由 `run_all.sh` 重新生成 Table 1 v3。数值与本报告相同。
- **SCONE-ε0.3 敏感性实验（未运行）：** 加入论文脚本里的 30% 随机动作，对照 0.534。只作敏感性分析，需要用户批准，约 20 GPU-h。
