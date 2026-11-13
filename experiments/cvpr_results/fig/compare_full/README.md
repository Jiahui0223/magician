# 完整对比（含我们复现的 FisherRF）：室外 100 步与室内 150 步（2026-11-13）

由 `experiments/cvpr_results/scripts/make_compare_full.py` 生成，不要手改。重新生成：
```bash
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_compare_full.py
```

用户要求（2026-11-13）：FisherRF 先用我们自己代码跑出的结果，给出完整对比，指标为 AUC@20 / 50 / 80 / 100 和 Cov@20 / 50 / 100；室内同理，另加 @150。

## 文件
| 文件 | 内容 |
|---|---|
| `compare_outdoor.{pdf,png,tex,csv}` | 室外 Macarons++ All-14 × 5 个官方起点，100 步 |
| `compare_indoor.{pdf,png,tex,csv}` | 室内 MP3D-150 ALL-22 × 2 个起点，150 步 |
| `compare_curves.{pdf,png}` | 覆盖率曲线，左室外、右室内；阴影为按场景或楼层 bootstrap 的 95% CI |
| `compare_curves_data.csv` | 曲线数值 |
| `compare_paired.csv` | Ours − 各基线的配对差：室外按场景、室内按楼层 bootstrap，10,000 次，seed 0；附 95% CI 和 Ours 更高的场景或楼层数 |

## 室外（Macarons++，100 步）
| 方法 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | Cov@20 | Cov@50 | Cov@100 |
|---|---|---|---|---|---|---|---|
| Random Walk† | 0.144 | 0.248 | 0.316 | 0.353 | 0.226 | 0.384 | 0.517 |
| MACARONS† | 0.352 | 0.540 | 0.625 | 0.660 | 0.538 | 0.741 | 0.810 |
| FisherRF†（一步） | 0.130 | 0.219 | 0.272 | 0.301 | 0.197 | 0.329 | 0.439 |
| FisherRF†（完整） | 0.145 | 0.250 | 0.329 | 0.367 | 0.222 | 0.406 | 0.543 |
| MAGICIAN | 0.307 | 0.544 | 0.664 | 0.711 | 0.537 | 0.806 | 0.906 |
| **Ours** | **0.420** | **0.633** | **0.728** | **0.763** | **0.646** | **0.854** | **0.911** |

## 室内（MP3D，150 步）
| 方法 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | AUC@150 | Cov@20 | Cov@50 | Cov@100 | Cov@150 |
|---|---|---|---|---|---|---|---|---|---|
| Random Walk† | 0.121 | 0.181 | 0.230 | 0.257 | 0.309 | 0.173 | 0.270 | 0.383 | 0.436 |
| FisherRF†（一步） | 0.078 | 0.133 | 0.171 | 0.192 | 0.233 | 0.120 | 0.212 | 0.287 | 0.338 |
| MAGICIAN | 0.176 | 0.290 | 0.358 | 0.387 | 0.434 | 0.264 | 0.437 | 0.512 | 0.540 |
| **Ours** | **0.237** | **0.360** | **0.432** | **0.471** | **0.539** | **0.353** | **0.508** | **0.645** | **0.699** |

室内没有运行 MACARONS 和 FisherRF（完整版，室内版本未实现），表中这两行为 —。

## 口径与来源
- **计算方式：** AUC@t = trapz(C[0..t]) / t，Cov@t = C[t]。室外 Cov@100 和室内 Cov@150 就是 Final。室外先对每个场景的 5 个起点取平均，再按场景等权平均；室内先对每层的 2 个起点取平均，再按楼层等权平均。
- **室外数据：**
  - Random、MACARONS、MAGICIAN、Ours：`manifest/episodes.csv` 中的 COMPLETE 记录，即 Table 1 的同一批，官方覆盖率逐位重放核对过。
  - FisherRF：`work_dirs/cvpr_results/fisherrf_reproduction/{full/outdoor/closed_loop_f1c4, outdoor/closed_loop}/`，每条 episode 的官方覆盖率重放都与存储曲线逐位相同。
- **室内数据：**
  - Random、MAGICIAN、Ours：EXP7 MP3D-150 中 valid = 1 的记录。
  - FisherRF（一步）：44 / 44 条通过 V1–V4 检查。
- **自动交叉核对（脚本内 assert）：** 与 `table1_all14_full.csv`、`exp7/results/mp3d150_summary.csv`（ALL-22）、`fisherrf_reproduction/results/{full14_f1c4_table.json, summary.json}` 的数值一致。
- **† 表示由我们按本协议适配：**
  - FisherRF 是我们的复现，与论文报告值（AUC 0.546 / Final 0.786）不同。原因见 `fisherrf_reproduction/AUDIT.md`：没有实现 bug，但 EIG 在本协议下没有覆盖增益信号。
  - 完整版 = frontier 目标 + Fisher 路径评估（f1c4 变体）。
  - 一步版 = 对相邻合法动作用 Fisher 打分（D3a）。

## 关键配对差（Ours − 基线，95% CI）
| 对比 | 室外 AUC@20 | 室外 AUC@100 | 室外 Cov@100 | 室内 AUC@150 | 室内 Cov@150 |
|---|---|---|---|---|---|
| − FisherRF†（完整） | +0.275 [+0.238, +0.313] | +0.396 [+0.361, +0.431] | +0.367 [+0.301, +0.429] | — | — |
| − FisherRF†（一步） | +0.290 | +0.463 | +0.472 | +0.307 [+0.263, +0.352] | +0.362 [+0.295, +0.433] |
| − MAGICIAN | +0.113 [+0.073, +0.155] | +0.053 [+0.019, +0.091] | +0.004 [−0.027, +0.032] | +0.106 [+0.080, +0.132] | +0.159 [+0.114, +0.208] |

- 对 FisherRF：两个版本在室外都是 14/14 个场景 Ours 更高；一步版在室内是 22/22 个楼层 Ours 更高。
- 对 MAGICIAN：室外 Final 无显著差异（6/14），室内 Final 在 22/22 个楼层上 Ours 更高。
