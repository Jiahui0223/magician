# 完整对比 v2：加入 SCONE 和室内 FisherRF-full（2026-11-14）

由 `experiments/cvpr_results/scripts/make_compare_full_v2.py` 生成，不要手改。v1（`fig/compare_full/`）保留不变，本目录是新版本。重新生成：
```bash
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_compare_full_v2.py
```

**相对 v1 的改动：**
- 新增 SCONE† 一行，室内外都有。来自 EXP9-SCONE，见 `experiments/cvpr_results/scone_baseline/FINAL_REPORT.md`。
- 室内新增 FisherRF†（完整）一行，即 f1c4，见 `fisherrf_reproduction/INDOOR_RESULTS.md`。
- 室外新增 Cov@80 一列（SCONE 任务书要求）。
- 其余各行和数据来源与 v1 完全相同，脚本会断言这些行与 v1 及 Table 1 / EXP7 一致。

| 文件 | 内容 |
|---|---|
| `compare_outdoor.{pdf,png,tex,csv}` | 室外 Macarons++ All-14 × 5 个官方起点，100 步：AUC@20/50/80/100、Cov@20/50/80/100 |
| `compare_indoor.{pdf,png,tex,csv}` | 室内 MP3D-150 ALL-22 × 2 个起点，150 步：AUC@20/50/80/100/150、Cov@20/50/100/150 |
| `compare_curves.{pdf,png}`、`compare_curves_data.csv` | 覆盖曲线，左室外、右室内，阴影为 95% bootstrap 置信区间 |
| `compare_paired.csv` | Ours − 各基线，以及 SCONE − Random / MACARONS / FisherRF（完整） / MAGICIAN；室外按场景、室内按楼层 bootstrap 10,000 次，seed 0 |

## 室外（100 步）
| 方法 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | Cov@20 | Cov@50 | Cov@80 | Cov@100 |
|---|---|---|---|---|---|---|---|---|
| Random Walk† | 0.144 | 0.248 | 0.316 | 0.353 | 0.226 | 0.384 | 0.476 | 0.517 |
| MACARONS† | 0.352 | 0.540 | 0.625 | 0.660 | 0.538 | 0.741 | 0.789 | 0.810 |
| SCONE† | 0.308 | 0.431 | 0.494 | 0.520 | 0.441 | 0.565 | 0.618 | 0.629 |
| FisherRF†（一步） | 0.130 | 0.219 | 0.272 | 0.301 | 0.197 | 0.329 | 0.395 | 0.439 |
| FisherRF†（完整） | 0.145 | 0.250 | 0.329 | 0.367 | 0.222 | 0.406 | 0.500 | 0.543 |
| MAGICIAN | 0.307 | 0.544 | 0.664 | 0.711 | 0.537 | 0.806 | 0.892 | 0.906 |
| **Ours** | **0.420** | **0.633** | **0.728** | **0.763** | **0.646** | **0.854** | **0.903** | **0.911** |

## 室内（150 步）
| 方法 | AUC@20 | AUC@50 | AUC@80 | AUC@100 | AUC@150 | Cov@20 | Cov@50 | Cov@100 | Cov@150 |
|---|---|---|---|---|---|---|---|---|---|
| Random Walk† | 0.121 | 0.181 | 0.230 | 0.257 | 0.309 | 0.173 | 0.270 | 0.383 | 0.436 |
| SCONE† | 0.224 | 0.317 | 0.370 | 0.395 | 0.442 | 0.315 | 0.430 | 0.507 | 0.559 |
| FisherRF†（一步） | 0.078 | 0.133 | 0.171 | 0.192 | 0.233 | 0.120 | 0.212 | 0.287 | 0.338 |
| FisherRF†（完整） | 0.110 | 0.189 | 0.248 | 0.279 | 0.339 | 0.162 | 0.308 | 0.423 | 0.490 |
| MAGICIAN | 0.176 | 0.290 | 0.358 | 0.387 | 0.434 | 0.264 | 0.437 | 0.512 | 0.540 |
| **Ours** | **0.237** | **0.360** | **0.432** | **0.471** | **0.539** | **0.353** | **0.508** | **0.645** | **0.699** |

MACARONS 没有在 MP3D 上运行，所以室内表不含这一行（PDF 和 CSV 中显示为 —）。

**† 表示由我们按本协议适配：**
- SCONE：官方 MACARONS 中的 SCONE 代码 + 官方预训练权重（只在 ShapeNet 上训练），零样本。
- FisherRF：我们的复现，与论文报告值不同，原因见 `fisherrf_reproduction/AUDIT.md`。
- 口径与 v1 相同：室外场景等权，室内楼层等权；每个场景或楼层内先对起点取平均。
