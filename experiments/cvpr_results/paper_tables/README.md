# CVPR paper tables 与图（展示层，paper-tables-v1）

本目录只做**展示**：所有我们的数值都从 CVPR Table 1 已验证的输出自动读取，不手填，也不改动 Table 1。完整的 Table 1（Markdown / TeX / CSV / JSON、各模型结果与历史记录）保持原样，见 `../table1/`。

## 数据流
```
冻结模拟器 + 官方评估器（每条 episode：官方覆盖曲线，重放逐位一致）
   → work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json   （规范记录：曲线、执行位姿、ckpt_sha256、来源文件）
   → table1/scripts/validate.py → results/validation.json              （完整性与合法性）
   → table1/scripts/aggregate.py → results/per_scene.csv、per_model.csv （场景均值、划分 / All-14 均值，官方口径）
   → paper_tables/scripts/make_tables.py  → tables/*.{md,tex,csv}、figures/table_*.{png,pdf}
   → paper_tables/scripts/make_figures.py → figures/coverage_curves_*、improvement_over_magician、exploratory/radar_all14
```
- **MAGICIAN 论文数值**（表格左半部分）读自 `../table1/SCENE_REGISTRY.yaml` 的 `magician_paper` 字段。它们抄自论文附录表 6 / 7（第 12 页），并已与 PDF 文本独立核对（`../protocol_audit/results/metric_audit.json`，0 处不同）。

## 一键刷新
`bash refresh.sh`：先运行 Table 1 的 `validate.py` 与 `aggregate.py`，再生成全部表与图。任何一致性检查失败都会以非零退出码中止。检查结果、输入文件 sha256、生成时间与 git HEAD 记在 `MANIFEST.json`。

**加入新模型**：
1. 先按 Table 1 的流程注册并跑完 episode（`../table1/README.md`）；
2. 再把模型 id 加入 `config.yaml` 的 `main_models` / `supp_models` / `curve_models`；
3. 运行 `refresh.sh`。

**partial 规则**：
- 表格：不完整的场景显示 `--`，平均显示 `partial`，不参与完整均值；
- 图：只画 14 个场景 × 5 个起点都完整的模型，其余记入 `MANIFEST.json` 的 `excluded_partial`。

## 自动检查（每次生成都运行）
- **表**（166 项）：
  - 每列模型在注册表中为 READY，且 validation 中没有无效 episode；
  - 15 行场景，其中 14 行计入 All-14；sofia_church 没有复现值；
  - 每列 All-14 平均 = 行均值 = `per_model.csv` 的 All-14 值（误差 < 1e-9）；
  - 论文各方法 15 场景行均值与其报告的平均一致（论文 FisherRF Final 除外，见下）；
  - TeX 每行列数正确，括号配平。
- **图**（371 项）：
  - 每条曲线都是 101 个值，起点、场景对应正确；
  - 平均曲线在 t = 20 / 50 / 100 的值与梯形积分 / 100 分别等于 `per_model.csv` 的 Cov@20、Cov@50、Final、AUC@100（误差 < 1e-9）。

## 产物与建议用途
| 文件 | 内容 | 建议 |
|---|---|---|
| `tables/table_auc_full_scenes.*`、`figures/table_auc_full_scenes.*` | 逐场景 AUC@100。左：论文公布值；右：MAGICIAN（复现）、B2、V3-D1 | **正文**（或附录主表） |
| `tables/table_final_full_scenes.*`、`figures/table_final_full_scenes.*` | 逐场景 Final，结构同上 | **正文 / 附录** |
| `tables/*_supp.*` | 同上，右侧加入 MAGICIAN-Greedy、B0、V3-T1（H0）、V3-YBEST-H3 | **Supplementary** |
| `figures/coverage_curves_3panel.*` | All-14 / Dev-5 / Test-3 的平均覆盖曲线，0–100 步 | **正文** |
| `figures/coverage_curves_{all14,dev5,test3}.*` | 单图版本 | 视版面而定 |
| `figures/improvement_over_magician.*` | 逐场景相对 MAGICIAN（复现）的 ΔAUC@20 / ΔAUC@100 / ΔFinal | **正文**（最能说明“早期效率提升、最终覆盖持平”） |
| `figures/exploratory/radar_all14.*` | 6 个指标的雷达图，所有轴统一 0–1 | **只作探索，不进论文** |

## 图注（可直接改写进论文）
- **覆盖曲线**：
  - 每个模型在每个场景对 5 个固定起点（settings.json `start_positions[0..4]`）的官方覆盖曲线（MAGICIAN 评估器，ε = 1.0）取平均，再对场景等权平均；
  - 阴影是场景级 bootstrap 95% CI（10,000 次，seed 0），只为 MAGICIAN（复现）和 V3-D1 绘制；Test-3 只有 3 个场景，区间较粗；
  - 竖线标出 AUC@20 与 AUC@50 的时间点；
  - 所有方法共用一个评估器、同样的起点和 100 步预算。
- **逐场景提升图**：
  - 每个点是该场景 5 个配对起点差的均值，横线为这 5 个差的 ±1 SE；
  - 最下一行是 All-14 均值，误差线为场景级 bootstrap 95% CI；
  - 参照为同一协议下复现的 MAGICIAN（beam 10 × 10，官方代码与权重）。

## 追溯（论文中的每个数字）
- **表格中的单元格**：
  1. → `../table1/results/per_scene.csv` 中（model, scene）这一行；
  2. → 5 条规范 episode `work_dirs/cvpr_results/table1/episodes/<model>/<scene>_s<k>.json`，含 `coverage_official`、`executed_keys`、`ckpt_sha256`、`episode_file`、`metrics_file`；
  3. → `../table1/MODEL_REGISTRY.yaml` 中的 checkpoint 路径与 sha256。
- **论文公布值**：→ `SCENE_REGISTRY.yaml` 的 `magician_paper` 字段 → 论文第 12 页表 6 / 7。

## 已知限制（必须在论文中披露）
- **All-14**：St. Sofia Church 在冻结评估器中无法加载（官方 2026-08-29 才修复加载 bug，见 `../protocol_audit/MAGICIAN_EVAL_AUDIT.md` §1.3）。表格同时给出论文在相同 14 个场景上的均值与它报告的 15 场景均值。
- **两组数值不可直接比较**：论文公布值与我们的复现值来自不同的运行、起点（论文未公布位姿）和 GT 点云定义（论文：渲染的可见点；发布的代码：mesh 采样，再除以 visibility_ratio）。因此只在右侧组内标注最优，**不跨组比较**。
- **论文内部不一致**：论文表 7 的 FisherRF 平均（0.786）与其 15 行的均值（0.772）不一致，表中用星号标出。
