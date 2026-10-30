# Task 3 / E2 — V2-Mix Fine-Tuning Pilot（2026-10-29）

| 文件 | 内容 |
|---|---|
| `protocol.md` | v1.0，训练前冻结（组别、数据、训练、开发集状态、G1 / G2 / G3 关口） |
| `reports/E2_AUDIT.md` | 训练代码、数据与标签审计；冒烟；数据量、显存、时间预算 |
| `reports/E2_G12_REPORT.md` | 训练曲线、G1 预测、G2 排序、中间 checkpoint 诊断、成本 |
| `reports/E2_G3_REPORT.md` | G3 闭环（探索性：G2 未通过，按用户指示运行） |
| `DEVIATIONS.md` | 偏差记录 |
| `configs/e2_{d100,s10}_v1.json` | 两组训练配置（只差数据） |
| `scripts/` | `e2_paths.py`、`e2_build.py`、`e2_train.py`、`e2_valsel_oracle.py`、`e2_fixed.py`、`e2_analyze.py` |
| `analysis/`、`figures/` | `g12_s{1000,3000,5000}.{json,md}`；`g12_s5000.png`、`training_curves.png` |

大文件在 `work_dirs/exp5_future_view/task3_full_depth_p1/e2/`：
- D100 数据：`w2_d100/`、`out_views_d100/`；
- 训练：`training/{s10,d100}/`，EMA 每 1k 步一个；
- 固定状态结果：`fixed/`；
- VALSEL 状态：`valsel/`。

## 结论（按冻结协议）
- **G1 通过**：E2-D100 相对 B2，unknown 与 truly-new 误差下降，离散度未塌缩超过阈值。
- **G2 不通过**：Macarons++ Dev-3 上 E2-D100 的排序差于 B2；MP3D VALSEL 上更好。
- **G3 闭环**（探索性，用户 10-29 指示）：按冻结标准判为 **Conditional Go**。
  - MP3D：E2-D100 − E2-S10 ΔAUC +0.017（4/5），有意义；E2-D100 相对 B0 ΔAUC +0.043、ΔFinal +0.084。
  - Macarons++：E2-D100 − E2-S10 −0.008（未退化），但两个微调组都不如冻结模型（E2-D100 − B2 −0.019）。

## 变更记录
- 2026-10-29：审计、协议、D100 数据、两组训练、G1 / G2 完成。
- 2026-10-29：用户指示运行 G3（探索性），已完成。
