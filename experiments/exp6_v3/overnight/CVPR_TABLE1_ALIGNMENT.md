# CVPR_TABLE1_ALIGNMENT — 本夜新增的 Table 1 行

- **注册**：所有新行都通过 `experiments/cvpr_results/table1/MODEL_REGISTRY.yaml` 注册，由 Table 1 自己的 `t1_queue.sh` 运行。
- **覆盖与导入**：每个 episode 经 `t1_out_cov` 做官方覆盖重放（与模拟器存档逐位一致），再由 `import_episodes.py --new` 导入规范记录。
- **汇总**：最后用 Table 1 的 `validate.py`、`aggregate.py`（官方 + 单调）、`plot.py`、`results_analysis.py`、`compute_cost.py` 重新生成。
- **旧数值**：未手工修改；旧行的数值没有变化，所有生成文件都可以从规范记录重新生成。

| model_id | 模型 / 策略 | checkpoint sha256 | episodes（Train / Dev / Test / All-14） | 状态 |
|---|---|---|---|---|
| V3-D1-H3-NORISK | V3-T1 40K + D1（D-v1，depth 2，λ 0） | 639c9d3a…（V3-T1） | 30 / 25 / 15 / **70** | 完整 |
| V3-D2-DEPTH3-B3 | V3-T1 40K + depth 3，beam 3 | 639c9d3a… | 5 / 25 / 0 / 30 | partial（Dev + pisa） |
| V3-D3-DEPTH3-B5 | V3-T1 40K + depth 3，beam 5 | 639c9d3a… | 5 / 25 / 0 / 30 | partial（Dev + pisa） |
| V3-D4-DEPTH4-B3 | V3-T1 40K + depth 4，beam 3 | 639c9d3a… | 5 / 25 / 0 / 30 | partial（Dev + pisa） |
| V3-Y45K-H0 | Y 45K + H0 | 37da75b2… | 0 / 25 / 0 / 25 | partial（Dev） |
| V3-Y50K-H0 | Y 50K + H0 | 64cea1d6… | 30 / 25 / 15 / **70** | 完整 |
| V3-YBEST-H3 | Y 50K + H3（H-v1，未改） | 64cea1d6… | 30 / 25 / 15 / **70** | 完整 |
| V3-YD-COMBINED-Y50K-D1 | Y 50K + D1 | 64cea1d6… | 0 / 25 / 0 / 25 | partial（Dev） |

**所有新行共用的推理设置**：
- b2：D100 模型输入 + S10 band_dil 评分；
- R10，K = 4，Euler 10；
- 官方 settings 起点 0–4，100 步。

**策略与配置的哈希**：记录在每个规划器日志的 `D_POLICY` / `H_POLICY` 行。

**All-15**：sofia_church 仍为 PENDING，没有 All-15 行。
