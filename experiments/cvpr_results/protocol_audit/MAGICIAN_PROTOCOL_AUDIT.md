# MAGICIAN_PROTOCOL_AUDIT — 协议审计总览（2026-11-04）

**范围**：核对我们的 CVPR Results（`experiments/cvpr_results/table1/`）与以下三方是否一致：
- MAGICIAN 论文：arXiv 2603.22650，正文与附录；
- 官方代码：github shiyao-li/MAGICIAN，HEAD e697009；
- 我们冻结模拟器实际导入的 fork：`MAGICIAN_clean_9521d69`。

另外核对 MAGICIAN、MACARONS、SCONE、FisherRF 与我们各模型的训练条件。

**原则**：
- 只读，没有修改任何历史结果、Table 1 文件或已有审计文档；
- 没有训练；
- 唯一的计算是 CPU 上的重算与核对。

## 文件
| 文件 | 内容 |
|---|---|
| `MAGICIAN_EVAL_AUDIT.md` | 评估协议逐项审计：场景、起点、步数与动作空间、覆盖率、AUC / Final、论文 vs 复现 |
| `AUC_COVERAGE_FORMULA_AUDIT.md` | 公式的最小数值测试与各种定义下的数值 |
| `MAGICIAN_TRAINING_AUDIT.md` | 基线的训练（MAGICIAN / MACARONS / SCONE / FisherRF / Random Walk）、对已有审计的更正、未知项 |
| `TRAINING_DATA_COMPARISON.md` | 我们各模型的训练数据、选择方式与公平性披露表 |
| `INITIAL_POSES_COMPARISON.csv` | 15 场景 × 5 起点：settings 下标、位置 / 朝向、LMDB 第一位姿与差值、Table 1 各模型第 0 步核对、训练重叠 |
| `SCENE_BY_SCENE_PAPER_VS_REPRODUCED.csv` | 15 场景：论文 5 种方法的 AUC / Final，MAG-B 与 LMDB 两份复现在各种定义下的值、起点标准差、差值 |
| `AUDIT_CHECKLIST.md` | 41 项清单（PASS / PARTIAL / FAIL / UNKNOWN / UNRESOLVED） |
| `FINAL_AUDIT_REPORT.md` | 结论：哪些结果可以写进 CVPR、哪些需要脚注或补充验证 |
| `scripts/` | `diff_eval_functions.py`、`audit_starts.py`、`audit_metrics.py`（全部可重跑） |
| `results/` | 自动生成的中间结果：`eval_function_diff.{md,json}`、`initial_poses_summary.json`、`metric_audit.json`、`lmdb_coverage_cache.npz` |
| `drafts/TRAINING_FACTS_DRAFT.md` | 训练事实的原始草稿（子代理整理，主会话已抽查 5 项，全部一致） |

## 复现
```
python experiments/cvpr_results/protocol_audit/scripts/diff_eval_functions.py                 # seen2scene 环境
env -u LD_PRELOAD -u PYTHONPATH $MAGICIAN_PY experiments/cvpr_results/protocol_audit/scripts/audit_starts.py   # magician 环境（读 LMDB）
python experiments/cvpr_results/protocol_audit/scripts/audit_metrics.py                      # seen2scene 环境（LMDB 用缓存）
```
