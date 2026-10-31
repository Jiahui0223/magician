# Experiment A — 校验（协议 §2 必过检查，自动生成）

| 池 / 数据集 | 已加载 | 缺失 | 回归：B2 / E2D / T0 / T1 逐位（有参照的状态） | 数值等同（|Δ| ≤ 1e-8，偏差 #1） | GTD = EXP5（逐位） | fallback 选点（任一评分器） |
|---|---|---|---|---|---|---|
| cl / indoor | 368 | 0 | 0/0 / 0/0 / 0/0 / 0/0 | 0/0 / 0/0 / 0/0 / 0/0 | 365/368 | 8 |
| cl / outdoor | 1093 | 0 | 147/150 / 147/150 / 147/150 / 148/150 | 150/150 / 150/150 / 150/150 / 150/150 | 1079/1093 | 26 |
| fx / indoor | 90 | 0 | 88/90 / 88/90 / 88/90 / 88/90 | 90/90 / 90/90 / 90/90 / 90/90 | 0/0 | 2 |
| fx / outdoor | 125 | 0 | 121/125 / 121/125 / 121/125 / 121/125 | 125/125 / 125/125 / 125/125 / 125/125 | 0/0 | 3 |

- 回归 = 新重放的 K=4 `distp1_k` 与 EXP6 `eval_fixed` 行（B2 / T0 / T1）、E2 行（E2D = D100）逐位相同；Pool CL 的参照 = Dev-3 B0 层 150 个状态，Pool FX = 全部状态。
- GTD = EXP5：Pool CL 每个状态由 `ab_score.py` 在稀疏几何上重算的 GT 深度 DistP1 与 EXP5 `v2.json` 逐位相同（Pool FX 无历史值）。
- `legal_idx` 与 `oracle.npz` 的一致性在打分时断言；候选集 `cand` 来自清单；选点规则 = 线上规则（`v2nv_planner.nv_choice` / `m_score.choose`，含 fallback）。
- GT（oracle.npz）在该状态全部模型分数算完之后读取（`ab_score.run_episode` 的代码顺序）。- **GTD 的 17 个非逐位状态**（MP3D 3、Macarons++ 14）：max |Δ| ≤ 1.14e-12（相对 ≤ 1.8e-14），所有状态的 GTD argmax 不变 → 与偏差 #1 同一现象（GPU float64 求和顺序），按"数值等同"计 368 / 368、1,093 / 1,093。
- **回归总览**：有参照的 365 个状态（Pool FX 215 + Pool CL Dev-3 B0 150）× 4 个模型 = 1,460 次比对，逐位 1,427 次，其余 33 次 |Δ| ≤ 7.0e-12 且 argmax 相同 → 数值等同 1,460 / 1,460。
- 检查 (5) 候选置换不变性：`tests/test_metrics.py` / `test_b_scorer.py`（5 passed）；检查 (6) GT 读取顺序：`ab_score.run_episode` 中 `oracle.npz` 在四个模型的 `distp1_k` 写入 `res` 之后才打开。
