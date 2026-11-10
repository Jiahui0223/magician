# MISSING — 尚缺的 episode 与可直接执行的下一步（生成于 2026-11-10 16:15:34）

| 方法 / 配置 | 完成 | 缺失 | 失败（放弃） | 状态 |
|---|---|---|---|---|
| RANDOM（Random Walk） | 70/70 | 0 | 0 | 完成 |
| SCONE（SCONE） | 0/70 | 0 | 0 | BLOCKED |
| MACARONS（MACARONS） | 70/70 | 0 | 0 | 完成 |
| FISHERRF（FisherRF） | 0/70 | 0 | 0 | BLOCKED |
| MAG-B（MAGICIAN (beam 10x10)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-D1（Ours (V3-T0 + D1)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K4D1（V3-T0, K = 4, depth 1） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D2（V3-T0, K = 1, depth 2） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D1（V3-T0, K = 1, depth 1） | 70/70 | 0 | 0 | 完成 |
| MAG-G（MAGICIAN (greedy)） | 70/70 | 0 | 0 | 完成 |
| V3-D1-H3-NORISK（V3-T1 + D1 (cross-attention control)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D2-N1（V3-T0, K = 1, depth 2, noise map 1） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D2-N2（V3-T0, K = 1, depth 2, noise map 2） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D2-N3（V3-T0, K = 1, depth 2, noise map 3） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D1-N1（V3-T0, K = 1, depth 1, noise map 1） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D1-N2（V3-T0, K = 1, depth 1, noise map 2） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D1-N3（V3-T0, K = 1, depth 1, noise map 3） | 70/70 | 0 | 0 | 完成 |

## 逐项缺口与下一步命令

### SCONE — BLOCKED

- 原因：no scene-level SCONE planner for Macarons++ in the repositories (only the ShapeNet object demo: MACARONS/demos/scone_demo.py, configs/scone, weights/scone); MAGICIAN's SCONE numbers exist only as paper-reported values
- 需要的工作（未做）：为 SCONE 写 Macarons++ 场景级规划器适配（把 `MACARONS/demos/scone_demo.py` 的占据 + coverage-gain 网络接到冻结 EpisodeSim 的合法动作列表上，参照 `experiments/exp3b_macaronspp_transfer/m_t7_macarons.py` 的 R1–R8 修复），再按 `table1/README.md` 的“加入新模型”流程登记并跑 70 条。

### FISHERRF — BLOCKED

- 原因：no FisherRF code, weights or cached episodes in the repositories; it needs an online 3D Gaussian-splatting model per step and has no adapter for the frozen simulator
- 需要的工作（未做）：引入 FisherRF 官方代码与 3DGS 依赖（用户负责环境安装），写冻结模拟器适配器（每步在线训练 3DGS、对合法动作列表算 Fisher 信息增益），再登记并跑 70 条。

## 本轮未安排的项目（计划书 P2）

- Ray Encoder 消融：未做（需要重新训练，不在 8 小时窗口内）。
- Runtime 对照（独占 GPU 的逐方法计时）：未做；Table 3 的 CSV 只给出共享 GPU 下的规划器每步决策秒数，不能当作正式 runtime。
- sofia_church（第 15 个场景）：冻结协议 v1.0 下所有方法都不可运行，始终报告 All-14。

