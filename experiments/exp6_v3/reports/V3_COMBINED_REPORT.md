# EXP6 — V3 Combined Report: Simple 3D VAE vs 3D VAE + Cross-Attention

> STATUS（2026-10-30）：**IN PROGRESS** — Phase 1 / 2 完成，G0 进行中，训练未开始。本文件按 `../EXP6_V3_REPORT_TEMPLATE.md` 结构填写，每一节标注 `Completed / Running / Failed / Not Run / Exploratory`；所有数字只来自真实日志（路径随数字给出）。
> 真相来源：`../protocol.md`、run manifests（`work_dirs/exp6_v3/task*/runs/*/run_*.json`）、checkpoints、G1/G2/G3 原始结果。

## 1. Executive Summary
- 问题 1：从零训练的 3D VAE partial-TSDF 表示能否改善未来视角生成与闭环 NBV？ — **Not Run**
- 问题 2：同预算下一处 Camera-Ray Cross-Attention 是否有可测收益？ — **Not Run**
- 总体结论：**Not Run**
- 负面结果与协议偏差：见 `V3_DEVIATIONS.md`（#1–#6，均为工程偏差）。

## 2. Experimental Protocol and Data Provenance（Completed：数据审计；训练数据构建 Running）
| Split | Scene / house IDs | 轨迹 | 候选视角 | 独立性 |
|---|---|---|---|---|
| MP3D train | 61 个官方 train 房屋的 114 个楼层（E2 轨迹作业 `purpose = train`） | 228 | 263,358 | 场景级与 val / test 不交；每楼层 1 条随机 + 1 条 GT 一步贪心轨迹 |
| MP3D dev（选择） | 6 个 val 房屋 9 个楼层（VALSEL） | 9 | 10,473 | 与 train 不交，也不是闭环评估楼层 |
| MP3D 闭环评估 | 5 个 val 楼层（Z6MF、oLBM、QUCT、2azQ、TbHJ）× 2 起点 | — | — | 与 train / VALSEL 不交；T0.11 为可视化挑选，非盲 |
| Macarons++ train | Train-6（bannerman、dunnottar、fushimi、liberty、pisa、redeemer）去掉 S-c 起点 | 138 | 98,887 | |
| Macarons++ dev | S-c（Train-6 同场景的留出起点） | 42 | ⟨待填⟩ | **同场景留出起点，不是未见场景** |
| Macarons++ 闭环评估 | Test-3（alhambra、bridge、neuschwanstein）+ Dev-3（barts、pantheon、colosseum）× 5 起点 | — | — | Dev-3 在 Task 2 / E1.2 / E2 中用于开发；Test-3 已被查看过，均**不是盲集** |

- 轨迹策略：MP3D `rnv`（随机）/ `ogr`（GT 一步贪心，GT 只塑造轨迹）；Macarons++ T1.1 的 `cg-*` 轨迹。
- 输入 `P1_100`：历史记录 0..t 的 D100 融合（BAND = 全部有效像素，EMPTY 挖空不变）；`P1_10`：原始 10% 采样点，只用于 DistP1。候选 t\* 只作标签。
- 无泄漏测试：缓存构建时每个状态的 V2 粗网格与 E2 分片逐位核对；`g0_real_data_check.py` 缓存 = 实时聚合逐位（5 个状态）。
- 物理尺度：见 `../task0_vae/report/T0_AUDIT.md` §2（MP3D 体素 8.7–17.6 mm；Macarons++ 0.11–0.33 scene units，非米）。

## 3. Architecture, Losses and Scratch Training（Completed：定义；训练 Not Run）
| 模块 | T0 V3-VAE | T1 V3-VAE-XAttn |
|---|---|---|
| VAE Conv3D Encoder（μ, logvar 16 ch @8×） | scratch | scratch，公共初始化哈希相同 |
| 3D Decoder（仅训练） | scratch | scratch |
| Ray Encoder（V2 结构，输入 μ） | scratch | scratch |
| 2D Rectified Flow U-Net（V2 结构） | scratch | scratch |
| Camera-Ray Cross-Attn | 无 | scratch；瓶颈处一块，`W_o` 零初始化 |
| DistP1 评分 | P1_10，固定 | P1_10，固定 |
| 优化 | L_FM + λ L_recon + β KL | 完全相同 |

参数：T0 23,539,622（推理 22,954,434；Decoder 585,188 仅训练）；T1 = T0 + Cross-Attn ⟨待填⟩。KL 调度、优化器、步数等：`../protocol.md` v1.0（G0 后冻结）。

## 4. G0（Running）
见 `../task0_vae/report/T0_G0_TESTS.md`、`../task1_xattn/report/T1_G0_TESTS.md`。

## 5. G1 — Not Run
## 6. G2 — Not Run
## 7. G3 — Not Run
| Dataset | Metric | B0 historical | B2 historical | E2-D100 historical | V3-T0 | V3-T1 |
|---|---|---:|---:|---:|---:|---:|
| MP3D | AUC | 0.322 | 0.334 | 0.364 | — | — |
| MP3D | Final | 0.469 | 0.530 | 0.553 | — | — |
| Macarons++ | AUC | 0.729 | 0.730 | 0.711 | — | — |
| Macarons++ | Final | 0.884 | 0.906 | 0.880 | — | — |
| Macarons++ | Cov@20 | 0.622 | 0.572 | 0.570 | — | — |

历史值来自 Task 3 / E2 报告，不重新评估。**MAGICIAN 参照（Task 2 共享动作协议闭环，不重跑）**：MP3D AUC@150 MAG-G 0.278 / MAG-B 0.292，Final 0.377 / 0.377；Macarons++ 的 MAG-G / MAG-B 行在 G3 分析时从 Task 2 闭环读取并与 T0 / T1 同表列出（AUC、Final、Cov@20）。

## 8. Main Comparisons and Attribution — Not Run
## 9. Training & Engineering Costs — 见 `V3_RESOURCES.md`（Running）
## 10. Limitations, Deviations and Final Decision — 待填；最终状态 **Not Run**
