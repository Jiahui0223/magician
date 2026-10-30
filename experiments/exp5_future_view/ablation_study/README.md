# EXP5 Ablation Study — A1 Prediction–Scoring Alignment / A2 Stochastic Future View Analysis

协议见 `protocol.md`（v1.0，正式运行前冻结），偏差见 `DEVIATIONS.md`。
- **运行结果**：放在 `work_dirs/exp5_future_view/ablation_study/`（W）。
- **Git**：大文件不进 Git；本目录也尚未提交。

## 目录
| 路径 | 内容 |
|---|---|
| `common/ab_common.py` | 常量、模型（路径与 sha256）、状态时间、候选集规则、路径 |
| `common/s0_states.py` | S0 固定状态清单（Task 2 服务记录 → 去重） → `W/manifests/states_<d>.json` |
| `common/s1_oracle_in.py` | S1 室内 oracle：C_obs 一步增益（T0.12-C 规则）+ t*（magician env） |
| `common/s1_oracle_out.py` | S1 室外 oracle：单调 / 官方覆盖一步增益（T7a 快照 / 恢复评估器）+ t*（magician env） |
| `common/s2_infer.py` | S2 规划器状态重建；V2O / V2M 同几何同噪声打分；GTD；逐候选与逐样本指标；K 消融（seen2scene env） |
| `common/ab_load.py` | 逐状态记录、线上选择规则、分层汇总与场景 bootstrap |
| `common/ab_cases.py` | 图例案例的自动挑选（协议 §5） |
| `common/s5_verify.py` | Stage 5 独立校验 |
| `A1_prediction_scoring/a1_analyze.py`、`a1_viz.py` | A1 表格与图 |
| `A2_stochastic_future_views/a2_analyze.py`、`a2_case_geometry.py`、`a2_viz.py` | A2 表格、案例几何、统计汇总图 |
| `A2_stochastic_future_views/a2v2_render.py`、`a2v2_viz.py`、`A2_FIGURE_GUIDE.md` | A2 案例图 v2（照片 / 深度 / 3D 重构，按用户设计）与阅读指南 |
| `reports/ABLATION_SUMMARY.md` | 总结 |

## 运行方式
- `W/run.sh GPU mg|s2 SCRIPT ARGS`：mg = magician env，s2 = seen2scene env，后者带 libtbb 预加载。
- `W/chain.sh`：S1 → S2 main → S2 kabl 全链路。

## 状态
| 步骤 | 状态 |
|---|---|
| Stage 0 审计 | DONE（2026-10-28） |
| S0 状态清单 | DONE：MP3D 368 个、Macarons++ 1,093 个去重状态 |
| S1 oracle + t* | DONE：逐位检查全部通过（MP3D 368/368、Macarons++ 1,093/1,093） |
| S2 推理 | DONE：源模型分数复现线上日志，最大差 0.0 |
| K 消融 | DONE：只在任务书时间点上做（DEVIATIONS #5） |
| A1 / A2 分析与图 | DONE：`A1_prediction_scoring/A1_REPORT.md`、`A2_stochastic_future_views/A2_REPORT.md` |
| 附加诊断 | DONE：区域、相关性、幻象遮挡、GTDn（DEVIATIONS #6–#10） |
| Stage 5 校验 | DONE：0 项失败（`W/verify/verify.json`） |
| 总结 | DONE：`reports/ABLATION_SUMMARY.md` |
| 计算量 | 约 19 进程小时（4 × L40S，共享，非独占） |

## 变更记录
- 2026-10-28：创建；协议 v1.0 冻结。
- 2026-10-28：A1 / A2 完成，报告与校验完成；运行中的偏差见 DEVIATIONS #1–#10。
- 2026-10-28：A2 案例图按用户设计重做（`figures/v2/`，39 张），旧版删除；更正室外评分失败的原因（DEVIATIONS #11）。
