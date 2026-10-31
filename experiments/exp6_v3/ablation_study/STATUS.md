# EXP6 V3 Ablation — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 10:29**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| A4 / B2 | `a_analyze.py`（前台）与 `b_analyze.py`（后台，`logs/b_analyze.log`）在全部 1,676 状态上运行 | CPU（gpu117 节点） | — | `experiment_*/report/` | 10:27 | 10:35 |
| A2-重跑 | 6 个回归失配状态原样重跑一次 | gpu117 GPU 0 | 3429774 | `logs/rerun1_fx_*.log` | 09:55 | ✅ 09:57：与本进程第一次逐位相同，与参照仍差 1e-12 → 偏差 #1（数值等同判定） |

## AB0 审计与协议 — ✅
| 编号 | 内容 | 状态 |
|---|---|---|
| AB0.1 | 资源核实：3429774（gpu117，空闲）、3435033（gpu115，空闲）、3435072（排队，不用）；git 工作区干净（`40ceaa7`） | ✅ 09:30 |
| AB0.2 | 缓存审计 `experiment_A_gtdepth_oracle/report/CACHE_REUSE_AUDIT.md`：Pool CL 1,461 状态 oracle / t\* / GTD / V2 样本全部可逐位复用，svc 含完整深度 → D100 重放可行；Pool FX 365 状态缺 GTD 与原始样本 | ✅ 09:45 |
| AB0.3 | `protocol.md` v1.0 冻结（A 三路径、B 评分器 α=0.5 / λ=1.0、预注册 Gate、停滞 / 高 regret 定义、必过检查） | ✅ 09:55 |

## A Experiment A — ✅（10:35）
| 编号 | 内容 | 状态 |
|---|---|---|
| A1 | `shared/ab_score.py` 实现 + 1 状态冒烟 + 单测 | ✅ 09:45：三路冒烟通过，B2 / E2D / T0 / T1 在 EXP6 / E2 已有状态上逐位一致；GTD 与 EXP5 逐位一致；`tests/` 5 passed（指标 / 选择规则 / DepthU） |
| A2 | Pool FX 打分（VALSEL 9 ep + Dev-3 B2 15 ep；回归 vs EXP6 / E2 行）| 🔄 09:46 起；09:53 已 90 + 121 状态；✅ 10:00 完成 90 + 125 状态；209 / 215 逐位，6 个差 1e-12（偏差 #1：数值等同） |
| A3 | Pool CL 打分（室内 40 + 室外 120 来源 episode；Dev-3 B0 回归） | ✅ 10:22：368 + 1,093 状态；Dev-3 B0 回归 147 / 150 逐位 + 3 个 1e-12（偏差 #1）；GTD 与 EXP5 逐位核对见 `A_G0_VALIDATION.md`；8 卡 ≈ 4.8 GPU-h |
| A4 | `a_analyze.py`：A1 / A2 / A3 表、拆分、停滞、高 regret、CI、图 → `A_FIXED_STATE.md`、`A_FAILURE_CASES.md`、`A_G0_VALIDATION.md` | ✅ 10:30：CL MP3D regret B2 0.112 / T1 0.114 / GTD 0.064（Δ/r_pred ≈ 0.43）；CL Macarons++ B2 0.468 / **T1 0.319** / GTD 0.160（B2 Δ/r_pred 0.66 → T1 0.50）；停滞 53 状态 G\* 中位 0.019 pp（候选空间 / 区域限制）；高 regret 状态 B2 20% / T1 16%，GTD 选对 57% / 48% |
| A5 | 阶段报告一 | ✅ 10:35 |

## B Experiment B — 🔄
| 编号 | 内容 | 状态 |
|---|---|---|
| B1 | `b_scorer.py` 纯函数 + 单测 + `B_SCORE_AUDIT.md` | ✅ 10:00（2 passed；审计随 B2 生成） |
| B2 | 固定状态 G2（P0 / P1 / P2 / RiskAware；B2、T1 主，T0 / E2D 对照）→ `B_FIXED_STATE.md` | 🔄 10:28 `b_analyze.py` 运行中（CPU） |
| B3 | Gate 判定 → `B_DECISION.md`；通过则 G3 闭环 → `B_CLOSED_LOOP.md` | ⬜ |
| B4 | 阶段报告二 | ⬜ |

## R 总报告 — ⬜
`reports/V3_ABLATION_SUMMARY.md`、`V3_ABLATION_COMPARISON_TABLES.md`、`V3_ABLATION_DEVIATIONS.md`、`V3_ABLATION_RESOURCES.md`；提交 `[V3][ablation]`。
