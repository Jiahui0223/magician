# Experiment A — 实现审计（2026-11-01）

## 1. 真实调用链（代码实测）
| 环节 | 位置 | A0（V4 原 G3，`T3_P1_MODE=b2`） | A1（`T3_P1_MODE=full`） |
|---|---|---|---|
| 历史真实深度 → P1 | `exp3b/m_p1.P1Fusion`（S10：模拟器存储的 10 % 采样点进 BAND + 全深度 carving FREE）；`task3/scripts/p1_full.P1FusionFull`（D100：10 % 采样点 **+** 其余全部有效像素反投影进 BAND，FREE 相同）；`P1FusionB2` = Full + 同记录的稀疏孪生 `self.sparse`。两者都只 `integrate_record` 已执行的记录（规划器子进程每步收到执行后的 record；`a_fixed.py` 断言 `fus.n_frames == t + 1`） | `P1FusionB2`：输入体 / 结构 = Full | `P1FusionFull` |
| 模型条件 | `v2_plan_core.Geometry.from_fusion(fr, dims, dom, fu)` → `SceneCaster(fu.input_volume())`、`vol_obs = cast_volume(...)`、`struct = fu.structure()`；`geo.coarse()` = `G.coarse_grid(struct, domain)` → V4 编码器输入；`geo.view(R, T)` → 投影统计 `pr`（11 通道）与 `t_known` | 全部来自 **Full** 融合 | 全部来自 **Full** 融合（逐位相同，`tests/test_single_variable.py`） |
| 预测 → 命中 | `score_state`：`core.pred.cond / sample`（K = 4，共享噪声）→ `G.decode(b, y, o, d, pr, geo.struct, geo.domain)`（稠密 `t_known` 之前的 UNKNOWN 段放命中） | 与 band_dil 无关 | 相同（同噪声种子 → 逐位相同；`a_fixed.py` 用同一命中对象评分两次） |
| DistP1 | `scoring.new_set_from_depth(t, o, d, dz, k_M, dims, domain, **band_dil**, z0)`：命中体素 ∉ `band_dil`（3×3×3 膨胀 BAND）为"新"，权重 min(1, (z/z0)²)，唯一体素，K 均值 | `band_dil` = **稀疏孪生**（`p1_full._patch_geometry_b2` 在几何建好后只替换 `caster.band_dil`） | `band_dil` = **Full 融合自身**的 `caster.band_dil` |
| 其余 P1 字段 | `score_state` / 规划器还读 `geo.struct`（解码）、`geo.domain`、`pr`（UNKNOWN 射线、n_u）、`t_known`：全部来自 Full 融合，两组相同 | | |
| 规划器子进程 | `exp6_v4/scripts/v4_{in,out}_planner.py` → `t23_v2_in_planner` / `t24_v2_out_planner`（冻结）+ `p1_full.patch()`（按 `T3_P1_MODE`）；模拟器 `sim_{in,out}_v4.py` 通过 `V4_P1_MODE` 设置（默认 b2） | | |

结论：**A1 与 A0 的唯一差别就是 `caster.band_dil`**（稀疏孪生 vs 稠密）；两组的 S10 / D100 都来自同一批已执行记录的真实融合流程（不是从 D100 删 90 %），模型输入、投影、解码、K、噪声、候选、n_win、z0、去重、合法性都不变；GT 不进入评分。

## 2. G0 结果
| 检查 | 结果 |
|---|---|
| A0 复现原 V4 Score / Chosen Camera / Coverage | `a_g0_check.py`：MP3D Z6MFQCViBuw L0 s6 与 Macarons++ barts s0 各 3 步，位姿、覆盖率、每步 DistP1 分数与存档 V4 episode **逐位相同**（`smoke/g0_check.json`） |
| A1 与 A0 的 Future Depth 逐位一致 | 代码：band_dil 不参与 cond / sample / decode；`a_fixed.py` 对同一命中对象评分两次，并断言稠密评分 = 规划器进程内 `distp1_k`；`tests/test_single_variable.py`：输入体 / 结构 / 域 / 投影体 / 粗网格逐位相同 |
| 唯一差别 = band_dil | 同上；稠密 band_dil 体素数 > 稀疏（测试输出） |
| D100 band_dil / BAND 数 / 预测 Hit 被排除比例合理变化 | 见 `A_FIXED_STATE.md`（逐状态 band_dil 体素数与被判已知比例） |
| 没有使用未执行候选的真实观测 | 融合只吃执行后的 record；`fus.n_frames == t + 1` 断言；GT 只在评估侧（oracle 增益 / 覆盖率） |
| A1 冒烟 | 两域各 3 步运行正常；分数每步不同；室内从第 1 步起选择不同，室外 3 步选择相同 |
