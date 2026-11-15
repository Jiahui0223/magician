# EXP9-SCONE Phase C：Smoke test

## 0. 预先声明的检查项（2026-11-14 16:40 写下，早于任何 SCONE 闭环输出）

两条短轨迹，均经共享 GPU 队列运行（任务名 `scone_0smoke_*`，排在 FisherRF 室内任务之后）：
- 室外：Macarons++ colosseum（Dev 划分）官方起点 0，20 步；
- 室内：MP3D oLBMNvg9in8_L1（HO-12）起点 36，20 步（EXP7 Round-1 smoke 用的同一条）。

| 编号 | 检查 | 通过标准 |
|---|---|---|
| G1 | 端到端运行 | rc = 0；无 hover / 默认动作 / fallback；室内 guard 拒绝列表为空 |
| G2 | 覆盖更新 | 室外：冻结 `t25_out_cov.py` 重放官方覆盖与存档逐位相同；室内：C_obs 重放最大差 ≤ 1e-12，曲线单调 |
| G3 | 动作合法 | 每步执行动作 ∈ 仿真合法列表，且等于 planner 的 pick；室外重放得到的合法列表与运行时记录逐项相同 |
| G4 | 权重 | 两个官方文件 sha256 正确、与 `pretrained_macarons.pth` 逐位相同、未加载 `trained_macarons.pth` |
| G5 | 占据预测 sanity | 有限、非常数；描述性：靠近 GT 表面（室外 < 2 rt，室内 < 1 rt）与远离（> 10 / > 3.5 rt）的 proxy points 的平均占据概率与 AUROC（预期 > 0.5，不设硬门槛） |
| G6 | 候选打分 | 增益有限、≥ 0、同一步内不全相等；描述性：与 oracle 一步真实覆盖增益（评估侧另行渲染）的 Spearman、Top-1 命中 |
| G7 | 无 GT 泄漏 | 室外：planner surface 点云每个点都是本轨迹观测到的 10% 深度采样点；决策函数只接收观测构建的 surface/proxy scene、相机位姿和合法动作键；室内：LeakGuard 无拒绝 |
| G8 | 起点 / 尺度 / 坐标 | 起点 C0 与 Table 1 / EXP7 其他方法相同；预测增益与 oracle 正相关可作为尺度和坐标系正确的间接证据 |
| G9 | AUC 计算 | AUC@T = trapz(C[0..T])/T 与运行器记录一致 |

结果见下文（运行后补写）。

## 1. 运行记录

| 任务 | worker | 开始 | 结束 | rc | 输出 |
|---|---|---|---|---|---|
| `scone_0smoke_a_out_colosseum_s0` | gpu114 GPU 2 | 16:59:26 | 17:02:32 | 0 | `work_dirs/cvpr_results/scone_baseline/smoke/outdoor/`（episode、diag、check、`smoke_check_out.json`） |
| `scone_0smoke_b_in_oLBMNvg9in8_L1_S36` | gpu114 GPU 3 | 16:59:30 | 17:00:59 | 0 | `work_dirs/cvpr_results/scone_baseline/smoke/indoor/`（SCONE/<tag>/{svc, episode.json, planner.jsonl, diag}、`smoke_check_in.json`） |

另在提交前做过一次 CPU 干跑（室内 planner 喂 MAG-B 存档的 svc_t0/t1，只验证代码路径，不进结果）：权重加载、占据场、8 个候选增益均正常。

## 2. 检查结果

| 编号 | 室外 colosseum s0（20 步） | 室内 oLBMNvg9in8_L1 S36（20 步） | 判定 |
|---|---|---|---|
| G1 运行 | rc 0；hover 0、默认动作 0 | rc 0；fallback 0；guard 拒绝 0 | PASS |
| G2 覆盖 | `t25_out_cov` 重放官方覆盖与存档**逐位相同**（`official_equal_stored = True`）；第二次独立重放也逐位相同；官方曲线最大回落 0.0015（cell cap 机制，Table 1 其他方法同样存在） | C_obs 重放最大差 **0.0**，曲线单调 | PASS |
| G3 合法 | 20 步中，重放得到的合法列表与运行时记录逐项相同，执行动作 = pick | 每步执行位姿 = svc 合法列表[pick] | PASS |
| G4 权重 | sha256 三项正确；容器与官方文件逐位相同；`trained_macarons_loaded = False` | 同左（室内用官方 MAGICIAN 仓库里的同一 `pretrained_macarons.pth`，sha 相同） | PASS |
| G5 占据 | 有限、非常数；t = 1/5/10/15/19 时 AUROC（近表面 < 2 rt 对 > 10 rt）0.73 / 0.83 / 0.88 / 0.90 / 0.87；近表面平均占据 0.36–0.58，远处约 0（t = 0 只有 31 个远点，AUROC 0.49 不具参考性） | AUROC（< 1 rt 对 > 3.5 rt）0.65 / 0.59 / 0.58 / 0.66 / 0.64 / 0.66；近表面平均 0.27–0.43，远处 0.13–0.22 | PASS（描述性） |
| G6 打分 | 增益有限、≥ 0、各步不全相等；与 oracle 一步真实增益的 Spearman：t=1 0.68、t=5 0.76、t=10 0.45、t=15 0.41；Top-1 命中 1/4，4 步中 pick 的真实增益都高于候选平均 | Spearman：0.25 / 0.29 / 0.00 / 0.18；Top-1 命中 1/4；pick 真实增益高于平均 2/4 | PASS（描述性；室内相关性弱） |
| G7 泄漏 | planner surface 点云 10,405 点全部是本轨迹观测到的 10% 深度采样点（`n_not_observed = 0`） | LeakGuard 无拒绝；child 只读 svc 记录 | PASS |
| G8 起点/尺度 | C0 与 Random / MACARONS / MAG-B / Ours 的 C0 逐位相同；打分与 oracle 正相关 | 起点与 OURS / MAG-B / RANDOM / OURS-H0 相同，C0 逐位相同 | PASS |
| G9 AUC | 运行器 AUC 0.378861049… 与 trapz 重算逐位相同 | 由 C 曲线直接计算 | PASS |

说明：
- SconeOcc 是 MSE 回归输出，没有 sigmoid，实测范围约 [−0.17, 1.5]（约 12% 的点 < 0）。这是官方行为（`compute_occupancy_probability` 不截断）；覆盖增益只用 occ > 0.1 的点，所以负值不进入增益。从未进入视野的 proxy points 取官方默认值 0.5（未知体积按半占据计入增益，这是 SCONE/MACARONS 的原始设计）。
- 室内 oracle 相关性弱于室外（单条 20 步，样本很少）。只作记录，不据此调参。

## 3. 前 20 步对照（单条轨迹，仅作 sanity，不是结论）

| 方法 | 室外 AUC@20 | 室外 Cov@20 | 室内 AUC@20 | 室内 Cov@20 |
|---|---|---|---|---|
| SCONE（本次） | 0.379 | 0.503 | 0.237 | 0.400 |
| MACARONS（Table 1 存档） | 0.384 | 0.621 | — | — |
| MAGICIAN beam（存档） | 0.356 | 0.601 | 0.222 | 0.336 |
| Ours V3-T0 + D1（存档） | 0.433 | 0.628 | 0.266 | 0.377 |
| Random（存档） | 0.093 | 0.201 | 0.103 | 0.210 |

SCONE 的早期行为落在 MACARONS / MAGICIAN 之间，明显高于 Random，没有出现坐标系或尺度错误的迹象（错误时通常表现为接近 Random 或停在原地）。MAGICIAN 论文 SCONE colosseum 的 100 步 AUC 为 0.589，20 步数字不可与之比较。

## 4. 运行时间与显存

| | 室外 | 室内 |
|---|---|---|
| 单步决策 | 0.3 s（t=0）→ 约 3.1 s（t ≥ 10 后稳定） | 1.1 s → 3.7 s（t=19） |
| 20 步墙钟 | 97 s（含加载约 25 s） | 67 s（含加载） |
| 峰值显存（torch allocated） | 4.0 GB | 2.5 GB |

## 5. Phase D 入队命令与成本估计（未执行）

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I \
  experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py full            # dry run，列出 114 个任务
env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I \
  experiments/cvpr_results/scone_baseline/scripts/scone_queue_tasks.py full --write    # 写入 gpu_queue/pending
```

- 任务：`scone_d_in_00..43_<level>_S<k>.sh`（44 条室内，150 步，闭环 → `scone_validate_in.py`）和 `scone_d_out_00..13_<scene>_s<k>.sh`（70 条室外，100 步，闭环 → `scone_out_cov.py` 官方覆盖重放）。按队列名字排序，排在所有 `frf_*` 之后。
- 可恢复：已完成（室外 metrics 存在 / 室内 validation 为 valid）的任务直接退出 0；中断残留的目录先移到 `_failed/`；失败任务进入 `gpu_queue/failed/`，可把脚本移回 `pending/` 重跑。
- 成本估计（按 smoke 实测外推）：
  - 室外：稳定后每步约 3.1 s 决策 + 约 1.2 s 渲染/建图/覆盖 ≈ 每条 7–8 min，加官方覆盖重放约 2–4 min；70 条约 **12–14 GPU-h**（同一运行器跑 MACARONS 行 70 条实测 8.4 GPU-h，SCONE 早期占据点更多，按上限估）。
  - 室内：每步约 4–5 s（决策随已预测点数增长，t=19 时 3.7 s，proxy points 已接近饱和）≈ 每条 11–13 min，加验证约 1 min；44 条约 **9–10 GPU-h**。
  - 合计约 **21–24 GPU-h**；8 张卡全部可用时墙钟约 3 h；与 FisherRF 共用队列时更长。显存 < 10 GB/任务，独占 GPU 无风险。
- Phase D 完成后还需（另行批准）：汇总脚本（室外写入 Table 1 canonical 记录格式、室内按 EXP7 口径汇总 + 楼层 bootstrap）、`fig/compare_full/` 新版本输出。这些脚本本轮未写。
