# EXP7 UAVScenes — 暂停记录（2026-11-11 17:20，用户决定：让出 GPU 给主线，空闲时一次性补完）

> **已恢复（2026-11-12 19:25 起，走共享独占队列）：** 主表 UAV-final 所需的 6 块切块已全部算完，结果见 `UAVSCENES_RESULTS.md` §5。只有 v0 记录表用的 HKisland_00 在 21:53 时仍在计算。下文保留为当时的记录。

## 1. 暂停时的状态
| 部分 | 状态 | 位置 |
|---|---|---|
| 切块 | ✅ v0 6 块 + v1 封洞 2 块（AMvalley_01_v1、HKisland_00_v1；可达的地表下位置为 0） | `work_dirs/cvpr_results/exp7/uav_macpp/uav_<tile>[_v1]/` |
| 闭环 | ✅ 240 / 240 条：v0 6 块 × 10 起点 × {Ours, Random, MAGICIAN} = 180，加 v1 2 块 × 10 × 3 = 60 | `work_dirs/cvpr_results/exp7/episodes/uavscenes/raw/{OURS/V2M,RANDOM/random,MAG-B/MAG-B}/episodes/` |
| 官方覆盖率重放 | ✅ 240 / 240 逐位相同（`official_equal_stored`） | `work_dirs/cvpr_results/exp7/episodes/uavscenes/metrics/` |
| 结果表（冻结协议，即 `settings.json` 的可见率） | ✅ `results/uav_final_summary.{md,json}`、`results/uav_final_*_per_tile.csv` | 见 `UAVSCENES_RESULTS.md` §4 |
| 可视化（含 MAGICIAN） | ✅ `fig/uavscenes/`：`overview_uav_s0`、`overview_uav_v1_s0`、`curves_uav`、`curves_uav_v1`、`traj_*` | — |
| 地表下诊断（含 MAGICIAN 和 v1） | ✅ `results/uav_under_surface.csv` | — |
| **未完成：覆盖率 > 1 的归一化修正** | 🟡 部分完成，见下 | — |

## 2. 未完成项：可见率与评价器 GT 不一致
- **现象：** v1 切块的官方覆盖率超过 1。AMvalley_01_v1 最高 1.039（Ours 10/10 条、MAGICIAN 6/10 条），单调覆盖率最高 1.047；HKisland_00_v1 单调覆盖率最高 1.007。
- **原因（已核实）：**
  - `settings.json` 的 visibility_ratio（`uav_tile_gpu.py` 第 7 步）是 10 万个按面积均匀采样的 GT 点中可见的比例；
  - 官方评价器（`setup_test_scene` → `gt_scene.fill_cells`）的 GT 是按格截断的子集，约 2.9 万点；
  - 封洞加了 52,750 个塞子面，其中大量是看不见的竖直侧壁。按面积采样时它们占比偏大，可见率被低估。
  - 把并集去重的格子从 0.25 细化到 0.05，面积采样的比例不变（0.86389 对 0.8638），说明不是去重太粗造成的。
- **修正方法（诊断脚本，不改 `settings.json`）：** `scripts/uavscenes/uav_visibility_official.py`
  1. 取所有可达位置的 50 个视角，构建观测并集；
  2. 对每个起点 k，按 `derived_seed(tile, k, -1, "gt")` 和 `setup_test_scene` 生成评价器自己的 GT 点，与 `m_t6_sim.EpisodeSim` 完全相同；
  3. 计算这些 GT 点中落在 eps = 1.0 以内的比例；
  4. 同时给出 ±0.0433（并集格子误差）的上下界。
  - 输出 `uav_macpp/<tile>/visibility_official.json`。
  - 换算：覆盖率' = 官方覆盖率 × visibility_ratio_settings / ratio_k。这对同一块切块上的所有方法是同一个常数，排序和配对符号不变。
- **进度：**

  | 切块 | 评价器 GT 可见率 | 原值（settings） |
  |---|---|---|
  | AMvalley_01_v1 | **0.9607**（各起点 0.959–0.962） | 0.8638 |
  | AMtown_11、AMtown_12、HKisland_00_v1、HKairport_00 | ❌ 17:41 按 PID 停止，未写出 json。FisherRF 试点结束时它们已跑 57–95 min；与 FisherRF 共用 GPU 时变慢，且在 GPU 3 上显存峰值约 42 GB，造成 FisherRF 几次 OOM | — |
  | HKairport_01、AMvalley_01、HKisland_00 | 未开始 | — |

  **共 7 块待算。** 脚本没有进度日志；单卡独占时 AMvalley_01_v1 用了 37.5 min。较大的切块显存可能到 40 GB 以上，所以每张卡只跑 1 个，不要与其他任务共卡。
- **还没写的部分：** 在 `uav_final.py` 中加入按 `visibility_official.json` 换算的版本，即增加 `--norm evaluator` 选项，输出与冻结协议表并列的第二张表。之后更新 `UAVSCENES_RESULTS.md`。

## 3. 重跑命令（gpu114，magician 环境；每块切块约 35–40 min，单卡独占时）
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
# 只补缺的切块（已有 visibility_official.json 的会被重算，所以只传缺的）；暂停时缺 7 块：
for t in uav_AMtown_11 uav_AMtown_12 uav_HKisland_00_v1 uav_HKairport_00 uav_HKairport_01 uav_AMvalley_01 uav_HKisland_00; do
  [ -f work_dirs/cvpr_results/exp7/uav_macpp/$t/visibility_official.json ] || echo "missing $t"; done
# 每张卡一条链，顺序跑传入的切块（日志 work_dirs/cvpr_results/exp7/logs/uav_visoff_<tile>.log）：
setsid nohup bash experiments/cvpr_results/exp7/scripts/uavscenes/uav_visoff_chain.sh 0 uav_HKisland_00_v1 uav_HKairport_00 > /dev/null 2>&1 < /dev/null &
setsid nohup bash experiments/cvpr_results/exp7/scripts/uavscenes/uav_visoff_chain.sh 1 uav_AMtown_11 uav_HKairport_01 > /dev/null 2>&1 < /dev/null &
setsid nohup bash experiments/cvpr_results/exp7/scripts/uavscenes/uav_visoff_chain.sh 2 uav_AMtown_12 uav_AMvalley_01 > /dev/null 2>&1 < /dev/null &
setsid nohup bash experiments/cvpr_results/exp7/scripts/uavscenes/uav_visoff_chain.sh 3 uav_HKisland_00 > /dev/null 2>&1 < /dev/null &
# 4 卡独占时约 1.5 h（每块约 40 min）
# 单块手动运行：
MACARONS_PLUSPLUS_ROOT=$PWD/work_dirs/cvpr_results/exp7/uav_macpp CUDA_VISIBLE_DEVICES=0 env -u LD_PRELOAD -u PYTHONPATH \
  /bsuscratch/jiahuizhang/envs/magician/bin/python -I experiments/cvpr_results/exp7/scripts/uavscenes/uav_visibility_official.py uav_HKairport_01
# 结果表（CPU，seen2scene 环境）：
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/exp7/scripts/uavscenes/uav_final.py
# 可视化（CPU）：
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/exp7/scripts/uavscenes/uav_viz.py --start 0
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/exp7/scripts/uavscenes/uav_viz.py --start 0 --version v1
```
注意：停进程时要按 PID 停，不要用 `pkill -f` / `pgrep -f` 匹配脚本名，否则会把发命令的 shell 自己也杀掉（本次又遇到一次）。

## 4. 闭环本身的重跑（一般不需要；240 条都已完成并验证）
- 闭环队列：`scripts/uavscenes/uav_queue.sh`（方法 `OURS` / `RANDOM` / `MAG-B`）；协调脚本 `uav_v1_coord.sh`、`uav_after_v1.sh`。
- 覆盖率重放：`scripts/uavscenes/uav_cov_queue.sh GPU`，会跳过已有 metrics 的条目。
- 封洞：`scripts/uavscenes/uav_seal_tile.py`；切块构建：`uav_build_tile.py` → `uav_tile_gpu.py`。
