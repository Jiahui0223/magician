# UAVScenes 表（零样本，EXP7）：论文可直接使用的版本（2026-11-13 保存）

由 `experiments/cvpr_results/scripts/make_table_uav.py` 生成，不要手改。重新生成：
```bash
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_table_uav.py
```

## 主表（`table_uav.{tex,pdf,png,csv}`）
| 方法 | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Cov@50 | Final |
|---|---|---|---|---|---|---|
| Random Walk | 0.196 | 0.275 | 0.351 | 0.266 | 0.374 | 0.467 |
| MAGICIAN | 0.304 | 0.469 | 0.651 | 0.440 | 0.722 | 0.879 |
| **Ours** | **0.433** | **0.650** | **0.773** | **0.672** | **0.862** | **0.910** |

**配对差值**（按切块 bootstrap 10,000 次，seed 0，95% CI；完整数据见 `table_uav_paired.csv`）：

| 对比 | AUC@20 | AUC@100 | Final | 前者更高 |
|---|---|---|---|---|
| Ours − MAGICIAN | +0.129 [+0.103, +0.150] | +0.123 [+0.107, +0.142] | +0.031 [+0.018, +0.044] | 6/6 块；AUC@100 为 58/60 条 |
| Ours − Random | +0.237 | +0.423 | +0.443 | 60/60 条 |

逐块结果见 `table_uav_per_tile.{csv,tex}`（补充材料用）。

## 口径
- **切块（UAV-final）：**
  - UAV-v0：AMtown_11、AMtown_12、HKairport_00、HKairport_01；
  - UAV-v1（封洞后，可达的相机位置都不在重建表面下方）：AMvalley_01_v1、HKisland_00_v1（表中用 † 标记）。
  - 切块由几何规则预先确定：260 m 网格，mesh 覆盖率 ≥ 90%。规则见 `exp7/report/UAV_PROTOCOL.md`。
- **运行：** 每块 10 个带种子的起点，每条 100 步。使用冻结的 Macarons++ 室外闭环链，1 runtime 单位 = 2 m。零样本：没有用 UAVScenes 训练或调参。
- **覆盖率：**
  - 用官方 MAGICIAN 覆盖率，240 / 240 条重放与闭环时存储的曲线逐位相同。
  - 归一化用评价器自己的 GT 点算出的可见率（`uav_visibility_official.py`），只有两块 v1 切块的数值因此改变。
  - 修正后所有覆盖率 ≤ 0.955。不做这个修正时，AMvalley_01_v1 的覆盖率会超过 1。
- **指标：** AUC@T = trapz(C[0..T]) / T，Cov@T = C[T]，Final = C[100]。先对每块的 10 个起点取平均，再对 6 块等权平均。
- **数据来源：**
  - `experiments/cvpr_results/exp7/results/uav_final_evalnorm_summary.json`、`uav_final_evalnorm_UAV-final_per_tile.csv`（由 `exp7/scripts/uavscenes/uav_final.py --norm evaluator` 生成）；
  - 原始 episode 在 `work_dirs/cvpr_results/exp7/episodes/uavscenes/`。
- **其他版本（只作记录，不进论文）：**
  - 冻结协议、未修正归一化的版本：`exp7/results/uav_final_summary.*`，Ours / MAGICIAN / Random 的 AUC@100 为 0.794 / 0.668 / 0.361。
  - 原 6 块 v0 的结果：`uav_final_*_UAV-v0-all_per_tile.csv`。
  - 详细报告：`exp7/report/UAVSCENES_RESULTS.md` §4–§5。
- **可视化：** `exp7/fig/uavscenes/` 下有 `overview_uav_v1_s0`、`curves_uav_v1`、`traj_*`。
- 2026-11-13：可视化中的覆盖曲线已改用同样的评价器 GT 归一化（`uav_viz.py` 读取 `visibility_official.json`），与本表一致。
