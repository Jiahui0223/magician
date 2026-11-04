# BASELINE_REPRODUCTION — MAGICIAN 基线：论文值 / 本地上游 LMDB / 本地冻结协议（自动生成，`scripts/baseline_report.py`）

**三类数值不可混用**：
- 论文值：Magician.pdf 第 12 页表 6 / 7，AUC 的归一化方式论文没有说明；
- LMDB：我们本地运行上游 `test_magician_planning.py` 的轨迹（未设种子，只作参照），AUC = trapz / 100；
- 本表 MAG-B：冻结 M 协议，带种子，与我们的模型同一个模拟器和评价器。

| 场景 | 划分 | 论文 AUC / Final | LMDB AUC / Final | 本地 MAG-B 官方 AUC / Final | 本地 MAG-B 单调 AUC / Final | MAG-B 起点数 |
|---|---|---|---|---|---|---|
| dunnottar | train | 0.745 / 0.975 | 0.721 / 0.968 | 0.721 / 0.973 | 0.731 / 0.986 | 5 |
| colosseum | dev | 0.704 / 0.872 | 0.701 / 0.872 | 0.710 / 0.870 | 0.733 / 0.900 | 5 |
| bannerman | train | 0.761 / 0.917 | 0.764 / 0.895 | 0.764 / 0.896 | 0.818 / 0.980 | 5 |
| pantheon | dev | 0.644 / 0.842 | 0.551 / 0.743 | 0.561 / 0.740 | 0.580 / 0.765 | 5 |
| redeemer | train | 0.793 / 0.973 | 0.794 / 0.973 | 0.784 / 0.977 | 0.791 / 0.982 | 5 |
| liberty | train | 0.797 / 0.947 | 0.797 / 0.949 | 0.792 / 0.947 | 0.812 / 0.969 | 5 |
| pisa | train | 0.723 / 0.941 | 0.726 / 0.937 | 0.721 / 0.941 | 0.738 / 0.959 | 5 |
| fushimi | train | 0.766 / 0.931 | 0.775 / 0.937 | 0.781 / 0.940 | 0.815 / 0.990 | 5 |
| alhambra | test | 0.631 / 0.852 | 0.627 / 0.841 | 0.616 / 0.849 | 0.638 / 0.881 | 5 |
| neuschwanstein | test | 0.608 / 0.848 | 0.577 / 0.836 | 0.565 / 0.797 | 0.576 / 0.812 | 5 |
| eiffel | dev | 0.754 / 0.923 | 0.750 / 0.908 | 0.766 / 0.931 | 0.784 / 0.957 | 5 |
| bridge | test | 0.705 / 0.955 | 0.749 / 0.959 | 0.732 / 0.957 | 0.752 / 0.989 | 5 |
| sofia_church | other_unassigned | 0.710 / 0.891 | — / — | — / — | — / — | 0 |
| barts | dev | 0.831 / 0.996 | 0.822 / 0.997 | 0.828 / 0.996 | 0.832 / 0.997 | 5 |
| sestino_museum | dev | 0.637 / 0.924 | 0.602 / 0.867 | 0.611 / 0.878 | 0.620 / 0.888 | 5 |

**14 个场景均值**（同时有论文值、LMDB 与 5 个本地 MAG-B 起点的场景）：
- 论文：AUC 0.721 / Final 0.921
- LMDB：AUC 0.711 / Final 0.906
- 本地 MAG-B（官方口径）：AUC 0.711 / Final 0.906
- 本地 MAG-B（单调口径）：AUC 0.730 / Final 0.932

## 回归重放（当前代码重跑 barts_s0，与历史 episode 比较）

```
B0 executed_keys_equal True official_coverage_equal True hist work_dirs/exp5_future_view/task2/eval/outdoor/closed_loop/V2M/episodes/ep_V2M_barts_s0_r0.json
V2O executed_keys_equal True official_coverage_equal True hist work_dirs/exp5_future_view/task2/eval/outdoor/closed_loop/V2O/episodes/ep_V2O_barts_s0_r0.json
B2 executed_keys_equal True official_coverage_equal True hist work_dirs/exp5_future_view/task3_full_depth_p1/closed_loop/outdoor/B2/V2M/episodes/ep_V2M_barts_s0_r0.json
E2D executed_keys_equal True official_coverage_equal True hist work_dirs/exp5_future_view/task3_full_depth_p1/closed_loop/outdoor/E2D/V2M/episodes/ep_V2M_barts_s0_r0.json
V4 executed_keys_equal True official_coverage_equal True hist work_dirs/exp6_v4/eval/closed_loop/outdoor/V4/V2M/episodes/ep_V2M_barts_s0_r0.json
V5 executed_keys_equal True official_coverage_equal True hist work_dirs/exp6_v5/eval/closed_loop/outdoor/V5/V2M/episodes/ep_V2M_barts_s0_r0.json
V3 executed_keys_equal True official_coverage_equal True hist work_dirs/exp6_v3/eval/closed_loop/outdoor/T1/V2M/episodes/ep_V2M_barts_s0_r0.json
MAG-B executed_keys_equal True official_coverage_equal True hist work_dirs/exp5_future_view/task2/eval/outdoor/closed_loop/MAG-B/episodes/ep_MAG-B_barts_s0_r0.json
REPLAY_DONE B0 V2O B2 E2D V4 V5 V3 MAG-B 2026-11-03 17:39:16
```

## 历史报告值（单调口径，Dev-3 + Test-3）由导入记录重算

| 模型 | 单调 AUC@100 | 单调 Final | 官方 AUC@100 | 官方 Final |
|---|---|---|---|---|
| MAG-B | 0.685 | 0.891 | 0.669 | 0.868 |
| MAG-G | 0.704 | 0.905 | 0.686 | 0.882 |
| V2O | 0.724 | 0.898 | 0.703 | 0.868 |
| B0 | 0.729 | 0.884 | 0.709 | 0.854 |
| B2 | 0.730 | 0.906 | 0.710 | 0.879 |
| B1 | 0.691 | 0.869 | 0.671 | 0.838 |
| E2S | 0.719 | 0.880 | 0.699 | 0.851 |
| E2D | 0.711 | 0.880 | 0.691 | 0.850 |
| V3 | 0.712 | 0.883 | 0.692 | 0.854 |
| V3.1 | 0.697 | 0.865 | 0.676 | 0.838 |
| V4 | 0.710 | 0.878 | 0.689 | 0.849 |
| V4C | 0.713 | 0.882 | 0.693 | 0.853 |
| V5 | 0.711 | 0.891 | 0.689 | 0.861 |
