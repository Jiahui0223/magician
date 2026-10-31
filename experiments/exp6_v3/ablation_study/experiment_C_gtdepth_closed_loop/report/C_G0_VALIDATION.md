# Experiment C — G0 校验（自动生成 `tests/g0_check.py`；冒烟运行 `tests/g0_run.sh`）

运行目录 `work_dirs/exp6_v3/ablation_study/experiment_C/g0`；室内冒烟 = Z6MFQCViBuw_L0 起点 6，5 步；室外 = barts 起点 0，3 步。**总判定：PASS**

| # | 检查 | 结果 | 细节 |
|---|---|---|---|
| 1 | single_view_gt | ✅ PASS | indoor t* shape (10, 7296) hit 0.991, legal == Pool CL t=0 True, t* == EXP5 oracle.npz bitwise True; outdoor shape (10, 7296) hit 1.000, legal == True, t* bitwise True |
| 2 | gtd_regression_vs_A | ✅ PASS | indoor bitwise True (max rel 0.0e+00, argmax same True); outdoor bitwise True (max rel 0.0e+00, argmax same True) |
| 3 | oracle_gain | ✅ PASS | indoor t=0 gains == oracle.npz True; outdoor mono True, official True; executed-candidate checks indoor {'C1': {'executed_sample_equal': 5, 'executed_gain_equal': 5, 'state_hash_equal': 5, 'pick_in_candidates': 5, 'n': 5}, 'C2': {'executed_sample_equal': 5, 'executed_gain_equal': 5, 'state_hash_equal': 5, 'pick_in_candidates': 5, 'n': 5}} outdoor {'C1': {'executed_sample_equal': 3, 'executed_official_equal': 3, 'executed_mono_equal': 3, 'state_hash_equal': 3, 'n': 3}, 'C2': {'executed_sample_equal': 3, 'executed_official_equal': 3, 'executed_mono_equal': 3, 'state_hash_equal': 3, 'n': 3}} |
| 4 | no_leakage | ✅ PASS | state hash unchanged at every query True; cq keys minimal True; guard denials 0; cq reads indoor cumulative per step [1, 2, 3, 4, 5], outdoor total 3; repeated-query identical True |
| 5 | legality | ✅ PASS | indoor pick in prefer-unvisited set every step True; outdoor pick < n_legal True |
| 6 | monotonicity | ✅ PASS | C_obs non-decreasing True; outdoor cov_mono non-decreasing True; official max drop C1 0.0000, C2 0.0000 |
| 7 | determinism | ✅ PASS | indoor C1 twice identical True; outdoor C2 twice identical True |
| 8 | baseline_replay | ✅ PASS | indoor B2 5 moves: coverage+executed True, scores bitwise True; outdoor B2 3 moves: coverage+keys True, scores bitwise True |
| 9 | smoke | ✅ PASS | finite indoor True outdoor True; online mono == t25 replay True; official replay == stored True; walls indoor C1 16s C2 10s, outdoor C1 41s C2 38s |

## 冒烟计时（每步均值；共享 GPU，非独占）

| 数据集 | 组 | 子进程决策 s | GT 查询 s | 诊断 s | 子进程峰值显存 GB |
|---|---|---|---|---|---|
| indoor | C1 | 1.05 | 0.10 | 0.67 | 7.9 |
| indoor | C2 | 0.00 | 0.62 | 0.04 | 0.0 |
| outdoor | C1 | 0.29 | 0.20 | 1.46 | 0.8 |
| outdoor | C2 | 0.01 | 1.41 | 0.01 | 0.0 |

- 检查 2 的参照 = Experiment A 缓存 `work_dirs/exp6_v3/ablation_study/cache/cl/<dataset>/<sid>.json: gtd_distp1`（t=0 状态与 Pool CL 同 sid）；检查 1 / 3 的参照 = EXP5 `states/<dataset>/<sid>/oracle.npz`。
- 室外官方覆盖允许下降（MAGICIAN 单元格上限规则），`official_max_drop` 仅记录。
