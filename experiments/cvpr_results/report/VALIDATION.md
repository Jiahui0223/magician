# VALIDATION — 自动质量关口（生成于 2026-11-10 15:18:48，git f3007a09972e）

由 `experiments/cvpr_results/scripts/validate.py` 生成；每一项都从磁盘上的文件重新计算。PENDING = 相关运行尚未完成，不能判定，**不等于通过**。

| 关口 | 检查项 | 结果 | 细节 |
|---|---|---|---|
| 1 | T0 checkpoint sha256 == 7ed0ddd7... (plan value) | **PASS** | /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt |
| 1 | T0 has no cross-attention (model_cfg.xattn.enabled false, 0 unet.xattn tensors in the EMA weights) | **PASS** | step 40000, flow {'x_ch': 2, 'euler_steps': 10, 'K': 4} |
| 1 | 70 / 70 T0 + D1 episodes: planner handshake = T0 sha, K 4, R10, frozen D-v1 policy d1 (depth 2, lambda 0, eps 1.0, cell 1.0) with the current frozen file hashes | **PASS** | 70/70 ok |
| 1 | LeakGuard: no denied file access; the only dataset file the planner process opens is the official settings.json (no mesh / GT file) in the 70 T0 + D1 episodes | **PASS** | violations 0; dataset files opened: ['settings.json'] |
| 2 | identical (scene, start) grid for the complete Table 1 methods ['RANDOM', 'MACARONS', 'MAG-B', 'V3-T0-D1'] | **PASS** |  |
| 2 | every COMPLETE record: 101 finite values, official replay bitwise equal to the stored curve (hash of replay == hash of record) | **PASS** | 1050/1050 records |
| 2 | no FAILED record enters an aggregate (FAILED rows are excluded by status) | **PASS** | 0 FAILED rows in the manifest |
| 2 | V3-T0-D1 All-14 recomputed == historical verified values (0.42, 0.646, 0.633, 0.763, 0.911) | **PASS** | recomputed (0.42, 0.646, 0.633, 0.763, 0.911) |
| 2 | MAG-B All-14 recomputed == historical verified values (0.307, 0.537, 0.544, 0.711, 0.906) | **PASS** | recomputed (0.307, 0.537, 0.544, 0.711, 0.906) |
| 2 | V3-T0-D1: this library == the independent Table 1 aggregator (table1/results/per_model.csv) | **PASS** | max abs diff 2.22e-16 |
| 2 | MAG-B: this library == the independent Table 1 aggregator (table1/results/per_model.csv) | **PASS** | max abs diff 2.22e-16 |
| 3 | Table 1 values == recomputation from manifest/episodes.csv (the immutable intermediate) | **PASS** | 28 cells, max abs diff 0.0e+00 |
| 3 | Table 1: BLOCKED baselines are blank, no paper-reported number is inserted | **PASS** |  |
| 3 | Table 1: methods with < 70 episodes show a dash (never a partial All-14 number) | **PASS** | partial now: [] |
| 3 | Table 2 values and paired differences == recomputation from the manifest; split sizes 6 / 5 / 3 / 8 from the registry | **PASS** | max abs diff 2.8e-16 |
| 3 | documentation of n scenes / n starts / formula / CI method | **PASS** | manifest/protocol.json |
| 4 | Table 3: the four cells have the same 14 scenes x 5 starts | **PASS** | {'V3-T0-K1D1': 70, 'V3-T0-K4D1': 70, 'V3-T0-K1D2': 70, 'V3-T0-D1': 70} |
| 4 | Table 3: every recorded episode ran the intended K / depth with the same T0 checkpoint, frozen policy files, value_cov parameters, lambda 0, R10, noise map 0 | **PASS** | 280 episodes checked, 0 mismatches |
| 4 | Table 3: every K = 1 run used the intended noise map: planner handshake noise offset = j, and at t = 0 its per-candidate values are closest (ties allowed) to sample j of the K = 4 set | **PASS** | closest 560/560 (of which ties 24), handshake 560/560; median relative difference 0.0030 |
| 4 | runner regression: K4 / depth 2 reproduces V3-T0-D1 and K4 / depth 1 (T1 weights) reproduces V3-H1 bitwise | **PASS** | /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/overnight_20261110/logs/regression_check.txt |
| 5 | Fig. 4: every pool state passes the exactness guards (4x grid hash and per-sample values equal the planner log) | **PASS** | 180 states |
| 5 | Fig. 4: selected examples are eligible under the pre-declared rule, cover train / dev / test, include success and hard cases | **PASS** | ['train / success: bannerman s0 t0', 'train / hard: bannerman s2 t30', 'dev / hard: pantheon s1 t50', 'dev / success: pantheon s4 t20', 'test / success: alhambra s3 t30', 'test / hard: alhambra s4 t20'] |
| 5 | Fig. 4: the four panels are the K = 4 samples of ONE state and ONE candidate (same stored arrays), one shared colour scale; the real depth is the next observation, read only for display | **PASS** | fig4_scan.py reads service records + planner logs only (no mesh / GT loader) |
| 6 | Fig. 5: 14 scenes in the frozen order, values == manifest, one common 0-1 radial scale, no scene dropped | **PASS** | max abs diff 0.0e+00 |
| 7 | re-running make_tables.py + make_figures.py reproduces every CSV / TeX / PDF / PNG byte-for-byte | **PASS** | 37 files, 0 differ [] |

合计：PASS 25，FAIL 0，PENDING 0。

## 复现命令

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
bash experiments/cvpr_results/scripts/run_all.sh        # manifest -> tables -> figures -> Fig. 4 -> validation -> checksums
```

关口定义见 `cvpr_overnight_experiment_plan.md` §10：1 主方法权重 / 无 xattn / D1 冻结配置；2 同一 scene / start 配对、101 点曲线、官方重放逐位一致、与历史核验值三位小数一致；3 每张表的来源、n、划分、公式、CI，不混入论文数字；4 Table 3 四格同一评估集合；5 Fig 4 同一状态同一候选、同色尺、真实生成的深度、GT 不进在线规划器；6 雷达图 14 轴同尺度、不隐藏场景；7 所有产物由同一份不可变中间记录生成、重跑一致。
