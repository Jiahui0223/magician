# MISSING — 尚缺的 episode 与可直接执行的下一步（生成于 2026-11-10 05:23:33）

| 方法 / 配置 | 完成 | 缺失 | 失败（放弃） | 状态 |
|---|---|---|---|---|
| RANDOM（Random Walk） | 24/70 | 46 | 0 | 部分完成 |
| SCONE（SCONE） | 0/70 | 0 | 0 | BLOCKED |
| MACARONS（MACARONS） | 0/70 | 70 | 0 | 未开始 |
| FISHERRF（FisherRF） | 0/70 | 0 | 0 | BLOCKED |
| MAG-B（MAGICIAN (beam 10x10)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-D1（Ours (V3-T0 + D1)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K4D1（V3-T0, K = 4, depth 1） | 20/70 | 50 | 0 | 部分完成 |
| V3-T0-K1D2（V3-T0, K = 1, depth 2） | 18/70 | 52 | 0 | 部分完成 |
| V3-T0-K1D1（V3-T0, K = 1, depth 1） | 0/70 | 70 | 0 | 未开始 |
| MAG-G（MAGICIAN (greedy)） | 70/70 | 0 | 0 | 完成 |
| V3-D1-H3-NORISK（V3-T1 + D1 (cross-attention control)） | 70/70 | 0 | 0 | 完成 |
| V3-T0-K1D2-N1（V3-T0, K = 1, depth 2, noise map 1） | 0/70 | 70 | 0 | 未开始 |
| V3-T0-K1D2-N2（V3-T0, K = 1, depth 2, noise map 2） | 0/70 | 70 | 0 | 未开始 |
| V3-T0-K1D2-N3（V3-T0, K = 1, depth 2, noise map 3） | 0/70 | 70 | 0 | 未开始 |
| V3-T0-K1D1-N1（V3-T0, K = 1, depth 1, noise map 1） | 0/70 | 70 | 0 | 未开始 |
| V3-T0-K1D1-N2（V3-T0, K = 1, depth 1, noise map 2） | 0/70 | 70 | 0 | 未开始 |
| V3-T0-K1D1-N3（V3-T0, K = 1, depth 1, noise map 3） | 0/70 | 70 | 0 | 未开始 |

## 逐项缺口与下一步命令

### RANDOM — 缺 46 条

- 缺失：dunnottar:2、dunnottar:3、dunnottar:4、colosseum:2、colosseum:3、colosseum:4、bannerman:2、bannerman:3、bannerman:4、pantheon:2、pantheon:3、pantheon:4、redeemer:2、redeemer:3、redeemer:4、liberty:2、liberty:3、liberty:4、pisa:2、pisa:3、pisa:4、fushimi:2、fushimi:3、fushimi:4、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:1、bridge:2、bridge:3、bridge:4、barts:1、barts:2、barts:3、barts:4、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_RANDOM.txt','w').write(''.join(f'RANDOM:{x}\n' for x in ['dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_RANDOM.txt > /dev/null 2>&1 < /dev/null & done
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### SCONE — BLOCKED

- 原因：no scene-level SCONE planner for Macarons++ in the repositories (only the ShapeNet object demo: MACARONS/demos/scone_demo.py, configs/scone, weights/scone); MAGICIAN's SCONE numbers exist only as paper-reported values
- 需要的工作（未做）：为 SCONE 写 Macarons++ 场景级规划器适配（把 `MACARONS/demos/scone_demo.py` 的占据 + coverage-gain 网络接到冻结 EpisodeSim 的合法动作列表上，参照 `experiments/exp3b_macaronspp_transfer/m_t7_macarons.py` 的 R1–R8 修复），再按 `table1/README.md` 的“加入新模型”流程登记并跑 70 条。

### MACARONS — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_MACARONS.txt','w').write(''.join(f'MACARONS:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_MACARONS.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### FISHERRF — BLOCKED

- 原因：no FisherRF code, weights or cached episodes in the repositories; it needs an online 3D Gaussian-splatting model per step and has no adapter for the frozen simulator
- 需要的工作（未做）：引入 FisherRF 官方代码与 3DGS 依赖（用户负责环境安装），写冻结模拟器适配器（每步在线训练 3DGS、对合法动作列表算 Fisher 信息增益），再登记并跑 70 条。

### V3-T0-K4D1 — 缺 50 条

- 缺失：dunnottar:3、dunnottar:4、colosseum:2、colosseum:3、colosseum:4、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:1、liberty:2、liberty:3、liberty:4、pisa:2、pisa:3、pisa:4、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:2、bridge:3、bridge:4、barts:1、barts:2、barts:3、barts:4、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K4D1.txt','w').write(''.join(f'V3-T0-K4D1:{x}\n' for x in ['dunnottar:3', 'dunnottar:4', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K4D1.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D2 — 缺 52 条

- 缺失：dunnottar:2、dunnottar:3、dunnottar:4、colosseum:2、colosseum:3、colosseum:4、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:1、liberty:2、liberty:3、liberty:4、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:2、bridge:3、bridge:4、barts:1、barts:2、barts:3、barts:4、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2.txt','w').write(''.join(f'V3-T0-K1D2:{x}\n' for x in ['dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D1 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1.txt','w').write(''.join(f'V3-T0-K1D1:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D2-N1 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N1.txt','w').write(''.join(f'V3-T0-K1D2-N1:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N1.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D2-N2 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N2.txt','w').write(''.join(f'V3-T0-K1D2-N2:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N2.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D2-N3 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N3.txt','w').write(''.join(f'V3-T0-K1D2-N3:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D2-N3.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D1-N1 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N1.txt','w').write(''.join(f'V3-T0-K1D1-N1:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N1.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D1-N2 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N2.txt','w').write(''.join(f'V3-T0-K1D1-N2:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N2.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

### V3-T0-K1D1-N3 — 缺 70 条

- 缺失：dunnottar:0、dunnottar:1、dunnottar:2、dunnottar:3、dunnottar:4、colosseum:0、colosseum:1、colosseum:2、colosseum:3、colosseum:4、bannerman:0、bannerman:1、bannerman:2、bannerman:3、bannerman:4、pantheon:0、pantheon:1、pantheon:2、pantheon:3、pantheon:4、redeemer:0、redeemer:1、redeemer:2、redeemer:3、redeemer:4、liberty:0、liberty:1、liberty:2、liberty:3、liberty:4、pisa:0、pisa:1、pisa:2、pisa:3、pisa:4、fushimi:0、fushimi:1、fushimi:2、fushimi:3、fushimi:4、alhambra:0、alhambra:1、alhambra:2、alhambra:3、alhambra:4、neuschwanstein:0、neuschwanstein:1、neuschwanstein:2、neuschwanstein:3、neuschwanstein:4、eiffel:0、eiffel:1、eiffel:2、eiffel:3、eiffel:4、bridge:0、bridge:1、bridge:2、bridge:3、bridge:4、barts:0、barts:1、barts:2、barts:3、barts:4、sestino_museum:0、sestino_museum:1、sestino_museum:2、sestino_museum:3、sestino_museum:4
- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
python - <<'EOF'
open('work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N3.txt','w').write(''.join(f'V3-T0-K1D1-N3:{x}\n' for x in ['dunnottar:0', 'dunnottar:1', 'dunnottar:2', 'dunnottar:3', 'dunnottar:4', 'colosseum:0', 'colosseum:1', 'colosseum:2', 'colosseum:3', 'colosseum:4', 'bannerman:0', 'bannerman:1', 'bannerman:2', 'bannerman:3', 'bannerman:4', 'pantheon:0', 'pantheon:1', 'pantheon:2', 'pantheon:3', 'pantheon:4', 'redeemer:0', 'redeemer:1', 'redeemer:2', 'redeemer:3', 'redeemer:4', 'liberty:0', 'liberty:1', 'liberty:2', 'liberty:3', 'liberty:4', 'pisa:0', 'pisa:1', 'pisa:2', 'pisa:3', 'pisa:4', 'fushimi:0', 'fushimi:1', 'fushimi:2', 'fushimi:3', 'fushimi:4', 'alhambra:0', 'alhambra:1', 'alhambra:2', 'alhambra:3', 'alhambra:4', 'neuschwanstein:0', 'neuschwanstein:1', 'neuschwanstein:2', 'neuschwanstein:3', 'neuschwanstein:4', 'eiffel:0', 'eiffel:1', 'eiffel:2', 'eiffel:3', 'eiffel:4', 'bridge:0', 'bridge:1', 'bridge:2', 'bridge:3', 'bridge:4', 'barts:0', 'barts:1', 'barts:2', 'barts:3', 'barts:4', 'sestino_museum:0', 'sestino_museum:1', 'sestino_museum:2', 'sestino_museum:3', 'sestino_museum:4']))
EOF
for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_V3-T0-K1D1-N3.txt > /dev/null 2>&1 < /dev/null & done   # dunnottar 需独占一张卡
bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py
```

## 本轮未安排的项目（计划书 P2）

- Ray Encoder 消融：未做（需要重新训练，不在 8 小时窗口内）。
- Runtime 对照（独占 GPU 的逐方法计时）：未做；Table 3 的 CSV 只给出共享 GPU 下的规划器每步决策秒数，不能当作正式 runtime。
- sofia_church（第 15 个场景）：冻结协议 v1.0 下所有方法都不可运行，始终报告 All-14。

