# E3 events (chronological; facts only; 2026-11-04 unless noted)

| time | event |
|---|---|
| 19:25 | configs/dv2_v1.yaml written (reliable occluders by 3x3x3 support >= 9) before any D-v2 result |
| 19:32 | smoke of v1 on colosseum s1 (Dev; t = 0, 10, 20): switch off == log; ksv1 mode == E2 ON scores; wrong-cut share NOT reduced (t0 23 % -> 23 %, t10 37 % -> 37 %) |
| 19:38 | diagnostic on colosseum s1 / s3 (Dev; logs/diag/wrong_cut_summary.json): wrong cuts = downward rays clipping a corner of a surface voxel (occupied run median 0.5-0.75; 50-60 % leave the layer again; 40-50 % are hits on an already observed surface) -> a neighbour count cannot remove them |
| 19:42 | design changed to the ray-footprint test; configs/dv2_v2.yaml frozen (gates unchanged from v1) before any formal fixed-state or closed-loop run. v1 never run formally. Smoke of v2 on colosseum s1: wrong-cut share t0 23 % -> 6 %, t10 37 % -> 7 %, plateau-phase cuts kept (5796 vs 5810) |
| 19:46 | formal fixed-state runs started (colosseum s0-s4, pisa s1; 4 modes); closed-loop switch-off regressions started (colosseum s0 rep 0 and rep 1) |
| 19:54 | switch off reproduces D1 bitwise in closed loop: colosseum s0 rep 0 == Table 1 D1, rep 1 == experiment E D1 rep 1 (coverage, pose keys, all scores, all picks, 100 steps each) |
| 19:57 | fixed-state gates F1-F5 PASS (results/fixed_summary.json): off == log 120 / 120; ksv1 == E2 120 / 120; D-v2 Top-1 0.38 -> 0.54, selected gain 0.88 -> 0.94 pp (t < 20: 3.48 -> 3.55); wrong-cut share t < 20 0.11 -> 0.02; layer-2 cuts > 0 |
| 19:58 | stage A started: D-v2 colosseum s0-s4 + pisa s0-s4 rep 0; alhambra s0 fixed-state run (report only, after freezing) |
| 20:13 | stage A done (10 / 10, exit 0): colosseum dAUC@20 +0.000, dAUC@100 +0.089, dFinal +0.091, plateau 1 -> 0; pisa dAUC@20 +0.004, dAUC@100 +0.018, dFinal +0.029 -> stage_A positive (pre-registered rule) |
| 20:14 | stage B started: D-v2 rep 1 / 2 for colosseum + pisa (20), paired D1 for pisa rep 1 / 2 with the switch off (10), first-layer-only diagnostic on colosseum rep 0 (5); 2 workers per GPU on 8 GPUs |
| 20:08 | alhambra s0 fixed-state (report only, after freezing): wrong-cut share ksv1 0.28 -> D-v2 0.12; selected gain 0.90 -> 0.93 pp; nothing changed after it |
| 20:47 | stage B done (35 / 35, exit 0): colosseum 15 pairs dAUC@20 -0.008, dAUC@100 +0.097 [0.060, 0.131], dFinal +0.111 [0.061, 0.164], plateaus 5 -> 0; pisa 15 pairs dAUC@20 +0.002, dAUC@100 +0.017 [0.002, 0.035], dFinal +0.027 [0.008, 0.050] -> stage_B_proceed (pre-registered rule). First-layer-only on colosseum rep 0: dAUC@100 +0.047, dFinal +0.037, plateau 1 -> 1 (full: +0.089 / +0.091, 1 -> 0) |
| 20:48 | Dev-5 started: D-v2 barts / eiffel / pantheon / sestino_museum x 5 starts, rep 0 (colosseum rep 0 from stage A); 2 workers per GPU on 8 GPUs |
| 21:08 | Dev-5 done (20 / 20 new episodes, exit 0; 25 pairs): dAUC@20 -0.001 [-0.003, 0.000], dAUC@100 +0.022 [0.004, 0.056] (5 / 5 scenes), dFinal +0.020 [-0.002, 0.056] (4 / 1), plateaus 2 -> 1, plan time 6.13 -> 6.20 s; gates G1-G5 PASS (without colosseum: dAUC@100 +0.005, dFinal +0.002) |
| 21:12 | registered V3-DV2-KSV in CVPR Table 1 (MODEL_REGISTRY.yaml entry appended, t1_queue.sh case added; existing entries verified unchanged); All-14 started through the Table 1 pipeline (70 episodes incl. re-runs of the 30 Dev-5 + pisa rep-0 episodes; dunnottar 1 job per GPU on gpu114, the rest 2 per GPU on gpu115) |
| 21:30 | dunnottar x 5 done (1 job per GPU, 22-38 GB, no OOM); second workers added on gpu114 |
| 22:21 | All-14 done: 70 / 70 canonical records, exit 0, no retries; the 30 re-run episodes (Dev-5 + pisa rep 0) equal the E3 runs bitwise (coverage + pose keys). All-14 dAUC@20 -0.001 [-0.003, +0.002], dAUC@100 +0.010 [-0.001, +0.025], dFinal +0.011 [-0.001, +0.026], plateaus 3 -> 1 |
| 22:30 | Table 1 refreshed (validate, aggregate official + mono, paper_tables refresh: 166 + 371 checks passed); rows of existing models identical before / after (per_model, per_split, per_scene, per_episode) |
| note | colosseum s3 rep 1 early detour diagnosed on the D1 rep-1 state at t = 3 (logs/diag/state_colosseum_s3_r1_t3.json): cuts 98-99 % correct; the descend candidate wins by its second-step value (765 vs 728) |
| 22:45 | housekeeping: raw per-ray diagnostic dumps (2 x 15 MB) moved to work_dirs/.../E3_dv2_closed_loop/diag, summary kept in logs/diag/wrong_cut_summary.json; the only edit to configs/dv2_v2.yaml after freezing is this file path inside the change_from_v1 text (no parameter or gate changed) |
