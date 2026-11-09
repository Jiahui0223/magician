# Experiment XB -- event log (facts, deviations, decisions; times MST)

## 2026-11-08
- 16:20 task received (experiments/exp6_v3/XB_Experiment_Protocol.md). Resources: allocation 3461041 (gpu114) ends 18:30 today; 3465477 (gpu115) ends 11-09 15:12. All 8 GPUs idle (XA ended 05:19).
- 16:25 outdoor training data per scene measured from the D100 shards (results in AUDIT_BASELINE.md): 138 training trajectories / 98,887 views in total (the XA / T1 run manifests report
  the rank-0 shard only, 25,225 views); labelled-ray shares bannerman 20.7 %, dunnottar 25.4 %, fushimi 9.9 %, liberty 16.2 %, pisa 16.1 %, redeemer 11.8 %.
- 16:45 Task C scene rule fixed before any training (FROZEN_PROTOCOL.md §2): three morphology groups {bannerman, dunnottar} {liberty, redeemer} {fushimi, pisa}, alphabetically first of each
  -> bannerman, fushimi, liberty (68 trajectories, 46.8 % of the labelled rays). S-c, MP3D, sampling ratio, recipe, seed unchanged; early stop off.
- 16:28-16:31 overfit smoke of scripts/train_xb.py (1 GPU, 200 steps): filter keeps 68 / 138 trajectories, S-c unchanged; loss 3.640 -> 0.727 (drop 80 %), grad check pass, exit 0.
- Deviation from the task book §6 order: Task C training starts now, in parallel with Task A / B, because the gpu115 allocation ends 11-09 15:12 and training + closed loops need about 18.5 h.
  Task C's definition does not depend on A / B results. Task A / B diagnostics run on gpu114 until 18:30, then on gpu115 next to the training (<= 1 diagnostic process per GPU).
- 16:31:49 Task C training launched on gpu115 (xb_chain.sh 29651 xb_t1_red3 train_t1_red3.json; git 570aafc): 68 outdoor trajectories kept, S-c 42 unchanged.
- 16:38 Task A tool scripts/xb_fixed.py (CT evaluator + attention interventions BASE@IV, A1 statistics) tested on one indoor state: T1 residual / input RMS 0.145, attention entropy 3.54 (uniform 3.57);
  XA-MI ratio 0.386, entropy 2.01 (uniform 3.81), 53 % of the attention mass on the local along-ray samples; zero_wo changes the planner scores substantially. Frozen pool (FROZEN_PROTOCOL §3 refined):
  indoor 9 VALSEL trajectories (90 states), outdoor Dev one trajectory per scene (g2_B0_barts_s0, g2_B0_pantheon_s0, e3_colosseum_s0, conf_eiffel_s0, conf_sestino_museum_s0; 80 states),
  Train (conf_bannerman_s0, conf_liberty_s0, e3_pisa_s1; 60 states). 21 cores. Outdoor planner scores in this tool are the CT D-v2 two-step path (ranking agreement only; not D1 numbers).
- 16:42 Task A launched on gpu114 (4 workers, jobs_a.txt, claims; allocation ends 18:30 -- unfinished trajectories will be rerun on gpu115 after the training).
- 16:52 Task B1 (scripts/xb_b1_fidelity.py, 8 outdoor pool trajectories) started on gpu115 GPU 1; Task B2 (scripts/xb_cf.py = CT ct_cf.py with the XB core loader; T1 / XA-SR / XA-MI on 9 indoor + 6
  outdoor trajectories = 45 jobs; colosseum and sestino excluded while the training occupies the GPUs -- planner memory 8.4-8.9 GB) on gpu115 GPUs 2-3. Task D analysis (scripts/xb_d_uncertainty.py)
  done: All-14 sd across the seven near-equivalent models (T1 + six XA) = 0.003 AUC@20 / 0.005 Cov@20 / 0.003 AUC@100 / 0.004 Final (range 0.418-0.425 / 0.648-0.660 / 0.758-0.765 / 0.897-0.910);
  scene-bootstrap paired SE 0.005 / 0.007 / 0.003 / 0.005; within-scene start sd (T1) 0.085 / 0.082 / 0.058 / 0.060 -- the start-to-start noise is 20-30x the model-to-model spread.
- 16:58 Task A interim (indoor, 8 of 9 trajectories, 80 states; results/A_attention/summary.json): T1's cross-attention is inert -- zero_wo / shuffle_tok / shuffle_pe / drop_band / drop_free leave
  precision 0.753, recall 0.790, Brier 0.1057, MAE 1.858 and every pick unchanged (Top-1 agreement 1.000, score change <= 0.5 %); attention entropy 5.745 vs uniform 5.781 over ~467 tokens
  (max weight 0.009), residual / input RMS 0.059. XA-MI uses its block: zero_wo MAE 1.873 -> 2.125, regret 0.147 -> 0.267, pick agreement 0.838; drop_local 0.787 (regret 0.317), drop_f8 0.900
  (0.332), drop_f16 0.950 (0.144), shuffle_tok / shuffle_pe 0.95 / 0.94; residual ratio 0.362, 19 % of attention mass on the local along-ray samples, 50 % on F8, 30 % on F16.
  XA-SR: zero_wo agreement 0.925 (regret 0.165 -> 0.177), ratio 0.457. Normal picks: T1 Top-1 0.463 / regret 0.143, XA-MI 0.450 / 0.147, XA-SR 0.438 / 0.165.
  Regression: T1@none reproduces the CT g1 picks on 80 / 80 states (scores bitwise on 7 / 80 -- the patched forward uses an explicit softmax instead of SDPA; picks identical).
- 17:35 progress: Task C step 3150 (1.4 s / step); Task A 13 / 17 trajectories (all indoor + barts, pantheon, eiffel, colosseum; 20-state outdoor trajectories take 27-31 min each with 21 cores),
  the last 4 outdoor trajectories running on gpu114 (allocation ends 18:30); B1 8 / 8 done; B2 4 / 45 (about 20 min per indoor trajectory next to the training); B3 1 / 17.
- B1 result (8 outdoor trajectories, 9.4-10 M observed points per trajectory at the last state): every observed depth point lies in a BAND voxel of the S10 map and of the D100 map, in a 4x cell
  with band > 0 and in an 8x cell with a scene token -- retention 100.0 % at all four levels and in all distance strata (0-10 / 10-25 / 25-45 / 45-70 runtime units); thin 4x BAND cells
  (<= 2 of 64 fine voxels) are 1.4-3.6 % of the BAND cells; 43-77 % (mean 65 %) of the 4x UNKNOWN cells adjacent to observed cells have an 8x parent that carries a token.
  Interpretation so far: observed geometry is not lost between the depth maps and the latent grid; the information limit is about geometry that was never observed, not about encoding what was.
- Task A interim with outdoor Dev (60 states: barts, pantheon, eiffel, colosseum): T1 zero_wo identical (agreement 1.0, metrics unchanged) -- the T1 attention is inert in both domains.
  XA-MI outdoor: zero_wo agreement 0.833, regret 0.337 -> 0.284 (not worse), drop_f16 0.240, shuffle_tok 0.236, drop_local 0.330 -- the block changes the picks but not systematically for the
  better on these states; indoor it does hurt when removed (0.158 -> 0.266). XA-SR outdoor Top-1 0.417 (T1 0.533, XA-MI 0.517 on the same 60 states); zero_wo 0.467 / regret 0.379.

## 2026-11-08 18:30 — Task A 完成
- 17 / 17 条轨迹（室内 9、室外探索 4、室外确认 4）全部在 gpu114 到期（18:30）前完成，最后一条 e3_pisa_s1 于 17:51:56 结束；无失败、无重跑。
- `xb_a_analyze.py` 在完整池上重跑（`logs/a/analyze_final.log`）：T1@none 对 CT g1 选择一致 228 / 230。
- 结论写入 `A_ATTENTION_AUDIT.md`：T1 注意力惰性（所有干预结果相同）；XA 注意力被使用但方向在域间相反；最有用的 K/V 是局部 F4 沿射线样本（XA-MI drop_local：Dev FNR 0.257 → 0.436）；drop_f16 反而降低 FNR。
- 同时刻其他任务：Task C step 5,500（16:31 开始，0.80 step/s，预计 11-09 ~06:30 训练完、~09:00 闭环完）；B2 8 / 45（T1@none 室内 8 条）；B3 4 / 17。gpu114 到期后 B2 / B3 仅在 gpu115 与训练并行。

## 2026-11-08 18:55 — XB 暂停（用户指令：优先补跑 Table 1 闭环）
- 用户 18:50：先用 `experiments/cvpr_results/table1` 的流水线跑 V3-T0 + D1 的 All-14 闭环，拿到可对比的 AUC / Final；其他实验暂停，闭环跑完后再恢复。
- 停止：Task C 训练链（`xb_chain.sh 29651 xb_t1_red3 train_t1_red3.json`，step 6,800，`ckpt_last.pt` = step 6,000，恢复时重跑 800 步）；B2 队列（gpu 2 / 3，10 / 45 完成）；B3 队列（gpu 0，5 / 17 完成）；看门狗。被中断的 3 个诊断任务（B2 `EU6Fwq7SyZv_L1_ogr_p50__XA_SR@none`、`EU6Fwq7SyZv_L2_ogr_p11__XA_SR@none`；B3 `X7HyMhZNoso_L1_rnv_p71`）的 claim 已清除，恢复时自动重跑。
- 恢复命令（在 gpu115 分配内，`setsid nohup`）：
  - 训练：`bash scripts/xb_chain.sh 29651 xb_t1_red3 train_t1_red3.json`（自动 resume）
  - B2：`bash scripts/xb_b2_queue.sh <GPU> work_dirs/.../jobs_b2.txt`（两卡）；B3：`bash scripts/xb_b3_queue.sh <GPU> work_dirs/.../jobs_a.txt`
- 偏差记录：Task C 的训练被中断一次并从 step 6,000 的 `ckpt_last.pt` 恢复；数据流 / 采样器状态随 checkpoint 恢复，与 XA 的 OOM 重试恢复机制相同。

## 2026-11-08 20:45 — XB 恢复
- Table 1 的 V3-T0-D1 闭环 20:44 完成（70 / 70），gpu115 四卡空出；20:45 按原命令恢复：Task C 训练链 `xb_chain.sh 29651 xb_t1_red3 train_t1_red3.json`（resume，从 step 6,000 续训，首条日志 step 6001）、B3 队列 gpu0、B2 队列 gpu2 / gpu3；看门狗重启。
- 预计：Task C 训练 ~08:30（11-09）完成，选模 + 闭环 ~10:45；分配 15:12 到期。
