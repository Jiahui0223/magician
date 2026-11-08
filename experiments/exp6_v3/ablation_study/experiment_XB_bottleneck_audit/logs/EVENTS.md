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
