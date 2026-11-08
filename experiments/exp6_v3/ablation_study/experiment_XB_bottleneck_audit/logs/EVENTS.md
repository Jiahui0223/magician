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
