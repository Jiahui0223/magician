# Experiment XA -- event log (facts, deviations, decisions; times MST)

## 2026-11-05
- 19:05 task received (experiments/exp6_v3/XA_Experiment_Protocol.md). Slurm: own allocations 3461041 (gpu114, ends 11-08 18:30) and 3465477 (gpu115, ends 11-09 15:12),
  4 x L40S each; QOS `gpu` MaxTRESPU gres/gpu=8 is filled by them; every other L40 node is allocated to other users.
- CT state at that time: CT-M continuation (experiment_CT_continued_training/runs/ct_m) running on gpu114 since 18:57 from the 40K parent; the V3 trainer has no signal-save handler, it writes
  the full state (model, EMA, AdamW, step, RNG) atomically every 1000 steps. No signal was sent; the periodic checkpoint is used.
- 19:20 CT-M wrote ckpt_last.pt at step 41000; 19:22 snapshot copied to work_dirs/.../experiment_CT_continued_training/archive_paused/snap_041000 (read-only), sha256 3b657431...,
  strict load of EMA and raw weights OK. CT-M keeps running on gpu114 only while XA is being implemented (the GPUs would otherwise idle); it is archived again and stopped at a
  checkpoint boundary before XA training starts there.
- 19:10-19:20 seen2scene/v3/xa implemented (vae_pyramid, memory, queries, attention, xa_model); six model / train configs written.
- 19:14 engineering checks (tests/xa_checks.py, results/engineering_checks.json): 6 / 6 PASS -- common-module init hash == T1 (6a9d6071...), cond and velocity == T0 bitwise at init,
  context finite and deterministic, gradients reach every new parameter and the VAE encoder, samples are 0 outside UNKNOWN rays.
  First run of the check reported zero gradients: the check perturbed only W_o while the flow output conv is zero-initialised too (V2 / V3); the check was corrected, the model was not changed.
- 19:20 CT's Y50K closed loop on gpu115 (started 19:08 to use idle GPUs) was stopped by me: it competed for memory with the XA smokes (5 planner OOM failures). CT evaluation of Y50K is INCOMPLETE.
  The stop command also killed its own shell (pattern matched the issuing command line); no other process was affected (checked).
- 19:20 FROZEN_PROTOCOL.md (XA-v1) written before any XA training step.
- 19:27 tiny-overfit smokes (trainer --smoke overfit, 200 steps on a fixed 8 + 8 view batch, 1 GPU each): 6 / 6 exit 0; loss 3.640 -> last-20 mean 0.685-0.732 (drop 80-81 %); the trainer's
  gradient check passed for every variant. Step-1 loss is identical for all six (3.6397971570789975): the new block is zero-initialised, so the six networks start as the same function.
- 19:27 largest-scene memory (tests/xa_bigstate.py, results/bigstate_memory.json; dunnottar t = 99, 4x grid 278 x 398 x 64, 8 views, one forward + backward, bf16):
  T1 16.40 GB / 1.03 s; xa_s2d 16.51 / 0.96; xa_sr 16.53 / 1.07; xa_si 16.53 / 1.04; xa_m2d 17.59 / 1.06; xa_mr 17.61 / 1.08; xa_mi 17.61 / 1.19.
  The M memory costs about 1.2 GB on the largest state (the 4x encoder feature is kept for the local samples). T1 trained at a 33 GB peak per rank on the same 46 GB cards, so no
  micro-batch change is needed; a separate 4-rank profile was not run (the real runs log s / step and peak memory every 50 steps).
- 19:33 regression of the XA evaluation chain with the V3-T1 checkpoint (tests/xa_regression.py, results/regression.json): PASS.
  Outdoor barts s0, D1, 100 steps: executed poses and all 101 official coverage values equal the CVPR Table 1 record V3-D1-H3-NORISK bitwise; official replay == stored curve; same checkpoint sha.
  Indoor Z6MFQCViBuw_L0 S6, H0, 150 steps: executed poses, picks and all 151 C_obs values equal the archived exp6_v3 T1 episode bitwise.
  So the XA chain changes only the model loader; the T1 + D1 anchor (70 / 70 Table 1 records) is reused, not rerun.
- 19:31 short closed-loop smokes with the six overfit-smoke EMAs (12 moves): outdoor D1 + official replay 6 / 6, indoor H0 6 / 6, all exit 0. The six runs execute the same poses
  (final coverage 0.5054 outdoor, 0.1206 indoor): checked that each planner loaded its own checkpoint (six different sha256) and that the scores differ between variants (by about 3 on
  values around 740-1060). The smoke EMAs are 200-step EMAs (decay 0.999) of networks that start as the same function, so they are still nearly identical; this is not a result.
- 19:34 the CT counterfactual workers on gpu115 were stopped (wrappers first, then workers; they write each trajectory atomically): 26 of 63 trajectories complete, resumable. gpu115 free.
