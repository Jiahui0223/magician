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
- 19:34 code, configs, FROZEN_PROTOCOL.md and manifests committed as 7ae012e (manifests/*.frozen.json re-written just before; clean tree).
- 19:34:47 chain gpu115 launched inside allocation 3465477: `run_chain.sh 29642 xa_sr xa_m2d xa_mi` (setsid nohup; log logs/chain_gpu115.log).
- 19:45:08 chain gpu114 launched inside allocation 3461041 after CT-M was archived (step 42000) and stopped: `run_chain.sh 29641 xa_s2d xa_si xa_mr` (log logs/chain_gpu114.log).
- 19:54 both first runs train: xa_sr and xa_s2d report common_init_sha256 == T1, data manifest == T1 (9f432c3f...), world 4, git 7ae012e clean. Step-1 loss 3.674514578066773 is bitwise the
  V3-T1 step-1 loss (same seed, sample stream and initial function); at steps 50-450 the losses follow T1 to 3-4 digits. 1.33-1.44 s / step and 30.8-32.7 GB per rank, the same as T1.
  Expected: about 16.5 h training + 2.5-3 h closed loops per variant; chains end around 2026-11-08 06:00 (gpu114 allocation ends 11-08 18:30, gpu115 11-09 15:12).
- Deviation (FROZEN_PROTOCOL §6): no sbatch / afterok -- the account's 8-GPU QOS limit is filled by these two allocations; the chain scripts provide the retries and the stage order.
- Rule while the chains run: scripts/run_chain.sh and scripts/xa_eval_queue.sh are not edited (bash reads a running script from disk).
- 19:57 scripts/aggregate.py changed after the freeze (analysis code only; no model, training or evaluation code): (1) an episode whose curve is not 101 (outdoor) / 151 (indoor) values long is
  listed as bad and never aggregated (before, it would have raised an error); (2) planner peak memory is read from the planner step logs; (3) test-only overrides XA_TEST_IDS / XA_TEST_W / XA_TEST_OUT;
  (4) the run-manifest glob counts restarts correctly. Tested end to end on a synthetic 70 + 6 episode set built from the Table 1 V3-H3 records and the T0 indoor episodes
  (reproduces the known H3 row 0.424 / 0.762 / 0.901 and the paired differences vs D1), on the REG_T1 output and on a 12-move smoke output (flagged, not aggregated).
  manifests/code_hashes.json is refreshed; code_hashes.frozen.json keeps the version of 19:34 (the difference is aggregate.py only).
- 21:54 progress: xa_s2d step 5100, xa_sr step 5450, both 1.38 s / step, peak 33.3 GB per rank, no restart. Validation at step 0 is identical to T1 / T0 for both (same initial function).
  Validation at 5000 (auxiliary, never a gate): fair ES indoor / outdoor xa_s2d 9.524 / 8.920, xa_sr 9.565 / 8.898, T1 at the same step 9.581 / 8.952, T0 9.582 / 9.001 -- within 1 %.
- 21:55 user request: report and record every time a result arrives. Cadence: per variant (a) training finished + selected checkpoint + validation curve, (b) closed loops finished + tables;
  each time STATUS.md (P12 result log), this file and reports/FINAL_REPORT.md (living report) are updated and committed.
- 23:56 progress and re-estimate (user request): xa_s2d step 10150, xa_sr step 10550, no restart, peak 33.3-33.4 GB per rank. Measured wall clock between validations: 2 h 00 min per 5000 steps
  including the validation (xa_sr 19:42:48 -> 21:42:48 -> 23:43:14; xa_s2d 19:51:56 -> 21:51:41 -> 23:51:58), so 16.0 h per variant after the step-0 validation.
  Validation at 10000 (auxiliary): fair ES indoor / outdoor xa_s2d 8.145 / 7.750, xa_sr 8.116 / 7.849, T1 8.142 / 7.847, T0 8.106 / 7.857.
  Closed-loop time re-estimated from the archived T1 + D1 episodes (8.8-12.7 min per outdoor episode, dunnottar 9.2 min, 12.8 worker-hours for 70) and the T1 indoor episodes (5.8-7.1 min):
  about 1.2-2 h per variant on 4 GPUs (the earlier 2.7 h was pessimistic; not yet measured for an XA model). Expected ends: round 1 about 11-06 13:15-13:30, round 2 about 11-07 07:00-07:30,
  round 3 about 11-08 01:00-03:00 (M variants may train 3-10 % slower).

## 2026-11-06
- 11:45:22 xa_sr TRAINED (gpu115): 40000 steps in 16 h 10 min 35 s, one attempt, EXIT 0, median 1.372 s / step, peak 33.4 GB per rank. select_final.py: step 40000 (score 1.3884; all scores in
  checkpoints/xa_sr/final.json), final.pt sha256 60069dc4c1419c32231f84f152bc012b72b547d9086cfba86fa667daed3f5ce7. Closed loops started 11:45:28.
- 11:53:22 xa_s2d TRAINED (gpu114): 40000 steps in 16 h 08 min 14 s, one attempt, EXIT 0, median 1.371 s / step, peak 33.3 GB per rank. Selected step 40000 (score 1.3927),
  final.pt sha256 939b7d79f0bb2f898a6be62abc914ae7388a8e82a52b490164ea68fbf7bbf606. Closed loops started 11:53:28.
- Validation at 40000 (auxiliary; fair ES indoor / outdoor): xa_sr 6.367 / 6.431, xa_s2d 6.434 / 6.396, T1 6.425 / 6.482, T0 6.366 / 6.444. Selection score xa_sr 1.3884, xa_s2d 1.3927, T1 1.3946, T0 1.3803.
  All within about 1 % of each other; no confidence interval exists for these numbers, so no difference is claimed.
- 11:54 xa_sr closed loop: the four dunnottar episodes at steps 84-90 after 8.5 min; planner peak memory 31.1 GB (T1 30.5 GB); no error.
- 11:45-14:02 closed loops of xa_sr / xa_s2d, round 1. Big phase (dunnottar x 5 + indoor x 6, one job per GPU): all exit 0 (dunnottar about 10 min, indoor about 7 min each; planner peak 31.1 GB).
  Small phase (65 outdoor episodes, 16 workers = 4 per GPU -- my change of 2026-11-05 19:25 from 3 to 4 per GPU): two engineering failures, both caused by that concurrency:
  (1) 12:12-12:18 the per-user thread limit was hit (RLIMIT_NPROC 4096; one episode = simulator about 70 + planner about 130 threads): planners failed to start ("CUDA device busy or unavailable",
      OpenBLAS "pthread_create failed"), exit 139 (segfault); 16 failed tries on gpu115, 4 on gpu114; one simulator per node kept spinning without progress (xa_sr pisa s2, xa_s2d pisa s0);
  (2) 13:24-13:41 GPU out of memory at planner start for bridge episodes (bridge planner peaks at 12.6 GB; four jobs with large scenes on one 44.5 GB card).
  The other episodes finished (xa_sr 68 / 70 by 13:49, xa_s2d 69 / 70 by 14:02; median 22 min per episode at 4 per GPU). The two hung simulators blocked the chains' `wait`:
  both nodes were idle from 13:49 / 14:02 until 15:01 (about 1 h 05 min on 8 GPUs). My watcher only checked training logs for staleness, not the closed-loop phase, and the previous
  session had ended, so nobody noticed until the user asked at 14:58.
- 15:01 the two hung simulators were terminated (SIGTERM; their planners exited): the queue workers logged the failed try and continued (gpu115: bridge s1, then the chain's round 2 retries pisa s2;
  gpu114: pisa s0 try 2). One orphaned planner (xa_sr pisa s0, parent dead since 12:12, no GPU memory, episode already redone by another worker) was removed at 15:07.
  No result file was deleted or edited; failed tries stay in eval/<id>/logs and eval/<id>/failed.
- 15:07 scripts/xa_eval_queue.sh replaced by v2 (atomic rename; scheduling only, the commands that run an episode are unchanged): at most 3 workers per GPU; a job starts only when the GPU's
  memory reservations allow it (measured per-scene peaks: dunnottar 40, bridge 18, colosseum / sestino 14, pisa 11, alhambra 10, others 9, indoor 24 GB; budget 42 GB); an episode running
  longer than 90 min is terminated and retried; an orphaned planner is removed after a failed try. Tested on gpu115 GPU 0 with 5 workers and 6 three-step jobs (run TESTQ): 3 workers admitted,
  reservations 18 + 14 + 9, 6 / 6 exit 0, locks released. scripts/run_chain.sh is unchanged (it still starts 16 workers; 4 of them now exit at once).
  manifests/code_hashes.json refreshed (changed since the freeze: aggregate.py, xa_eval_queue.sh). The watcher now also reports a closed-loop phase without progress for 50 min.
- 15:10:01 xa_s2d EVALUATED 70 / 70 + 6 / 6 (pisa s0 succeeded on try 2 after the hung process was removed); aggregate.py; 15:10:28 xa_s2d COMPLETE (status.json: replay check 70 / 70,
  planner checkpoint sha == final.pt). 81 episode tries, 5 failed. 15:10:28 xa_si START on gpu114 (scratch).
- 15:10:58 xa_sr eval round 2 (69 / 70) with the v2 queue script: pisa s2 exit 0 in 498 s. 15:19:17 EVALUATED 70 / 70 + 6 / 6; 15:19:21 xa_sr COMPLETE. 93 tries, 17 failed.
  15:19:21 xa_m2d START on gpu115 (scratch).
- Results (All-14, official coverage, scene-equal; difference vs V3-T1 + D1 with scene bootstrap 95 % CI):
  xa_s2d AUC@20 0.418 (-0.008 [-0.016, -0.001]), Cov@20 0.648 (-0.006 [-0.021, +0.009]), AUC@100 0.758 (-0.007 [-0.015, -0.000]), Final 0.897 (-0.013 [-0.031, +0.001]);
  xa_sr  AUC@20 0.423 (-0.003 [-0.012, +0.006]), Cov@20 0.660 (+0.006 [-0.009, +0.020]), AUC@100 0.765 (+0.000 [-0.005, +0.005]), Final 0.906 (-0.004 [-0.015, +0.004]).
  xa_sr - xa_s2d (scene-paired): AUC@20 +0.005 [-0.004, +0.013], Cov@20 +0.012 [-0.007, +0.029], AUC@100 +0.007 [-0.002, +0.017], Final +0.008 [-0.002, +0.024].
  Indoor sanity (6 episodes, C_obs AUC@150 / Final): xa_s2d 0.350 / 0.497, xa_sr 0.338 / 0.501, T1 + H0 0.329 / 0.520.
  One training seed per variant; differences of this size are not interpreted as a ranking.
- Measured closed-loop time per variant in round 1: about 2 h (11:45 -> 13:49 and 11:53 -> 14:02 for everything except the hung episode).
- 20:24 progress check (user request): xa_si step 12750 (1.389 s / step, 5000 steps per 2 h 01 min, peak 33.4 GB), xa_m2d step 12150 (1.402 s / step, 2 h 03 min per 5000, peak 36.6 GB); no restart,
  watcher alive, allocations 3461041 (1 d 22 h left) / 3465477 (2 d 18 h left). Validation at 10000 (auxiliary, fair ES in / out): xa_si 8.095 / 7.730, xa_m2d 8.104 / 7.944, T1 8.142 / 7.847.
  ETA: xa_si trained about 11-07 07:30, xa_m2d about 07:50; closed loops about 2 h each.

## 2026-11-07
- 07:29:47 xa_si TRAINED (gpu114): 40000 steps in 16 h 19 min, one attempt, median 1.385 s / step, peak 33.4 GB. Selected step 40000 (score 1.4134), final.pt sha256 af5ac851fd31ce08a9faebabaaafd186f92561fe7e07a6bb025a3993937ab49e.
  Closed loops 07:29:52 -> 09:40:01 with the v2 queue: 76 tries, 0 failed. 09:40:15 xa_si COMPLETE; xa_mr START (gpu114, scratch).
- 07:52:52 xa_m2d TRAINED (gpu115): 16 h 33 min, one attempt, median 1.400 s / step, peak 36.6 GB. Selected step 40000 (score 1.3960), sha256 e5d0fb545db2f08ee4497c523a737cc031c93c2ef2f1f679bb950483f8b9b8ee.
  Closed loops 07:52:56 -> 10:12:07: 76 tries, 0 failed. 10:12:23 xa_m2d COMPLETE; xa_mi START (gpu115, scratch).
- Results (All-14, official coverage, scene-equal; difference vs V3-T1 + D1 with scene bootstrap 95 % CI):
  xa_si  AUC@20 0.423 (-0.002 [-0.008, +0.004]), Cov@20 0.653 (-0.001 [-0.014, +0.010]), AUC@100 0.764 (-0.001 [-0.007, +0.004]), Final 0.905 (-0.005 [-0.016, +0.004]);
  xa_m2d AUC@20 0.424 (-0.001 [-0.010, +0.006]), Cov@20 0.648 (-0.006 [-0.024, +0.010]), AUC@100 0.759 (-0.006 [-0.015, +0.002]), Final 0.903 (-0.007 [-0.022, +0.004]).
  Single-factor pairs (scene-paired): si - s2d AUC@20 +0.005 [-0.002, +0.013], AUC@100 +0.006 [-0.002, +0.015]; si - sr AUC@20 +0.000, AUC@100 -0.001; m2d - s2d AUC@20 +0.006 [-0.004, +0.017] (11 / 3), AUC@100 +0.001 [-0.009, +0.013].
  Indoor sanity (C_obs AUC@150 / Final): xa_si 0.340 / 0.488, xa_m2d 0.322 / 0.479 (T1 + H0 0.329 / 0.520).
- 15:27 progress check (the session had been suspended; the 07:29 watcher event was only seen now): xa_mr step 14000 (1.404 s / step, peak 36.6 GB), xa_mi step 12500 (1.418 s / step, peak 36.7 GB), no restart.
  Validation at 10000 (auxiliary, fair ES in / out): xa_mr 7.81 / 7.74, xa_mi 7.52 / 7.70 vs T1 8.14 / 7.85 -- the first visible early difference (4-8 % lower indoor); not a gate.
  ETA: xa_mr trained about 11-08 02:15, complete about 04:30; xa_mi trained about 03:00, complete about 05:15.
- 23:00 progress check (user request): xa_mr step 32150 (1.402 s / step), xa_mi step 30450 (1.417 s / step); no restart; 2 h 03-05 min per 5000 steps; watcher alive.
  Validation (auxiliary, fair ES in / out) at 30000: xa_mr 6.04 / 6.36, xa_mi 6.30 / 6.60, T1 6.52 / 6.60, T0 6.51 / 6.57; xa_mr is below T1 at every milestone since 10000
  (15k 6.88 / 7.37, 20k 6.41 / 6.90, 25k 6.19 / 6.60). ETA: xa_mr trained about 11-08 02:15 (complete about 04:30), xa_mi trained about 02:57 (complete about 05:10).

## 2026-11-08
- 02:15:18 xa_mr TRAINED (gpu114): 40000 steps in 16 h 35 min, one attempt, EXIT 0, median 1.403 s / step, peak 36.6 GB. Selected step 40000 (score 1.3184, the lowest of all eight models),
  final.pt sha256 d758d4df0aca69b7... Closed loops started 02:15:24 (03:16: outdoor 19 / 70, indoor 6 / 6, 0 failed tries).
- 02:57:15 xa_mi TRAINED (gpu115): 16 h 45 min, one attempt, EXIT 0, median 1.415 s / step, peak 36.7 GB. Selected step 40000 (score 1.4301), final.pt sha256 71825e2f02f9b6d1...
  Closed loops started 02:57:20 (03:16: outdoor 4 / 70, indoor 3 / 6, 0 failed).
- Validation at 40000 (auxiliary; fair ES in / out, Brier in / out): xa_mr 5.921 / 6.179, 0.0707 / 0.0281; xa_mi 6.244 / 6.415, 0.0785 / 0.0292; T1 6.425 / 6.482, 0.0797 / 0.0291.
  xa_mr is 7.8 % / 4.7 % below T1 and was below at every milestone since 10000; the four S / M2D variants never left the 1.5 % band. No confidence interval; closed loops decide.
