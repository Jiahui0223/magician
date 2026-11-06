# Experiment CT -- event log (facts, deviations, decisions; times MST)

## 2026-11-05
- 18:10 resources: own Slurm jobs 3461041 (gpu114, ends 11-08 18:30) and 3465477 (gpu115, ends 11-09 15:12), 4 x L40S each, all idle; the other L40 nodes are allocated to other users; /bsuscratch 1.3 PB free.
- 18:15 CT0 audit read-only (results/CT0_checkpoint_audit/CT0.json). Protocol frozen: configs/CT_config.yaml (CT-v1), configs/train_ct.json, before any CT evaluation or training.
- Deviation from the plan §7 ("independent Slurm jobs"): training and evaluation run inside the two existing allocations (Job IDs above). All L40 nodes are allocated; new sbatch jobs would wait behind our own allocations. No effect on results.
- 18:20 smoke: indoor one floor, outdoor barts t = 15: 40K regression bitwise (eval_fixed T1 rows; F rows scores, V12, children). About 2 s per indoor state, 10 s per outdoor state for 2 checkpoints.
- 18:23 fixed states group g1 (A_40K, B_Y50K) launched: 8 workers, indoor -> explore -> confirm.
- 18:55 g1 finished (40K + Y50K, 845 states, 0 failures): 40K regression bitwise on all 845 states (755 outdoor vs the Experiment F rows: scores, V12, children; 90 indoor vs eval_fixed). Y50K: depth errors improve with CI excluding 0 in all layers, regret does not improve (indoor +0.036, outdoor Dev -0.016, CIs include 0; picks changed 3 / 90, 39 / 495, 9 / 260).
- 18:57 CT-M training launched per the run gate (CT0 found no fault): gpu114 GPUs 0-3, allocation 3461041, runs/ct_m, port 29631. Counterfactuals for 40K (group cf_A_40K) and the closed-loop regression on gpu115 (allocation 3465477).
- 19:05 the user replaced this experiment with Experiment XA (experiments/exp6_v3/XA_Experiment_Protocol.md; §1: archive CT safely, then stop). From here on CT only finishes what was running and is paused.
- 19:07 closed-loop regression of the 40K (R_40K barts s0 with D-v2; Z6MFQCViBuw_L0 S6 with H0): executed poses and coverage equal the archived episodes.
- 19:08 Y50K small closed loop started on gpu115 next to the counterfactual workers (2 jobs per GPU): 5 indoor planner failures (CUDA OOM) between 19:08 and 19:15. Partial folders moved to closed_loop/failed/.
- 19:20 Y50K closed loop stopped by me (it also competed with the XA engineering smokes on the same GPUs): 0 episodes complete. The 6 indoor episodes that were running were moved to closed_loop/failed/
  (*_stopped_1920) together with all B_Y50K claims; the outdoor step checkpoints of barts s0-s2 stay in closed_loop/outdoor/B_Y50K (the simulator can resume them). Nothing was deleted.
- 19:22 CT-M periodic checkpoint at step 41000 copied to archive_paused/snap_041000 (read-only), sha256 3b657431..., strict load of EMA and raw weights verified.
- 19:34 counterfactual workers stopped (wrappers first, then workers): 26 of 63 trajectories complete (indoor 9 / 9, outdoor 17 / 54: e3_colosseum 5, g2_B0_barts 5, g2_B0_colosseum 5, g2_B0_pantheon 1, e3_pisa 1). Each trajectory is written atomically; resumable.
- 19:43 CT-M wrote the periodic checkpoint at step 42000; 19:44 copied to archive_paused/snap_042000 with log, run manifest, configs, scripts and SHA256SUMS (read-only); sha256 5b82500b...; keys, optimizer state (394),
  strict load of EMA and raw weights, finite values verified. No save signal was sent (the trainer has no signal handler).
- 19:44:54 CT-M stopped on gpu114 at about step 42070: wrapper (retry loop) first, then torchrun, then the 4 ranks. runs/ct_m/ckpt_last.pt is the step-42000 state (same sha256 as the archive). The Slurm allocation was not cancelled.
  CT_PAUSED_MANIFEST.md written (resume command, limits). No 45K / 50K EMA exists for CT-M.
- 19:47 ct_analyze.py --groups g1 --cf cf_A_40K (CPU): fixed-state tables unchanged; counterfactual ladder added (indoor complete, outdoor partial). FINAL_REPORT.md written as a PARTIAL / PAUSED report.
