# Experiment F -- event log (facts, deviations, decisions; times MST)

## 2026-11-05
- 03:00 EXPERIMENT_PLAN.md = verbatim copy of experiments/exp6_v3/V3_Experiment_F_Coverage_Decision_Study.md (F-v1.0, sha256 6de83e9d...).
  Slurm: own jobs 3461041 (gpu114) and 3465477 (gpu115), 4 x L40S each, all 8 GPUs idle; no other experiment of ours running.
- 03:05 configs frozen before any F computation: configs/F_config.yaml (parameters, selection rules, gates, F5 pre-registration) and
  configs/confirmation_set.json (confirmation layer). Construction rule of the confirmation layer: D-v2's own closed-loop states
  (Table 1 V3-DV2-KSV, rep 0) on the six runnable scenes that are not among the exploration scenes (colosseum, pisa, barts, pantheon):
  eiffel, sestino_museum, bannerman, fushimi, liberty, redeemer; starts 0, 2, 4; t = 0, 5, ..., 95 (18 episodes, 360 states).
  dunnottar excluded (planner replay runs out of memory, ROI audit), Test-3 excluded. Steps of one episode are not independent:
  the pre-registered CI unit is the scene (6 clusters), episode-level CIs are reported as secondary.
- 03:06 confirmation-layer official truth: experiment E's frozen e_oracle.py (one-step GT counterfactual, no two-step) on the 18 episodes,
  4 GPUs gpu115, outputs only into F's work dir; 18/18 exit 0 (about 3 min per episode).
- 03:10 configs/F_config.yaml wording fix before any result (no parameter change): the GT-depth audit "gated" variant is
  encode_labels -> decode exactly (rays without an UNKNOWN hit target take the projection outcome t_known / no hit), not "no hit".
- 03:12 smoke (colosseum s1 t = 0, 5, 10 exploration; eiffel s0 t = 0, 5 confirmation): recorded sets -> planner scores exact (0.0),
  gamma = 1 bitwise, sigma 0 one-step == hard (<= 2e-13), sigma-0 supports == recorded sets (ids and weights), replayed scores ==
  multiview-study scores bitwise and == the closed-loop D-v2 planner log bitwise, P1 / S10 / D-v2 grid checksums unchanged.
  40-60 s per state, 2.8-8.8 GB. Added before the full run: offline diffusion diagnostic (share of first-layer soft mass on cells near
  the GT target surface and on the hard cells).
- 03:20 full cache launched: 8 workers (gpu114 g0-3, gpu115 g0-3), trajectory claims, exploration first, then confirmation.
  The confirmation states are cached in the same pass but are not analysed before results/selection.json is frozen from the
  exploration layer (f_analyze.py refuses otherwise).
- 03:59 exploration cache complete: 36 trajectories, 395 states, no error. F0 on the exploration layer PASS (recorded -> planner 0.0 for K4 and
  K8, gamma = 1 bitwise, sigma 0 == hard <= 2.3e-13, 395 / 395 replayed scores bitwise == multiview study, P1 unchanged, official truth complete).
  K8 first 4 vs the planner's K4 run: median relative score difference 0.3 %, 1 / 395 pick changes.
- 04:05 f_analyze.py --stage explore: results/selection.json written once by the frozen rules: F1 gamma* = 0.25, F2 variant* = s0.25_w
  (sigma = 0.25 tau_cov, depth weight, max-union; all 8 soft variants were eligible under the geo2 constraint), F3 = adaptive 4 -> 8.
  The confirmation layer had not been analysed at this point (its cache was still being computed).
- 04:19 confirmation cache complete (18 / 18, exit 0). f_analyze.py --stage confirm (once, after selection.json): F0 PASS (360 / 360 replayed
  scores bitwise == the closed-loop D-v2 planner log). Gates: F1 gamma 0.25 TREND (-0.007 pp, scene CI [-0.022, +0.006]), F2 s0.25_w TREND
  (-0.007, [-0.026, +0.013]), F3 adaptive FAIL (+0.017; vs equal-budget random +0.007). No PASS -> F5 not entered (no closed loop).
- 04:25 supplementary (pre-declared F0.1 / F2 items, no effect on gates): f0_clamp_check.py (4 exploration states, 27 views: OUT / NOHIT
  true-depth value is 100 % from hits outside the P1 grid, counted through value_cov's index clamp), f2_bench.py (s0.25_w: +1.0 s per decision
  over the planner's 3.2 s, 1.7 GB peak; eiffel s0, 5 states). f_analyze.py re-run with an extra audit comparison against the predictions;
  all gate numbers identical, selection.json unchanged (checked).
- Post-hoc observations recorded in the report as non-admissible: gamma 0.5 confirm CI excludes 0; clip-sum union significant on both
  layers; one-step soft scoring significant on confirm only; F1+F3 diagnostic combination significant on confirm only.
