# E2 events (chronological; facts only)

| time (2026-11-04) | event |
|---|---|
| 18:1x | configs/audit_config.yaml (E2-audit-v1) frozen; smoke colosseum s1 t<=30 (logs/smoke/) |
| 18:22 | audit colosseum s2 crashed at t=0: the first frame has no valid depth pixel, empty dense-observation set -> cKDTree error. Guard added (empty set -> no dense observation); rerun from scratch, exit 0 18:25. No other audit affected |
| 18:25 | audit 7/7 done (140 states); replay of D1 value from the reconstructed state == planner log (max rel err 0.0) |
| 18:2x | configs/ksv_v1.yaml (E2-KSV-v1) frozen before the fixed-state A/B; smoke colosseum s1 t<=30 |
| 18:35 | fixed-state A/B 7/7 done; OFF == planner log on 140/140 states; pre-registered gate PASS |
| 18:37 | closed-loop regression: KSV off, colosseum s0 rep 0 reproduces Table 1 D1 bitwise (coverage, pose keys, all scores, all picks, 100 steps) |
| 18:36 | closed-loop rule appended to configs/ksv_v1.yaml before any KSV-on episode finished; 35 KSV-on episodes queued (colosseum 5 x 3 reps, Dev-4 x 5 starts rep 0), 2 per GPU on 8 GPUs |
| note | alhambra s0 (Test-3) and pisa s1 (Train-6) were used only as fixed-state diagnostics with parameters frozen beforehand; no parameter was changed after any result |
| 19:10 | 35/35 KSV-on episodes exit 0 (no retries); queued colosseum s1 r0 == the earlier KSV-on smoke bitwise (determinism) |
| 19:12 | closed-loop gate FAIL on the pre-registered early-AUC condition (Dev-5 dAUC@20 -0.008 < -0.005); per protocol D1 kept, nothing registered in Table 1, All-14 not run |
