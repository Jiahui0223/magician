# Multi-view coverage study — events (2026-11-05; facts only)

| step | event |
|---|---|
| config | `configs/mv_v1.yaml` frozen before any output (truth definitions, pair sets, consistency rule, K study, CVAR rules and gate) |
| smoke | colosseum s1 t <= 20: first run failed an internal assert -- lookahead views were mapped back to pose keys by T alone, but 20,880 of 49,000 poses share T (same position, different rotation); fixed by keying on (R, T); afterwards recorded-set scores == planner scores exactly |
| run | 395 states (E3 120 + G2 275) on 8 GPUs, 01:03-01:34, 0 failures; peak 8.8 GB per process |
| finding during run | the value function on TRUE depth ranked worse than on predicted depth -> value-function diagnostic added (`mv_value_diag.py`, two passes: v1 weighted / unweighted / D1; v2 + area-uniform target coverage + dense-observation novelty); v1 rows kept in work_dirs/.../value_diag_v1 |
| cases | chosen by fixed rules (`mv_cases.py`) before any case figure; both case states re-run with saving, bitwise equal to the main run; candidate views rendered with the simulator renderer (`mv_render.py`) |
| gate | CVAR offline gate FAIL (official regret +0.007, Top-1 -0.010) -> stop, no closed loop |
