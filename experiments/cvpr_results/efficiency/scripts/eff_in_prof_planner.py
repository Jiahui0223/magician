"""EXP8-MP3D -- indoor RAYSCOUT planner child with synchronised component timers (seen2scene env). The frozen EXP7 indoor chain of
experiments/cvpr_results/exp7/scripts/d_in_planner.py (task2 t23_v2_in_planner: T0.6 planner V2K4 path, K = 4, R10 region refresh,
n_win = 5, prefer-unvisited choice; p1_full.patch() b2; planner_v3.install(); d_policy.install('d1') with the MP3D pose tables;
capture-once) with timers wrapped around module functions (no frozen file edited; picks compared with the records by the driver):
  map_update     m_p1.P1Fusion.integrate_record  (all calls of the step: the new record, and at a refresh (t % 10 == 0) the replay
                 of records 0..t into the new region frame -- the indoor planner rebuilds its map there)
  geometry       v2_plan_core.Geometry.from_fusion
  vae_encode / view_cond / ray_encoder / depth_decode / gen_l1 / gen_l2 / value_cov / neighbours / segment / union  as in
                 eff_prof_planner.py (outdoor), via eff_timers.py
The accumulators are added to the step's timing dict under "eff" (returned to the driver in the answer)."""
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "efficiency", "scripts"))
import d_in_planner as DIP  # noqa: E402  (imports the frozen chain; its __main__ block is not executed)
import eff_timers as EP  # noqa: E402  (timed(), ACC / CNT accumulators; no side effects)

PF, PV3, DP, P23 = DIP.PF, DIP.PV3, DIP.DP, DIP.P23

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    import m_p1
    PC = EP.install_hooks(DP)                    # component timers + inner score_state wrapper (layer 1 / layer 2)
    m_p1.P1Fusion.integrate_record = EP.timed("map_update", m_p1.P1Fusion.integrate_record)
    PC.Geometry.from_fusion = EP.timed("geometry", PC.Geometry.from_fusion)
    card = sys.argv[sys.argv.index("--card") + 1]
    sys.argv += ["--scene", card]
    DP.install(os.environ["D_POLICY"])           # wraps integrate_record (capture) and score_state (D1)
    del sys.argv[-2:]
    DIP.install_once_capture()
    policy_score = PC.score_state

    def outer(*a, **k):
        t0 = EP._sync()
        out, tm, views = policy_score(*a, **k)
        total = EP._sync() - t0
        tm = dict(tm, eff=dict({f"{n}_s": v for n, v in EP.ACC.items()}, **{f"n_{n}": c for n, c in EP.CNT.items()}, outer_score_state_s=total))
        EP.ACC.clear(); EP.CNT.clear(); EP.STEP["gen_calls"] = 0
        return out, tm, views
    PC.score_state = outer
    P23.main()
