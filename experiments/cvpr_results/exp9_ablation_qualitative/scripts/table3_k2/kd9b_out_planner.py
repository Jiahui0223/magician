"""T3K2 (2026-11-16) -- copy of the frozen EXP9 table3_k/kd9_out_planner.py importing kd9b_policy (only change: K = 2 with noise offset 2 allowed).
EXP9 Table 3 (K in {1, 2, 4, 8}) -- copy of the frozen table3/kd_out_planner.py importing kd9_policy (only the allowed K set differs).
CVPR Table 3 -- Macarons++ planner child: the frozen d_out_planner.py chain (t24_v2_out_planner under p1_full.patch() b2 and
planner_v3.install()) with kd_policy.install(D_POLICY, KD_K) instead of d_policy.install(D_POLICY)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))    # experiments/
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import kd9b_policy as KD  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    KD.install(os.environ["D_POLICY"], os.environ["KD_K"])
    P24.main()
