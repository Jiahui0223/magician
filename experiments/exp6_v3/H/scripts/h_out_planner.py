"""EXP6 V3 / H -- Macarons++ planner child: exp6_v3/shared/v3_out_planner.py (task2 t24_v2_out_planner under p1_full.patch()
T3_P1_MODE=b2 and planner_v3.install()) plus h_policy.install(H_POLICY). Same arguments as v3_out_planner.py."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import h_policy as HP  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    HP.install(os.environ["H_POLICY"])
    P24.main()
