"""Experiment XA -- Macarons++ planner child (seen2scene env): the frozen D chain (experiment_D_h3_lookahead/scripts/d_out_planner.py:
t24_v2_out_planner under p1_full.patch() b2 + d_policy.install(D_POLICY)) with planner_xa.install() in place of planner_v3.install()."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import d_policy as DP  # noqa: E402
import planner_xa as PXA  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PXA.install()
    DP.install(os.environ["D_POLICY"])
    P24.main()
