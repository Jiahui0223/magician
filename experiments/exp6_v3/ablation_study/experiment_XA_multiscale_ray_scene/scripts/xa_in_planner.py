"""Experiment XA -- MP3D planner child (seen2scene env): exp6_v3/shared/v3_in_planner.py (task2 t23_v2_in_planner.py: frozen indoor path, H0
DistP1 + Greedy-NV, n_win 5, under p1_full.patch() b2) with planner_xa.install() in place of planner_v3.install()."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402  (sets the planner import paths)

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import planner_xa as PXA  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PXA.install()
    P23.main()
