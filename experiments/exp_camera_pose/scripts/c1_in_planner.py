"""Camera Pose Phase 1 -- indoor planner child for C0 / C1: Task 3 t32_in_planner.py (Task 2 child + T3_CKPT
override + P1 mode) run unchanged, with c1_patch.install() (env CP_LAMBDA, CP_STEPLOG) applied after the P1 patch."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t32_in_planner as P  # noqa: E402

sys.path.insert(0, HERE)
import c1_patch as CP  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("CP_V5") == "1":                     # V5 checkpoint: exp6_v5 v5_core.install() first, as v5_in_planner.py
        sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5", "scripts"))
        import v5_core
        v5_core.install()
        sys.stderr.write("CP_V5: v5_core installed\n")
    sys.stderr.write(f"T3_CKPT={P.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={P.PF.patch()}\n")
    sys.stderr.write(f"C1={CP.install()}\n")
    P.P23.main()
