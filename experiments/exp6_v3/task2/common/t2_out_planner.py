"""Task 2 -- Macarons++ planner child with a Task 2 scorer: the Task 2 / Task 3 B2 child (t24_v2_out_planner under
p1_full.patch() b2; V3 checkpoints through planner_v3.install() when env V3_CKPT is set) + t2_planner_patch.install()."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
EXP5 = os.path.join(os.path.dirname(E6), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(E6, "shared"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import t2_planner_patch as T2P  # noqa: E402

if __name__ == "__main__":
    os.environ.setdefault("T3_P1_MODE", "b2")
    os.environ["T2_DATASET"] = "outdoor"
    if os.environ.get("V3_CKPT"):
        import planner_v3 as PV3
        sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
        sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
        PV3.install()
    else:
        sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    T2P.install()
    P24.main()
