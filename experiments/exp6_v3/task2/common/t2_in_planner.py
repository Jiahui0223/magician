"""Task 2 -- MP3D planner child with a Task 2 scorer / frontier term: the Task 2 / Task 3 B2 child (t23_v2_in_planner
under p1_full.patch() b2; V3 checkpoints through planner_v3.install() when env V3_CKPT is set) + t2_planner_patch.install().
Everything else (region, refresh, candidates, choice rule, records, guard) is the frozen path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
EXP5 = os.path.join(os.path.dirname(E6), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(E6, "shared"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import t2_planner_patch as T2P  # noqa: E402

if __name__ == "__main__":
    os.environ.setdefault("T3_P1_MODE", "b2")
    os.environ["T2_DATASET"] = "indoor"
    if os.environ.get("V3_CKPT"):
        import planner_v3 as PV3
        sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
        sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
        PV3.install()
    else:
        sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    T2P.install()
    P23.main()
