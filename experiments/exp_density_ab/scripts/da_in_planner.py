"""EXP Density A/B -- MP3D planner child: task3 t32_in_planner.py (Task 2 child t23_v2_in_planner run unchanged, T3_CKPT
override) with the P1 mode from da_p1.patch (env DA_P1_MODE) instead of p1_full.patch."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t32_in_planner as T32  # noqa: E402  (sets the Task 2 planner import paths)

sys.path.insert(0, HERE)
import da_p1 as DP  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"T3_CKPT={T32.ckpt_override()}\n")
    sys.stderr.write(f"DA_P1_MODE={DP.patch()}\n")
    T32.P23.main()
