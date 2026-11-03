"""EXP Temporal Refresh -- MP3D planner child: task3 t32_in_planner (Task 2 t23 child, T3_CKPT override) with P1 mode b2
(D100 inference, S10 DistP1 scoring; exp_density_ab da_p1 'd100s10' = p1_full.patch('b2')) and the encoder-only refresh
period from env TR_REFRESH (tr_refresh.patch_indoor)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t32_in_planner as T32  # noqa: E402  (sets the Task 2 planner import paths)

sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
sys.path.insert(0, HERE)
import da_p1 as DP  # noqa: E402
import tr_refresh as TR  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"T3_CKPT={T32.ckpt_override()}\n")
    sys.stderr.write(f"DA_P1_MODE={DP.patch('d100s10')}\n")
    sys.stderr.write(f"TR_REFRESH={TR.patch_indoor(TR.period())}\n")
    T32.P23.main()
