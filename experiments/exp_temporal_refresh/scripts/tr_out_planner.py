"""EXP Temporal Refresh -- Macarons++ planner child: task3 t32_out_planner (t24 -> task1 v2_planner, T3_CKPT override) with
P1 mode b2 (da_p1 'd100s10') and the encoder-only refresh period from env TR_REFRESH (tr_refresh.patch_outdoor)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t32_out_planner as T32  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
sys.path.insert(0, HERE)
import da_p1 as DP  # noqa: E402
import tr_refresh as TR  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"T3_CKPT={T32.ckpt_override()}\n")
    sys.stderr.write(f"DA_P1_MODE={DP.patch('d100s10')}\n")
    sys.stderr.write(f"TR_REFRESH={TR.patch_outdoor(TR.period())}\n")
    T32.P24.main()
