"""EXP Density A/B -- Macarons++ planner child: task3 t32_out_planner.py (Task 2 child t24 -> task1 v2_planner unchanged,
T3_CKPT override) with the P1 mode from da_p1.patch (env DA_P1_MODE)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t32_out_planner as T32  # noqa: E402

sys.path.insert(0, HERE)
import da_p1 as DP  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"T3_CKPT={T32.ckpt_override()}\n")
    sys.stderr.write(f"DA_P1_MODE={DP.patch()}\n")
    T32.P24.main()
