"""EXP6 V5 -- Macarons++ planner child: exp6_v4/scripts/v4_out_planner.py (task2 t24_v2_out_planner unchanged, B2 P1:
D100 input + S10 sparse DistP1) with v5_core.install() so that v2_plan_core.Core builds the V5 model for a V5
checkpoint (env V4_CKPT carries the checkpoint path, as for V4)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import v4_out_planner as V4P  # noqa: E402
import v5_core  # noqa: E402

if __name__ == "__main__":
    v5_core.install()
    sys.stderr.write(f"V4_CKPT={V4P.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={V4P.PF.patch()}\n")
    V4P.P24.main()
