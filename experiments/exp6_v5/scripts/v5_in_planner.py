"""EXP6 V5 -- MP3D planner child: exp6_v4/scripts/v4_in_planner.py (task2 t23_v2_in_planner unchanged, B2 P1) with
v5_core.install() (V5 model for a V5 checkpoint; env V4_CKPT carries the checkpoint path)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import v4_in_planner as V4P  # noqa: E402
import v5_core  # noqa: E402

if __name__ == "__main__":
    v5_core.install()
    sys.stderr.write(f"V4_CKPT={V4P.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={V4P.PF.patch()}\n")
    V4P.P23.main()
