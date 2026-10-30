"""EXP6 V3 -- Macarons++ V2-planner child with a V3 model: task2/scripts/t24_v2_out_planner.py run unchanged under
p1_full.patch() (T3_P1_MODE=b2) and planner_v3.install()."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    P24.main()
