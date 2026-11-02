"""EXP6 V5.2 -- MP3D planner child: exp6_v4/scripts/v4_in_planner.py (task2 t23_v2_in_planner, unchanged, B2 P1,
n_win = 5) with v52_core.install(lambda); env V4_CKPT (V5.2 checkpoint), V52_LAMBDA, V52_SCORE_CFG."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import v4_in_planner as V4P  # noqa: E402
import v52_core  # noqa: E402

if __name__ == "__main__":
    cfgp = os.environ.get("V52_SCORE_CFG", os.path.join(os.path.dirname(HERE), "configs", "score_v52.json"))
    v52_core.install(float(os.environ["V52_LAMBDA"]), json.load(open(cfgp)))
    sys.stderr.write(f"V4_CKPT={V4P.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={V4P.PF.patch()}\n")
    V4P.P23.main()
