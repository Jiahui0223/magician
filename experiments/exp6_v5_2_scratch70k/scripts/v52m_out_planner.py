"""EXP6 V5.2-M (Scratch70k) -- Macarons++ planner child: exp6_v4/scripts/v4_out_planner.py (task2 t24_v2_out_planner, unchanged
planner code path, B2 P1 = D100 input + S10 novelty) with v52_core.install(lambda) -- V5.2 checkpoint (env V4_CKPT),
V5.2 camera utility in place of DistP1, flow / DistP1 guards. lambda from env V52_LAMBDA, score config from
env V52_SCORE_CFG (default configs/score_v52.json)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import v4_out_planner as V4P  # noqa: E402
import v52m_core as v52_core  # noqa: E402

if __name__ == "__main__":
    cfgp = os.environ.get("V52_SCORE_CFG", os.path.join(os.path.dirname(HERE), "configs", "score_v52m.json"))
    v52_core.install(float(os.environ["V52_LAMBDA"]), json.load(open(cfgp)))
    sys.stderr.write(f"V4_CKPT={V4P.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={V4P.PF.patch()}\n")
    V4P.P24.main()
