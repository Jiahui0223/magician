"""Experiment G -- Macarons++ planner child (seen2scene env; protocol G-v1): the frozen Table 3 chain of kd_out_planner.py
(t24_v2_out_planner under p1_full.patch() b2, planner_v3.install(), kd_policy.install("d1", K = 1) = the frozen D-v1 d1 config) plus
ONE switch, env G_MODE:
  flow  nothing else -- byte-for-byte the planner of the archived Table 3 row V3-T0-K1D2 (G1); used for the regression check
  g0    PC.Core loads the flow checkpoint with d0_head.ZeroNoisePredictor (noise maps replaced by zeros; Euler 10)       (G0)
  d0    PC.Core loads a D0 checkpoint (model_cfg head 'det') with d0_head.DetPredictor (one forward per candidate)     (D0)
Everything else (candidates, P1, R10, noise keys, D1 two-step value, m_score.choose, records) is the frozen path."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, os.path.join(X, "cvpr_results", "scripts", "table3"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import kd_policy as KD  # noqa: E402
import d0_head as DH  # noqa: E402


def install_mode(mode):
    import v2_common as LC
    import v2_plan_core as PC
    assert mode in ("flow", "g0", "d0"), mode
    if mode != "flow":
        class GCore:
            def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
                self.model, self.pred, self.ckpt_sha = DH.load(ckpt_path, device, mode, steps, expect_sha)
                self.dev, self.ckpt = device, ckpt_path
        PC.Core = GCore
    info = {"g_mode": mode, "protocol": "G-v1", "d0_head_sha256": DH.sha256_file(DH.__file__), "g_out_planner_sha256": DH.sha256_file(os.path.abspath(__file__))}
    sys.stderr.write("G_MODE " + json.dumps(info) + "\n")
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["g"] = info
    return info


if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    KD.install(os.environ["D_POLICY"], os.environ["KD_K"])
    install_mode(os.environ["G_MODE"])
    P24.main()
