"""Experiment G3 -- Macarons++ planner child (seen2scene env; protocol G3-v1): the frozen Table 3 chain of kd_out_planner.py
(t24_v2_out_planner under p1_full b2, planner_v3, kd_policy d1 = frozen D-v1 config) with the m_combine hooks installed BEFORE the D1 policy.
env G3_MODE: raw | depth -> K = 1 path, the planner's single noise map is map 0 and the hooks draw maps 0..3 (= G4's), combine the 4 decoded
samples into one map and hand it to D1 as the only sample; off -> hooks are identities and K = 4 (= Table 1 V3-T0-D1; regression only)."""
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
import m_combine as MC  # noqa: E402
import d0_head as DH  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    import v2_plan_core as PC
    import v2_common as LC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import scoring as SCO
    mode = os.environ["G3_MODE"]
    assert mode in ("off", "raw", "depth"), mode
    MC.install(PC, PR, G, SCO)
    MC.STATE["mode"] = mode
    K = "4" if mode == "off" else "1"
    KD.install("d1", K)                                   # d_policy captures PC.score_state (= the G3 hook) here
    info = {"g3_mode": mode, "K_planner": int(K), "samples_combined": 4 if mode != "off" else None, "protocol": "G3-v1",
            "m_combine_sha256": DH.sha256_file(MC.__file__), "g3_out_planner_sha256": DH.sha256_file(os.path.abspath(__file__))}
    sys.stderr.write("G3_MODE " + json.dumps(info) + "\n")
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["g3"] = info
    P24.main()
