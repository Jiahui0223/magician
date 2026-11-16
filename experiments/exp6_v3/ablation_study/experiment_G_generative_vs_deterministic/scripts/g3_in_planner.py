"""Experiment G3 -- MP3D planner child (seen2scene env; protocol G3-v1): EXACTLY the EXP7 OURS child (cvpr_results/exp7/scripts/d_in_planner.py,
imported read-only) with the m_combine hooks installed BEFORE d_policy.install. env G3_MODE: raw | depth -> the t06 Planner's K is set to
1 after construction (noise map 0 = key exp5-V2|task0cl|<tag>|t<t_r>|k0); the hooks draw maps 0..3 (= EXP7 OURS maps), combine the 4 decoded
samples into one map and give it to D1 as its only sample; off -> hooks identities, K = 4 (regression only: must reproduce EXP7 OURS)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
sys.path.insert(0, os.path.join(X, "cvpr_results", "exp7", "scripts"))
import d_in_planner as DIP  # noqa: E402  (frozen EXP7 child; __main__ not executed)

sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402
import m_combine as MC  # noqa: E402

P23, PF, PV3, DP = DIP.P23, DIP.PF, DIP.PV3, DIP.DP

if __name__ == "__main__":                       # = d_in_planner.__main__ + the G3 hooks (before DP.install) + K = 1
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    import v2_plan_core as PC
    from seen2scene.futureview import geometry as G
    from seen2scene.futureview import predictor as PR
    from seen2scene.futureview import scoring as SCO
    mode = os.environ["G3_MODE"]
    assert mode in ("off", "raw", "depth"), mode
    MC.install(PC, PR, G, SCO)
    MC.STATE["mode"] = mode
    card = sys.argv[sys.argv.index("--card") + 1]
    sys.argv += ["--scene", card]
    DP.install(os.environ["D_POLICY"])               # captures PC.score_state (= the G3 hook)
    del sys.argv[-2:]
    DIP.install_once_capture()
    K = 4 if mode == "off" else 1
    if K != 4:
        base = P23.TP.Planner

        class PlannerK(base):
            def __init__(self, method, card_key, tag, dev):
                super().__init__(method, card_key, tag, dev)
                assert method == "V2K4" and self.K == 4, (method, self.K)
                self.K = K
        P23.TP.Planner = PlannerK
    info = {"g3_mode": mode, "K": K, "protocol": "G3-v1", "m_combine_sha256": DH.sha256_file(MC.__file__),
            "g3_in_planner_sha256": DH.sha256_file(os.path.abspath(__file__))}
    sys.stderr.write("G3_MODE " + json.dumps(info) + "\n")
    P23.main()
