"""Experiment G2 item 5 -- MP3D planner child (seen2scene env; protocol G2-v1 section 5): EXACTLY the EXP7 OURS child
(experiments/cvpr_results/exp7/scripts/d_in_planner.py, imported read-only: t23_v2_in_planner under p1_full b2 + planner_v3 + frozen D-v1
d_policy d1 with the EXP7 MP3D config, once-per-record capture) plus two switches set by sim_in_g.py:
  G_K     1 -> the t06 Planner's K (4 for V2K4) is set to 1 after construction: one noise map, key exp5-V2|task0cl|<tag>|t<t_r>|k0
          (= map 0 of the K = 4 set); 4 -> unchanged (regression only: must reproduce EXP7 OURS)
  G_MODE  flow -> V3Core unchanged; d0 -> PC.Core loads the D0 checkpoint with d0_head.DetPredictor (one forward per candidate)
Nothing else changes (region n_win 5, R10, P1, candidates, D1 value / lookahead, prefer-unvisited NV choice)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
sys.path.insert(0, os.path.join(X, "cvpr_results", "exp7", "scripts"))
import d_in_planner as DIP  # noqa: E402  (frozen EXP7 child; its __main__ is not executed)

sys.path.insert(0, HERE)
import d0_head as DH  # noqa: E402

P23, PF, PV3, DP = DIP.P23, DIP.PF, DIP.PV3, DIP.DP


def install_g(mode, K):
    import v2_plan_core as PC
    assert mode in ("flow", "d0") and K in (1, 4), (mode, K)
    if mode == "d0":
        class GCore:
            def __init__(self, ckpt_path, device, expect_sha=None, steps=10):
                self.model, self.pred, self.ckpt_sha = DH.load(ckpt_path, device, "d0", steps, expect_sha)
                self.dev, self.ckpt = device, ckpt_path
        PC.Core = GCore
    if K != 4:
        base = P23.TP.Planner

        class PlannerK(base):
            def __init__(self, method, card_key, tag, dev):
                super().__init__(method, card_key, tag, dev)
                assert method == "V2K4" and self.K == 4, (method, self.K)
                self.K = K
        P23.TP.Planner = PlannerK
    info = {"g_mode": mode, "K": K, "protocol": "G2-v1", "d0_head_sha256": DH.sha256_file(DH.__file__),
            "g_in_planner_sha256": DH.sha256_file(os.path.abspath(__file__))}
    sys.stderr.write("G_MODE " + json.dumps(info) + "\n")
    return info


if __name__ == "__main__":                       # = d_in_planner.__main__ + install_g before P23.main()
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    card = sys.argv[sys.argv.index("--card") + 1]
    sys.argv += ["--scene", card]
    DP.install(os.environ["D_POLICY"])
    del sys.argv[-2:]
    DIP.install_once_capture()
    install_g(os.environ["G_MODE"], int(os.environ["G_K"]))
    P23.main()
