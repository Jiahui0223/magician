"""E2-KSV-v1 planner child: the frozen D-v1 chain (d_out_planner.py: t24_v2_out_planner under p1_full.patch() b2,
planner_v3.install(), d_policy.install(D_POLICY)) + ksv.install(scene). env E2_KSV=0 installs nothing extra (= frozen D1)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import d_policy as DP  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    DP.install(os.environ["D_POLICY"])
    if os.environ.get("E2_KSV", "1") == "1":
        import torch
        import ksv as KSV
        KSV.install(sys.argv[sys.argv.index("--scene") + 1], torch.device("cuda"), None)
        sys.stderr.write("E2_KSV installed " + str({"cell": KSV.CELL, "step": KSV.STEP, "margin": KSV.MARGIN, "min_t": KSV.MIN_T}) + "\n")
    P24.main()
