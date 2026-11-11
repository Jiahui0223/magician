"""EXP7 / UAVScenes -- planner child (seen2scene env): the frozen outdoor D1 child (experiment_D d_out_planner.py chain:
t24_v2_out_planner under p1_full.patch() b2, planner_v3.install(), d_policy.install(D_POLICY)) with uav_hooks.install_frame_hook()
(E-M2 rule for uav_* tiles). Started by uav_sim_out.py with MACARONS_PLUSPLUS_ROOT = the UAV data root."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments")
DS = os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts")
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_v2_out_planner as P24  # noqa: E402

sys.path.insert(0, os.path.join(X, "exp5_future_view", "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, DS)
sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import d_policy as DP  # noqa: E402
import uav_hooks as UH  # noqa: E402

if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    UH.install_frame_hook()
    UH.install_z0_hook()
    PV3.install()
    DP.install(os.environ["D_POLICY"])
    P24.main()
