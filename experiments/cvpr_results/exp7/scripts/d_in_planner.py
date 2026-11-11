"""EXP7 -- MP3D planner child (seen2scene env): the frozen V3 indoor chain of exp6_v3/shared/v3_in_planner.py (task2
t23_v2_in_planner.py: T0.6 planner V2K4 path, K = 4, R10, n_win = 5 planning region, prefer-unvisited choice, under p1_full.patch()
b2 = D100 input map + sparse S10 map, planner_v3.install()) with the frozen D-v1 policy (d_policy.install, policy D_POLICY = d1:
two-step lookahead, value_cov eps 1.0 / cell 1.0, lambda 0) and the MP3D grid pose table of the level
(configs/d_config_mp3d.yaml = D-v1 with only `pose_tables` changed; tables verified by scripts/check_mp3d_pose_tables.py).
One indoor-specific detail, no change of the decision: the indoor planner rebuilds its P1 in the region frame at every refresh and
replays records 0..t, so h_policy.capture would append the same S10 points again; here each record's points are captured once
(the nearest-observed-point test of value_cov is unchanged by duplicates, this only avoids the growth)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                       # experiments/
EXP5 = os.path.join(X, "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_v2_in_planner as P23  # noqa: E402  (sets the planner import paths)

sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "scripts"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "shared"))
sys.path.insert(0, os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "scripts"))
import p1_full as PF  # noqa: E402
import planner_v3 as PV3  # noqa: E402
import d_policy as DP  # noqa: E402

HP = DP.HP


def install_once_capture():
    seen, cap = set(), HP.capture

    def capture(rec):
        t = int(np.asarray(rec["t"]))
        if t in seen:                                   # replay of an already captured record: keep the latest state only
            HP.STATE["rec"] = rec
            HP.STATE["visited"] = {tuple(int(x) for x in r) for r in np.asarray(rec["executed_pose_idx_history"]).reshape(-1, 5)}
            return
        seen.add(t)
        cap(rec)
    HP.capture = capture


if __name__ == "__main__":
    sys.stderr.write(f"V3_CKPT={PV3.ckpt_override()}\n")
    sys.stderr.write(f"T3_P1_MODE={PF.patch()}\n")
    PV3.install()
    card = sys.argv[sys.argv.index("--card") + 1]
    sys.argv += ["--scene", card]                       # d_policy.install reads the pose table <pose_tables>/<scene>.npz
    DP.install(os.environ["D_POLICY"])
    del sys.argv[-2:]
    install_once_capture()
    P23.main()
