"""EXP5 Task 2 -- outdoor (Macarons++) V2 planner child, groups V2O (G0) and V2M (G1) (seen2scene env; planner side).

Runs the Task 1.4 V2 planner child (task1/scripts/v2_planner.py, unchanged code path: P1, R10 refresh, K = 4, DistP1,
m_score.choose) with exactly these Task 2 settings, patched in this process only:
  * groups V2O / V2M added to the arm table (score distp1, K 4, R10) -- same definition as Task 1 arm E1
  * checkpoint = the group's checkpoint (--ckpt, sha256 checked by the core)
  * z0 of every evaluation scene = MAGICIAN scene_scale / 2 from settings.json (abl_common.z0_of; equals the frozen Dev-3
    values)
  * LeakGuard additionally forbids every Task 2 folder except this episode's service records and planner logs
Greedy-NV: the outdoor (M-protocol) legal list already applies MAGICIAN's get_valid_neighbors (unvisited poses first)
before the GT collision test, so the Task 1 choice rule is the prefer-unvisited greedy rule (protocol §4.2).
Usage: same arguments as v2_planner.py + --ckpt PATH --sha256 HEX.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402
import v2_frozen as VF  # noqa: E402

ARM = {"name": "task2_v2", "score": "distp1", "K": 4, "refresh": "R10"}
for g in ("V2O", "V2M"):
    LC.ARMS[g] = dict(ARM, name=f"task2_{g}")
    if g not in LC.RUN_ORDER:
        LC.RUN_ORDER.append(g)


def _pop(flag):
    i = sys.argv.index(flag)
    v = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return v


def main():
    ckpt, sha = _pop("--ckpt"), _pop("--sha256")
    fv0 = VF.frozen_values()
    VF.frozen_values = lambda: dict(fv0, checkpoint={"path": os.path.relpath(ckpt, VF.REPO), "sha256": sha})
    import v2_planner as VP
    import abl_common as AC
    scene = sys.argv[sys.argv.index("--scene") + 1]
    LC.Z0[scene] = AC.z0_of(scene)
    root = os.path.abspath(sys.argv[sys.argv.index("--root") + 1])
    base = VP.V2Guard

    class Guard(base):
        def forbidden(self, p):
            if base.forbidden(self, p):
                return True
            if not isinstance(p, str):
                return False
            a = os.path.abspath(p)
            if any(a.startswith(x) for x in self.allow):
                return False
            return a.startswith(C.W2 + "/") and not a.startswith(root + "/planner_logs/")
    VP.V2Guard = Guard
    VP.main()


if __name__ == "__main__":
    main()
