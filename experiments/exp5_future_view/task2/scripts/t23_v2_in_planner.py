"""EXP5 Task 2 -- indoor (MP3D) V2 planner child, groups G0 (V2 Original) and G1 (V2-Mix) (seen2scene env; planner side).

= the T0.10 V2K4-NV adapter (task0/magician_comparison/adapters/v2nv_planner.py: frozen T0.6 planner scripts/t06_planner
V2K4, K = 4, DistP1, R10, unchanged scores + MAGICIAN's prefer-unvisited choice) with exactly two Task 2 settings
(protocol §4):
  * planning region n_win = 5 (t2_common.region_box_m patched into mp3d_frame.region_box_m in this process only)
  * checkpoint = the group's checkpoint (--ckpt path, sha256 checked), loaded through the frozen-values hook of the
    T0.6 planner in this process only
LeakGuard: the T0.6 guard + every Task 2 training / selection data folder (W2/mp3d) + Task 2 evaluation outputs other
than this episode's service records.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task0", "magician_comparison", "adapters"))
import v2nv_planner as NV  # noqa: E402

TP = NV.TP
import v2_frozen as VF  # noqa: E402  (task1/scripts, on the path through t06_planner; imported lazily by the planner)
TP.F.region_box_m = lambda card, center_rt: C.region_box_m(card, center_rt, C.N_WIN)

import numpy as np  # noqa: E402,F401
import torch  # noqa: E402


class Guard(TP.Guard):
    allow = ()

    @classmethod
    def forbidden(cls, p):
        if TP.Guard.forbidden(p):
            return True
        a = os.path.abspath(p) if isinstance(p, str) else ""
        if any(a.startswith(x) for x in cls.allow):
            return False
        return a.startswith(os.path.join(C.W2, "mp3d") + "/") or a.startswith(os.path.join(C.W2, "eval") + "/")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["V2O", "V2M"], required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--svc-dir", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    fv0 = VF.frozen_values()
    VF.frozen_values = lambda: dict(fv0, checkpoint={"path": os.path.relpath(a.ckpt, VF.REPO), "sha256": a.sha256})
    dev = torch.device("cuda")
    pl = TP.Planner("V2K4", a.card, a.tag, dev)
    Guard.allow = (os.path.abspath(a.svc_dir) + "/",)
    g = Guard().install()
    proto.write(json.dumps({"ready": True, "info": {"group": a.group, "base": "t06_planner V2K4 + prefer-unvisited (v2nv_planner)",
                                                    "n_win": C.N_WIN, "ckpt": os.path.relpath(a.ckpt, VF.REPO),
                                                    "ckpt_sha256": pl.core.ckpt_sha}}) + "\n")
    proto.flush()
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        rec = TP.MC.load_service_record(q["svc"])
        ans = pl.step(int(q["t"]), rec)
        ans["pick_greedy_rule"] = ans["pick"]
        ans["pick"], ans["fallback"], ans["nv_candidates"] = NV.nv_choice(ans["scores"], rec, f"T2|{a.group}|{a.tag}", int(q["t"]))
        if ans.get("refresh"):
            ans["refresh"]["region_box_m"] = ans["refresh"].get("box_m")
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
