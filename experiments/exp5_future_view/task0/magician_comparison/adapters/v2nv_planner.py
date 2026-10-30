"""EXP5 Task 0 / T0.10 ablation A1 "V2K4-NV" (seen2scene env; planner side): the frozen T0.6 V2K4 planner
(scripts/t06_planner.py, V2 K = 4 + DistP1, R10, unchanged scores) with MAGICIAN's validity rule added to the choice:
candidate set C = the legal poses NOT executed before (service-record history) if any, else all legal poses;
pick = argmax DistP1 over C (ties -> lowest index); max over C <= 0 -> seeded uniform over C
(sha256 "exp5-T0.10|NV|<tag>|<t>"). A separately named ablation, not a change of V2K4 (protocol v1.2 §12).
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import t06_planner as TP  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def nv_choice(scores, rec, tag, t):
    hist = {tuple(int(x) for x in r) for r in rec["executed_pose_idx_history"][:-1]}     # poses executed before now
    legal = [tuple(int(x) for x in r) for r in rec["legal_idx"]]
    if not legal:
        return None, False, []
    C = [j for j, p in enumerate(legal) if p not in hist] or list(range(len(legal)))
    s = np.asarray(scores, float)[C]
    if s.max() <= 0:
        rng = np.random.default_rng(int(hashlib.sha256(f"exp5-T0.10|NV|{tag}|{t}".encode()).hexdigest()[:15], 16))
        return C[int(rng.integers(len(C)))], True, C
    return C[int(np.flatnonzero(s == s.max())[0])], False, C


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["V2K4-NV"], required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda")
    pl = TP.Planner("V2K4", a.card, a.tag, dev)
    g = TP.Guard().install()
    proto.write(json.dumps({"ready": True, "info": {"base": "t06_planner V2K4", "rule": "prefer unvisited (MAGICIAN get_valid_neighbors)"}}) + "\n")
    proto.flush()
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        rec = TP.MC.load_service_record(q["svc"])
        ans = pl.step(int(q["t"]), rec)
        ans["pick_v2k4_rule"] = ans["pick"]
        ans["pick"], ans["fallback"], ans["nv_candidates"] = nv_choice(ans["scores"], rec, a.tag, int(q["t"]))
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
