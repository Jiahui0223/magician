"""G0-1: the V5 evaluation harness (eval_g1_g2.py with v5_core installed) replays V4 exactly: V4 final.pt evaluated as
the V4C arm on the first VALSEL and Dev-3 B0 episodes must give the same per-candidate DistP1 scores (bitwise) and the
same picks as the exp6_v4 V4 rows (work_dirs/exp6_v4/eval_fixed/<set>_s_final_v4); B2 replay rows equal as well.
Also reports the V5 smoke-checkpoint belief dump (shapes, finite). Output work_dirs/exp6_v5/eval_fixed/g0_replay_check.json"""
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W5, W4 = f"{R}/work_dirs/exp6_v5/eval_fixed", f"{R}/work_dirs/exp6_v4/eval_fixed"


def rows(root):
    out = {}
    for f in glob.glob(os.path.join(root, "*", "*.jsonl")):
        for x in open(f):
            r = json.loads(x)
            out[r["sid"]] = r
    return out


res = {}
for sset in ("valsel", "dev3"):
    a, b = rows(f"{W5}/{sset}_g0_v4replay"), rows(f"{W4}/{sset}_s_final_v4")
    common = sorted(set(a) & set(b))
    eq = [a[s]["scores"]["V4C"] == b[s]["scores"]["V4"] for s in common]
    eqb = [a[s]["scores"]["B2"] == b[s]["scores"]["B2"] for s in common]
    pick = [int(np.argmax(np.mean(np.asarray(a[s]["scores"]["V4C"], float).reshape(len(a[s]["scores"]["V4C"]), -1), 1))) ==
            int(np.argmax(np.mean(np.asarray(b[s]["scores"]["V4"], float).reshape(len(b[s]["scores"]["V4"]), -1), 1))) for s in common]
    res[sset] = {"states": len(common), "v4_scores_bitwise": int(sum(eq)), "b2_scores_bitwise": int(sum(eqb)), "same_pick": int(sum(pick))}
    bf = glob.glob(f"{W5}/{sset}_g0_v5init/*/belief_*.npz")
    if bf:
        z = np.load(bf[0])
        res[sset]["v5init_belief_dump"] = {"file": os.path.relpath(bf[0], R), "views": int(len(z["sid"])), "z": list(z["z"].shape),
                                           "finite": bool(np.isfinite(z["z"].astype(np.float32)).all()), "unknown_rays_sel_mean": float(z["n_unk_sel"].mean())}
    v5 = rows(f"{W5}/{sset}_g0_v5init")
    res[sset]["v5init_rows"] = len(v5)
    res[sset]["v5init_scores_finite"] = bool(all(np.isfinite(np.asarray(r["scores"]["V5"], float)).all() for r in v5.values())) if v5 else None
res["status"] = "PASS" if all(res[s]["states"] > 0 and res[s]["v4_scores_bitwise"] == res[s]["states"] == res[s]["b2_scores_bitwise"] for s in ("valsel", "dev3")) else "CHECK"
print(json.dumps(res, indent=1))
json.dump(res, open(f"{W5}/g0_replay_check.json", "w"), indent=1)
