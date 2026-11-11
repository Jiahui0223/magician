"""EXP7 -- verify the MP3D pose tables (CPU): (1) R, T, C of every legal pose in the stored service records of the frozen
V3-T0 indoor episodes (work_dirs/exp6_v3/eval/closed_loop/indoor/T0/V2M/<tag>/svc/svc_t*.npz) equal the table bitwise;
(2) h_policy.neighbours (used by the D1 lookahead) == grid_env.neighbours (official, environment) for every pose of the table."""
import glob
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "H", "scripts"))
import grid_env as GE  # noqa: E402
import h_policy as HP  # noqa: E402

PT = os.path.join(REPO, "work_dirs", "cvpr_results", "exp7", "pose_tables")
EP = os.path.join(REPO, "work_dirs", "exp6_v3", "eval", "closed_loop", "indoor", "T0", "V2M")
CARDS = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "scenes")
ok_all = True
for f in sorted(glob.glob(os.path.join(PT, "*.npz"))):
    key = os.path.basename(f)[:-4]
    z = np.load(f)
    lut = {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}
    grid = json.loads(str(z["grid"]))
    g = GE.grid_def(json.load(open(os.path.join(CARDS, f"{key}.json"))))
    n_rec = n_pose = bad = 0
    for d in sorted(glob.glob(os.path.join(EP, f"{key}_S*"))):
        for s in sorted(glob.glob(os.path.join(d, "svc", "svc_t*.npz")))[::5]:
            r = np.load(s)
            n_rec += 1
            for i, a in enumerate(r["legal_idx"]):
                j = lut[tuple(int(x) for x in a)]
                n_pose += 1
                if not (np.array_equal(z["R"][j], r["legal_R"][i]) and np.array_equal(z["T"][j], r["legal_T"][i]) and np.array_equal(z["C"][j], r["legal_center"][i])):
                    bad += 1
    rng = np.random.default_rng(0)
    sel = rng.choice(len(z["idx"]), size=min(20000, len(z["idx"])), replace=False)
    nb_bad = 0
    for j in sel:
        a = tuple(int(x) for x in z["idx"][j])
        hp = {tuple(int(x) for x in q) for q in HP.neighbours(a, grid)} - {a}
        ge = set(GE.neighbours(g, a))
        nb_bad += hp != ge
    ok = bad == 0 and nb_bad == 0 and n_pose > 0
    ok_all &= ok
    print(f"{key}: legal poses checked {n_pose} in {n_rec} records, R/T/C mismatches {bad} | neighbour sets checked {len(sel)}, mismatches {nb_bad} -> {'PASS' if ok else 'FAIL'}")
print("POSE TABLES", "PASS" if ok_all else "FAIL")
